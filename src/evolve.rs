//! The update loop around a ladder: what the upper rungs answer becomes training data for the
//! lower ones — but a candidate ladder replaces the current one only if it is no worse on rows a
//! person verified.
//!
//! The hazard this module is shaped around: a wrong upper-rung answer, trained into the lexical
//! rung, comes back as a CONFIDENT lexical answer — it stops escalating, so the error gets
//! cheaper and invisible. Two defences:
//!
//! - harvest only what the lower rungs passed up (the lexical rung never trains on itself);
//! - promote on verified rows, and count "silent errors" (wrong answers from a rung below the
//!   top, which nothing above ever sees) as a separate no-regression condition.

use crate::heimdall::{Chat, Embed};
use crate::ladder::{Answer, Example, Ladder};
use serde::Serialize;

/// One upper-rung answer, kept as a teacher label.
#[derive(Clone, Debug, Serialize)]
pub struct Harvest {
    pub text: String,
    pub label: String,
    pub rung: &'static str,
}

/// Run `ladder` over unlabeled `texts`; keep the answers that came from a rung above `lexical`.
pub fn harvest(ladder: &Ladder, texts: &[String], embed: Option<&dyn Embed>, chat: Option<&dyn Chat>) -> Vec<Harvest> {
    texts
        .iter()
        .filter_map(|t| {
            let a = ladder.decide(t, embed, chat);
            match (a.label, a.rung) {
                (Some(label), Some(rung)) if rung != "lexical" => Some(Harvest { text: t.clone(), label, rung }),
                _ => None,
            }
        })
        .collect()
}

/// A ladder's record on labeled rows.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Score {
    pub n: usize,
    pub answered: usize,
    pub right: usize,
    /// Answers per rung.
    pub by_rung: Vec<(String, usize)>,
    /// Wrong answers from any rung below the top one present (nothing above saw them).
    pub silent_errors: usize,
    /// Wrong answers from the lexical rung alone.
    pub lexical_errors: usize,
    /// Rows that reached the llm rung (its calls).
    pub llm_calls: usize,
}

pub fn score(ladder: &Ladder, rows: &[Example], embed: Option<&dyn Embed>, chat: Option<&dyn Chat>) -> (Score, Vec<Answer>) {
    let top = if ladder.llm.is_some() {
        "llm"
    } else if ladder.encoder.is_some() {
        "encoder"
    } else {
        "lexical"
    };
    let mut s = Score { n: rows.len(), ..Score::default() };
    let mut answers = Vec::with_capacity(rows.len());
    let mut by_rung = std::collections::BTreeMap::<String, usize>::new();
    for r in rows {
        let a = ladder.decide(&r.text, embed, chat);
        s.llm_calls += a.steps.iter().any(|st| st.rung == "llm") as usize;
        if let (Some(l), Some(rung)) = (&a.label, a.rung) {
            s.answered += 1;
            *by_rung.entry(rung.to_string()).or_default() += 1;
            let ok = *l == r.label;
            s.right += ok as usize;
            if !ok && rung != top {
                s.silent_errors += 1;
            }
            if !ok && rung == "lexical" {
                s.lexical_errors += 1;
            }
        }
        answers.push(a);
    }
    s.by_rung = by_rung.into_iter().collect();
    (s, answers)
}

/// The promotion verdict on verified rows.
#[derive(Clone, Debug, Serialize)]
pub struct Promotion {
    pub current: Score,
    pub candidate: Score,
    pub promoted: bool,
    pub reason: String,
}

/// Promote iff the candidate is right at least as often AND makes no more silent errors on the
/// verified rows. Ties go to the candidate only when both hold — never on accuracy alone.
pub fn compare(current: &Ladder, candidate: &Ladder, verified: &[Example], embed: Option<&dyn Embed>, chat: Option<&dyn Chat>) -> Promotion {
    let (cur, _) = score(current, verified, embed, chat);
    let (cand, _) = score(candidate, verified, embed, chat);
    let (promoted, reason) = if cand.right < cur.right {
        (false, format!("right {} < current {}", cand.right, cur.right))
    } else if cand.silent_errors > cur.silent_errors {
        (false, format!("silent errors {} > current {}", cand.silent_errors, cur.silent_errors))
    } else {
        (true, format!("right {} ≥ {} and silent errors {} ≤ {}", cand.right, cur.right, cand.silent_errors, cur.silent_errors))
    };
    Promotion { current: cur, candidate: cand, promoted, reason }
}

/// Every rung's own answer for one input, computed regardless of the gates — the material of
/// the 2-of-3 vote.
#[derive(Clone, Debug, Serialize)]
pub struct Votes {
    /// What the ladder itself answered, and from which rung.
    pub answer: Option<String>,
    pub rung: Option<&'static str>,
    pub confidence: Option<f32>,
    pub lexical: String,
    pub lexical_confidence: f32,
    pub encoder: Option<String>,
    pub llm: Option<String>,
}

pub fn votes(ladder: &Ladder, text: &str, embed: Option<&dyn Embed>, chat: Option<&dyn Chat>) -> Votes {
    let a = ladder.decide(text, embed, chat);
    let (lc, ls) = ladder.lexical.top(&ladder.lexical_x(text));
    let encoder = match (&ladder.encoder, embed) {
        (Some(enc), Some(em)) => em
            .embed(&[text.to_string()])
            .ok()
            .map(|v| ladder.labels[enc.head.top(&crate::features::dense(&v[0])).0].clone()),
        _ => None,
    };
    let llm = chat.and_then(|c| c.complete(&ladder.llm_prompt(), text).ok()).and_then(|r| ladder.parse_llm(&r));
    Votes {
        answer: a.label,
        rung: a.rung,
        confidence: a.confidence,
        lexical: ladder.labels[lc].clone(),
        lexical_confidence: ls,
        encoder,
        llm,
    }
}

/// 2-of-3 over (lexical, encoder, llm); no majority → the llm's label, else the ladder's own.
pub fn majority(v: &Votes) -> Option<String> {
    let all = [Some(&v.lexical), v.encoder.as_ref(), v.llm.as_ref()];
    for cand in all.iter().flatten() {
        if all.iter().flatten().filter(|x| **x == *cand).count() >= 2 {
            return Some((*cand).clone());
        }
    }
    v.llm.clone().or_else(|| v.answer.clone())
}

/// The 2-of-3 vote over the least-confident `share` of lower-rung answers (lexical, memory,
/// encoder): `(right, fixed, broke, silent errors left, extra llm calls)`.
pub fn vote_band(v: &[Votes], gold: &[Example], share: f64) -> (usize, usize, usize, usize, usize) {
    let lower_rung = |r: Option<&str>| matches!(r, Some("lexical") | Some("memory") | Some("encoder"));
    let mut lower: Vec<usize> = (0..v.len()).filter(|&i| lower_rung(v[i].rung)).collect();
    lower.sort_by(|&a, &b| v[a].confidence.unwrap_or(0.0).total_cmp(&v[b].confidence.unwrap_or(0.0)));
    let k = (lower.len() as f64 * share).round() as usize;
    let voted: std::collections::HashSet<usize> = lower[..k].iter().copied().collect();
    let (mut right, mut fixed, mut broke, mut silent) = (0, 0, 0, 0);
    for (i, (x, g)) in v.iter().zip(gold).enumerate() {
        let fin = if voted.contains(&i) { majority(x) } else { x.answer.clone() };
        let (was, now) = (x.answer.as_deref() == Some(&g.label), fin.as_deref() == Some(&g.label));
        right += now as usize;
        fixed += (now && !was) as usize;
        broke += (was && !now) as usize;
        silent += (!now && lower_rung(x.rung) && fin == x.answer) as usize;
    }
    (right, fixed, broke, silent, k)
}

/// Mann–Whitney AUC of `pos` over `neg` (NaN when either side is empty).
pub fn auc(pos: &[f32], neg: &[f32]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let w: f64 = pos.iter().flat_map(|p| neg.iter().map(move |n| if p > n { 1.0 } else if p == n { 0.5 } else { 0.0 })).sum();
    w / (pos.len() * neg.len()) as f64
}

pub const VOTE_SHARES: [f64; 6] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];

/// One ladder read on test rows: as served, with the 2-of-3 vote at the share chosen on the
/// VERIFIED rows (ties → the smaller share), and the two independence signals.
#[derive(Clone, Debug, Serialize)]
pub struct Reading {
    pub served: Score,
    pub vote_share: f64,
    /// `(right, fixed, broke, silent errors left, extra llm calls)` on test at `vote_share`.
    pub vote: (usize, usize, usize, usize, usize),
    pub lexical_conf_auc: f64,
    /// Of the lexical rung's wrong answers on test, how many the encoder head repeats.
    pub encoder_repeats_lexical_errors: (usize, usize),
}

pub fn read(ladder: &Ladder, verified: &[Example], test: &[Example], embed: Option<&dyn Embed>, chat: Option<&dyn Chat>) -> Reading {
    let (served, _) = score(ladder, test, embed, chat);
    let vd: Vec<Votes> = verified.iter().map(|r| votes(ladder, &r.text, embed, chat)).collect();
    let vt: Vec<Votes> = test.iter().map(|r| votes(ladder, &r.text, embed, chat)).collect();
    let vote_share = VOTE_SHARES
        .iter()
        .copied()
        .max_by(|x, y| vote_band(&vd, verified, *x).0.cmp(&vote_band(&vd, verified, *y).0).then(y.total_cmp(x)))
        .expect("shares");
    let lex: Vec<(f32, bool)> = vt.iter().zip(test).filter(|(v, _)| v.rung == Some("lexical")).map(|(v, g)| (v.confidence.unwrap_or(0.0), v.answer.as_deref() == Some(&g.label))).collect();
    let ok: Vec<f32> = lex.iter().filter(|x| x.1).map(|x| x.0).collect();
    let bad: Vec<f32> = lex.iter().filter(|x| !x.1).map(|x| x.0).collect();
    let errs: Vec<&Votes> = vt.iter().zip(test).filter(|(v, g)| v.rung == Some("lexical") && v.answer.as_deref() != Some(&g.label)).map(|(v, _)| v).collect();
    Reading {
        vote: vote_band(&vt, test, vote_share),
        vote_share,
        lexical_conf_auc: auc(&ok, &bad),
        encoder_repeats_lexical_errors: (errs.iter().filter(|v| v.encoder == v.answer).count(), errs.len()),
        served,
    }
}

impl Reading {
    pub fn line(&self) -> String {
        let s = &self.served;
        let (right, fixed, broke, silent_left, calls) = self.vote;
        format!(
            "served {}/{} · by rung {:?} · silent {} (lexical {}) · llm calls {} │ vote share {} +{calls} → {right}/{} fixed {fixed} broke {broke} silent left {silent_left} │ lexical AUC {:.2} · encoder repeats lexical errors {}/{}",
            s.right, s.n, s.by_rung, s.silent_errors, s.lexical_errors, s.llm_calls, self.vote_share, s.n,
            self.lexical_conf_auc, self.encoder_repeats_lexical_errors.0, self.encoder_repeats_lexical_errors.1
        )
    }
}
