//! The decision ladder: each rung answers only what the rung below was not sure about, and
//! every answer names the rung that gave it.
//!
//! | rung | what | cost |
//! |---|---|---|
//! | `lexical` | one-vs-all logistic over Thai-aware n-gram bags, trained on labels | µs, CPU, no model |
//! | `encoder` | the same head over bge-m3 embeddings (local Heimdall) | ~10–20 ms |
//! | `llm` | the local chat model picks a label or says UNSURE | seconds |
//!
//! A rung's threshold is fitted, never guessed: out-of-fold scores on the training rows, the
//! LOWEST cutoff whose at-or-above accuracy meets `target_accuracy` (fewer than 16 rows, or a
//! target no cutoff meets → the rung answers nothing, and the fit report says why). The encoder
//! is fitted on the rows the lexical rung would have passed up, when there are enough of them.
//! The llm rung has no threshold: it answers when it names a valid label.
//!
//! A trained head only knows its own labels — it will confidently file a cooking question
//! under "billing". So the lexical and encoder rungs also carry a novelty gate: the input's
//! cosine to its nearest training row must reach the `gate_rho` quantile of the same
//! nearest-neighbour cosine measured out-of-fold on the training rows (no off-task data
//! needed; ρ = 0.05 lets ~5% of in-distribution inputs through to the next rung). Below it,
//! the rung passes the input up no matter how confident it is.

use crate::features::{self, Sparse};
use crate::heimdall::{Chat, Embed};
use crate::linear::{Linear, TrainConfig};
use crate::threshold::{self, Recommendation, THIN_SUPPORT_FLOOR};
use crate::tokenize::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Example {
    pub text: String,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct LadderConfig {
    pub target_accuracy: f64,
    pub folds: usize,
    pub tokenizer: Tokenizer,
    pub train: TrainConfig,
    /// Novelty-gate quantile (see the module doc).
    pub gate_rho: f64,
    /// How the lexical rung is trained (see [`LexicalKind`]).
    pub lexical: LexicalKind,
}

/// The lexical rung's recipe — `bag`, `nbsvm`, `nbsvm-distill:<mix>` or `auto` ("Ultra
/// Instinct": switch from `bag` only when the best challenger's paired LB95 is above 0).
/// Lives in the ultra-instinct crate with the nested-teacher fit it drives.
pub use ultra_instinct::Recipe as LexicalKind;

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            target_accuracy: 0.9,
            folds: 4,
            tokenizer: Tokenizer::Unicode,
            train: TrainConfig::default(),
            gate_rho: 0.05,
            lexical: LexicalKind::Bag,
        }
    }
}

/// How a rung's threshold was chosen.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RungFit {
    /// The cutoff on the rung's top score; `None` = this rung never answers.
    pub threshold: Option<f32>,
    /// Out-of-fold rows the fit saw, and which rows they were.
    pub n: usize,
    pub basis: String,
    /// The target-accuracy recommendation (met or best-achievable), when there was support.
    pub recommendation: Option<Recommendation>,
    /// Out-of-fold forced-pick accuracy of the rung on the same rows.
    pub oof_accuracy: f64,
}

fn bag_features() -> String {
    "bag".into()
}

fn is_bag(s: &String) -> bool {
    s == "bag"
}

/// What the `auto` recipe saw and chose; stored in the ladder file.
pub use ultra_instinct::Choice as LexicalChoice;

/// Novelty gate: the training rows (L2-normalized) and the cosine cutoff to the nearest one.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gate {
    /// `None` = too few rows to fit (fewer than 16): the rung never answers.
    pub cutoff: Option<f32>,
    pub rho: f64,
    pub rows: Vec<Sparse>,
}

/// Dot product of two id-sorted sparse rows.
fn sparse_dot(a: &Sparse, b: &Sparse) -> f32 {
    let (mut i, mut j, mut s) = (0, 0, 0.0f32);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                s += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    s
}

impl Gate {
    fn fit(rows: &[Sparse], folds: usize, rho: f64) -> Self {
        let mut sims: Vec<f32> = (0..rows.len())
            .map(|i| {
                (0..rows.len())
                    .filter(|j| j % folds != i % folds)
                    .map(|j| sparse_dot(&rows[i], &rows[j]))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        sims.sort_by(f32::total_cmp);
        let cutoff = (sims.len() >= THIN_SUPPORT_FLOOR).then(|| sims[((sims.len() as f64 * rho) as usize).min(sims.len() - 1)]);
        Self { cutoff, rho, rows: rows.to_vec() }
    }

    /// `(nearest-row cosine, inside the gate)`.
    pub fn check(&self, x: &Sparse) -> (f32, bool) {
        let sim = self.rows.iter().map(|r| sparse_dot(r, x)).fold(f32::NEG_INFINITY, f32::max);
        (sim, self.cutoff.is_some_and(|c| sim >= c))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderRung {
    pub model: String,
    pub head: Linear,
    pub fit: RungFit,
    pub gate: Gate,
    /// Verified examples answered by nearest neighbour before the head (see [`Memory`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<Memory>,
}

/// Verified examples kept as-is in the encoder's space: an input whose nearest example is close
/// enough takes that example's label — no retraining, effective on the next call, removable row
/// by row, and the answer can name the example it came from. The cutoff is fitted by
/// leave-one-out over the memory itself (the lowest nearest-neighbour cosine whose at-or-above
/// accuracy meets the ladder's target); fewer than 16 rows → no cutoff, memory never answers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Memory {
    pub texts: Vec<String>,
    pub rows: Vec<Sparse>,
    pub labels: Vec<usize>,
    pub cutoff: Option<f32>,
    pub fit: Option<Recommendation>,
}

impl Memory {
    /// `(index, cosine)` of the nearest example, skipping `skip`.
    fn nearest(&self, x: &Sparse, skip: Option<usize>) -> Option<(usize, f32)> {
        (0..self.rows.len())
            .filter(|i| Some(*i) != skip)
            .map(|i| (i, sparse_dot(&self.rows[i], x)))
            .fold(None, |best: Option<(usize, f32)>, (i, s)| match best {
                Some((_, b)) if b >= s => best,
                _ => Some((i, s)),
            })
    }

    fn refit(&mut self, target: f64) {
        let obs: Vec<(f32, bool)> = (0..self.rows.len())
            .filter_map(|i| self.nearest(&self.rows[i], Some(i)).map(|(j, s)| (s, self.labels[j] == self.labels[i])))
            .collect();
        self.fit = threshold::target_accuracy(&obs, target);
        self.cutoff = self.fit.as_ref().filter(|r| r.met == Some(true)).map(|r| r.threshold);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ladder {
    pub task: String,
    pub labels: Vec<String>,
    /// Optional one-line meaning per label (the llm rung's prompt).
    pub descriptions: BTreeMap<String, String>,
    pub tokenizer: String,
    pub target_accuracy: f64,
    pub trained_on: usize,
    /// BLAKE3 of the training rows (text \t label \n, in order).
    pub training_digest: String,
    pub lexical: Linear,
    pub lexical_fit: RungFit,
    pub lexical_gate: Gate,
    /// Input the lexical head scores: "bag" (weighted, L2) or "presence" (NBSVM). The novelty
    /// gate always reads the bag. Absent in files written before the field existed = "bag".
    #[serde(default = "bag_features", skip_serializing_if = "is_bag")]
    pub lexical_features: String,
    /// The `Auto` selection record (candidates, out-of-fold accuracy, the gate's verdict).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_choice: Option<LexicalChoice>,
    pub encoder: Option<EncoderRung>,
    /// The chat model of the llm rung; `None` = no llm rung.
    pub llm: Option<String>,
    /// BLAKE3 of the ladder file's bytes (`save` writes exactly what `fit` hashed).
    #[serde(skip)]
    digest: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub rung: &'static str,
    pub label: Option<String>,
    pub confidence: Option<f32>,
    pub threshold: Option<f32>,
    /// Cosine to the nearest training row, and the gate's cutoff.
    pub similarity: Option<f32>,
    pub gate: Option<f32>,
    pub passed: bool,
    pub micros: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Receipt {
    pub ladder: String,
    pub input: String,
    pub decision: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    pub task: String,
    /// `None` = every rung abstained.
    pub label: Option<String>,
    pub rung: Option<&'static str>,
    pub confidence: Option<f32>,
    pub steps: Vec<Step>,
    pub receipt: Receipt,
}

fn hex(b: &[u8]) -> String {
    blake3::hash(b).to_hex().to_string()
}

/// Out-of-fold `(top class, top score)` per row: fold k = rows with `i % folds == k`.
fn out_of_fold(rows: &[Sparse], y: &[usize], labels: &[String], cfg: &LadderConfig) -> Vec<(usize, f32)> {
    let mut out = vec![(0, 0.0); rows.len()];
    for k in 0..cfg.folds {
        let tr: Vec<usize> = (0..rows.len()).filter(|i| i % cfg.folds != k).collect();
        let m = Linear::train(
            &tr.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>(),
            &tr.iter().map(|&i| y[i]).collect::<Vec<_>>(),
            labels,
            &cfg.train,
        );
        for i in (0..rows.len()).filter(|i| i % cfg.folds == k) {
            out[i] = m.top(&rows[i]);
        }
    }
    out
}

fn fit_threshold(oof: &[(usize, f32)], y: &[usize], rows: &[usize], basis: String, target: f64) -> RungFit {
    let obs: Vec<(f32, bool)> = rows.iter().map(|&i| (oof[i].1, oof[i].0 == y[i])).collect();
    let rec = threshold::target_accuracy(&obs, target);
    let all_right = oof.iter().zip(y).filter(|((p, _), g)| p == *g).count();
    RungFit {
        threshold: rec.as_ref().filter(|r| r.met == Some(true)).map(|r| r.threshold),
        n: rows.len(),
        basis: if rows.len() < THIN_SUPPORT_FLOOR {
            format!("{basis} — fewer than {THIN_SUPPORT_FLOOR} rows, no threshold")
        } else {
            basis
        },
        recommendation: rec,
        oof_accuracy: all_right as f64 / y.len() as f64,
    }
}

impl Ladder {
    /// Train every rung and fit its threshold. `embed` = `None` → no encoder rung;
    /// `llm_model` = `None` → no llm rung (the chat model is only named here, never called).
    pub fn fit(
        task: &str,
        examples: &[Example],
        descriptions: BTreeMap<String, String>,
        cfg: &LadderConfig,
        embed: Option<&dyn Embed>,
        llm_model: Option<&str>,
    ) -> Result<Self, String> {
        Self::fit_split(task, examples, examples, descriptions, cfg, embed, llm_model)
    }

    /// [`Self::fit`] with a separate training set per trained rung (tri-training: each rung
    /// learns from rows the OTHER rungs agreed on). With one shared set, the encoder's threshold
    /// is fitted on the rows the lexical rung passes up; with two sets there is no row-for-row
    /// residual, so it is fitted on all of the encoder's rows.
    pub fn fit_split(
        task: &str,
        lexical_examples: &[Example],
        encoder_examples: &[Example],
        descriptions: BTreeMap<String, String>,
        cfg: &LadderConfig,
        embed: Option<&dyn Embed>,
        llm_model: Option<&str>,
    ) -> Result<Self, String> {
        let shared = std::ptr::eq(lexical_examples, encoder_examples);
        for set in [lexical_examples, encoder_examples] {
            if set.len() < cfg.folds * 2 {
                return Err(format!("{} examples is too few to fit {} folds", set.len(), cfg.folds));
            }
        }
        let mut labels: Vec<String> = lexical_examples.iter().chain(encoder_examples).map(|e| e.label.clone()).collect();
        labels.sort();
        labels.dedup();
        if labels.len() < 2 {
            return Err("a decision needs at least 2 labels".into());
        }
        let ys = |set: &[Example]| -> Vec<usize> { set.iter().map(|e| labels.binary_search(&e.label).unwrap()).collect() };
        let y = ys(lexical_examples);
        let all: Vec<usize> = (0..lexical_examples.len()).collect();

        let lex_rows: Vec<Sparse> = lexical_examples.iter().map(|e| features::lexical(cfg.tokenizer, &e.text)).collect();
        let lexical_gate = Gate::fit(&lex_rows, cfg.folds, cfg.gate_rho);
        let (lexical, lex_oof, lexical_features, lexical_choice) =
            Self::fit_lexical(lexical_examples, &lex_rows, &y, &labels, cfg, embed)?;
        // The recipe is named in the basis only when it is not the plain bag, so Bag ladders keep
        // their pre-Instinct bytes (and digests).
        let recipe = lexical_choice.as_ref().map_or(cfg.lexical.name(), |c| c.chosen.clone());
        let basis = if recipe == "bag" && lexical_choice.is_none() {
            format!("{}-fold out-of-fold, all rows", cfg.folds)
        } else {
            format!("{}-fold out-of-fold, all rows · {recipe}", cfg.folds)
        };
        let lexical_fit = fit_threshold(&lex_oof, &y, &all, basis, cfg.target_accuracy);

        let encoder = match embed {
            None => None,
            Some(em) => {
                let ye = ys(encoder_examples);
                let texts: Vec<String> = encoder_examples.iter().map(|e| e.text.clone()).collect();
                let rows: Vec<Sparse> = em.embed(&texts)?.iter().map(|v| features::dense(v)).collect();
                let oof = out_of_fold(&rows, &ye, &labels, cfg);
                let every: Vec<usize> = (0..encoder_examples.len()).collect();
                // Fit where the encoder will actually work: the rows the lexical rung passes up.
                let (rows_for_fit, basis) = if !shared {
                    (every, format!("{}-fold out-of-fold, all of its own rows (separate training set)", cfg.folds))
                } else {
                    let residual: Vec<usize> = every
                        .iter()
                        .copied()
                        .filter(|&i| lexical_fit.threshold.is_none_or(|t| lex_oof[i].1 < t))
                        .collect();
                    if residual.len() >= THIN_SUPPORT_FLOOR {
                        (residual, format!("{}-fold out-of-fold, rows the lexical rung passes up", cfg.folds))
                    } else {
                        (every, format!("{}-fold out-of-fold, all rows (only {} passed up — too few)", cfg.folds, residual.len()))
                    }
                };
                let fit = fit_threshold(&oof, &ye, &rows_for_fit, basis, cfg.target_accuracy);
                Some(EncoderRung {
                    model: em.model().to_string(),
                    head: Linear::train(&rows, &ye, &labels, &cfg.train),
                    fit,
                    gate: Gate::fit(&rows, cfg.folds, cfg.gate_rho),
                    memory: None,
                })
            }
        };

        let mut digest_input = Vec::new();
        for e in lexical_examples {
            digest_input.extend_from_slice(format!("{}\t{}\n", e.text, e.label).as_bytes());
        }
        if !shared {
            digest_input.extend_from_slice(b"--encoder--\n");
            for e in encoder_examples {
                digest_input.extend_from_slice(format!("{}\t{}\n", e.text, e.label).as_bytes());
            }
        }
        let mut ladder = Self {
            task: task.into(),
            labels,
            descriptions,
            tokenizer: cfg.tokenizer.as_str().into(),
            target_accuracy: cfg.target_accuracy,
            trained_on: if shared { lexical_examples.len() } else { lexical_examples.len() + encoder_examples.len() },
            training_digest: hex(&digest_input),
            lexical,
            lexical_fit,
            lexical_gate,
            lexical_features,
            lexical_choice,
            encoder,
            llm: llm_model.map(str::to_string),
            digest: String::new(),
        };
        ladder.digest = hex(&ladder.to_bytes());
        Ok(ladder)
    }

    /// Train the lexical rung per `cfg.lexical`; returns the model, its out-of-fold
    /// `(top, score)`, the feature kind it scores, and the `Auto` record.
    #[allow(clippy::type_complexity)]
    fn fit_lexical(
        examples: &[Example],
        bag_rows: &[Sparse],
        y: &[usize],
        labels: &[String],
        cfg: &LadderConfig,
        embed: Option<&dyn Embed>,
    ) -> Result<(Linear, Vec<(usize, f32)>, String, Option<LexicalChoice>), String> {
        let pres: Vec<Sparse> = examples.iter().map(|e| features::presence(cfg.tokenizer, &e.text)).collect();
        // The distillation teacher reads the encoder's view of the same rows; ultra-instinct
        // refits it out-of-fold inside every fold (a teacher fitted across all rows leaks).
        let enc_rows: Option<Vec<Sparse>> = match embed {
            Some(em) if cfg.lexical.wants_teacher() => {
                let texts: Vec<String> = examples.iter().map(|e| e.text.clone()).collect();
                Some(em.embed(&texts)?.iter().map(|v| features::dense(v)).collect())
            }
            _ => None,
        };
        let rows = ultra_instinct::Rows { bag: bag_rows, presence: &pres, teacher: enc_rows.as_deref() };
        let f = ultra_instinct::fit(&rows, y, labels, cfg.lexical, cfg.folds, &cfg.train)?;
        Ok((f.model, f.oof, f.features, f.choice))
    }

    /// The lexical head's input for `text` (bag or presence, as the ladder was trained).
    pub fn lexical_x(&self, text: &str) -> Sparse {
        let tok = Tokenizer::parse(&self.tokenizer).unwrap_or(Tokenizer::Unicode);
        if self.lexical_features == "presence" { features::presence(tok, text) } else { features::lexical(tok, text) }
    }

    /// Refit the lexical and encoder thresholds on `verified` rows instead of out-of-fold
    /// training rows. Training rows can be a biased sample — rows two rungs agreed on are mostly
    /// the easy ones — and a threshold fitted on easy rows lets a rung answer hard inputs it gets
    /// wrong. The heads are untouched. The encoder is fitted on the verified rows the (new)
    /// lexical rung passes up when there are at least 16 of them, else on all of them.
    pub fn recalibrate(&mut self, verified: &[Example], embed: Option<&dyn Embed>) -> Result<(), String> {
        let y: Vec<usize> = verified
            .iter()
            .map(|e| self.labels.binary_search(&e.label).map_err(|_| format!("label {:?} is not one of this ladder's", e.label)))
            .collect::<Result<_, _>>()?;
        let lex: Vec<(usize, f32)> = verified.iter().map(|e| self.lexical.top(&self.lexical_x(&e.text))).collect();
        let all: Vec<usize> = (0..verified.len()).collect();
        let fit = fit_threshold(&lex, &y, &all, format!("verified rows ({})", verified.len()), self.target_accuracy);
        self.lexical_fit = fit;
        if let (Some(enc), Some(em)) = (self.encoder.as_mut(), embed) {
            let texts: Vec<String> = verified.iter().map(|e| e.text.clone()).collect();
            let scored: Vec<(usize, f32)> = em.embed(&texts)?.iter().map(|v| enc.head.top(&features::dense(v))).collect();
            let residual: Vec<usize> = all.iter().copied().filter(|&i| self.lexical_fit.threshold.is_none_or(|t| lex[i].1 < t)).collect();
            let (rows, basis) = if residual.len() >= THIN_SUPPORT_FLOOR {
                (residual, format!("verified rows the lexical rung passes up ({})", verified.len()))
            } else {
                (all.clone(), format!("all verified rows ({}; only {} passed up)", verified.len(), residual.len()))
            };
            enc.fit = fit_threshold(&scored, &y, &rows, basis, self.target_accuracy);
        }
        self.digest = hex(&self.to_bytes());
        Ok(())
    }

    /// Put verified examples into the encoder's memory and refit its cutoff — no head is
    /// retrained. Returns the memory size.
    pub fn remember(&mut self, examples: &[Example], embed: &dyn Embed) -> Result<usize, String> {
        let enc = self.encoder.as_mut().ok_or("remember needs an encoder rung")?;
        if embed.model() != enc.model {
            return Err(format!("encoder is {}, ladder was trained on {}", embed.model(), enc.model));
        }
        let texts: Vec<String> = examples.iter().map(|e| e.text.clone()).collect();
        let vecs = embed.embed(&texts)?;
        let mem = enc.memory.get_or_insert_with(Memory::default);
        for (e, v) in examples.iter().zip(vecs) {
            let label = self.labels.binary_search(&e.label).map_err(|_| format!("label {:?} is not one of this ladder's", e.label))?;
            mem.texts.push(e.text.clone());
            mem.rows.push(features::dense(&v));
            mem.labels.push(label);
        }
        mem.refit(self.target_accuracy);
        let n = mem.rows.len();
        self.digest = hex(&self.to_bytes());
        Ok(n)
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("ladder serializes")
    }

    /// BLAKE3 of the serialized ladder — the model identity a receipt names.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut ladder: Self = serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", path.display()))?;
        ladder.digest = hex(&b);
        Ok(ladder)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The llm rung's system prompt (labels + their one-line meanings).
    pub fn llm_prompt(&self) -> String {
        let mut s = format!(
            "Classify the user's message for the task \"{}\". Answer with exactly one label from this list, or UNSURE if none clearly fits.\n",
            self.task
        );
        for l in &self.labels {
            match self.descriptions.get(l) {
                Some(d) => s.push_str(&format!("- {l}: {d}\n")),
                None => s.push_str(&format!("- {l}\n")),
            }
        }
        s.push_str("Reply with the label only.");
        s
    }

    /// The label the chat model named, if it named exactly one of ours (case-insensitive).
    pub fn parse_llm(&self, reply: &str) -> Option<String> {
        let first = reply.lines().find(|l| !l.trim().is_empty())?;
        let word = first.trim().trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
        self.labels.iter().find(|l| l.eq_ignore_ascii_case(word)).cloned()
    }

    /// Run the ladder. A rung whose client is absent (`None`) or fails is skipped with a note —
    /// never silently treated as an answer.
    pub fn decide(&self, text: &str, embed: Option<&dyn Embed>, llm: Option<&dyn Chat>) -> Answer {
        let tok = Tokenizer::parse(&self.tokenizer).unwrap_or(Tokenizer::Unicode);
        let mut steps = Vec::new();
        let t = Instant::now();
        let x = features::lexical(tok, text);
        let (c, s) = if self.lexical_features == "presence" { self.lexical.top(&features::presence(tok, text)) } else { self.lexical.top(&x) };
        let (sim, inside) = self.lexical_gate.check(&x);
        let confident = self.lexical_fit.threshold.is_some_and(|th| s >= th);
        let passed = confident && inside;
        steps.push(Step {
            rung: "lexical",
            label: Some(self.labels[c].clone()),
            confidence: Some(s),
            threshold: self.lexical_fit.threshold,
            similarity: Some(sim),
            gate: self.lexical_gate.cutoff,
            passed,
            micros: t.elapsed().as_micros() as u64,
            note: (confident && !inside).then(|| "confident but unlike the training rows — passed up".into()),
        });

        if !passed && let Some(enc) = &self.encoder {
            let t = Instant::now();
            let step = match embed {
                None => Step { rung: "encoder", label: None, confidence: None, threshold: enc.fit.threshold, similarity: None, gate: enc.gate.cutoff, passed: false, micros: 0, note: Some("no encoder client".into()) },
                Some(em) if em.model() != enc.model => Step {
                    rung: "encoder", label: None, confidence: None, threshold: enc.fit.threshold, similarity: None, gate: enc.gate.cutoff, passed: false, micros: 0,
                    note: Some(format!("encoder is {}, ladder was trained on {}", em.model(), enc.model)),
                },
                Some(em) => match em.embed(&[text.to_string()]) {
                    Err(e) => Step { rung: "encoder", label: None, confidence: None, threshold: enc.fit.threshold, similarity: None, gate: enc.gate.cutoff, passed: false, micros: t.elapsed().as_micros() as u64, note: Some(e) },
                    Ok(v) => {
                        let x = features::dense(&v[0]);
                        let recalled = enc.memory.as_ref().and_then(|m| {
                            let (j, sim) = m.nearest(&x, None)?;
                            m.cutoff.filter(|c| sim >= *c).map(|c| (m, j, sim, c))
                        });
                        if let Some((m, j, sim, cutoff)) = recalled {
                            // A verified example this close answers; the head is not consulted.
                            Step {
                                rung: "memory",
                                label: Some(self.labels[m.labels[j]].clone()),
                                confidence: Some(sim),
                                threshold: Some(cutoff),
                                similarity: Some(sim),
                                gate: None,
                                passed: true,
                                micros: t.elapsed().as_micros() as u64,
                                note: Some(format!("nearest verified example: {}", m.texts[j])),
                            }
                        } else {
                            let (c, s) = enc.head.top(&x);
                            let (sim, inside) = enc.gate.check(&x);
                            let confident = enc.fit.threshold.is_some_and(|th| s >= th);
                            Step {
                                rung: "encoder",
                                label: Some(self.labels[c].clone()),
                                confidence: Some(s),
                                threshold: enc.fit.threshold,
                                similarity: Some(sim),
                                gate: enc.gate.cutoff,
                                passed: confident && inside,
                                micros: t.elapsed().as_micros() as u64,
                                note: (confident && !inside).then(|| "confident but unlike the training rows — passed up".into()),
                            }
                        }
                    }
                },
            };
            steps.push(step);
        }

        if !steps.last().is_some_and(|s| s.passed) && let Some(model) = &self.llm {
            let t = Instant::now();
            let step = match llm {
                None => Step { rung: "llm", label: None, confidence: None, threshold: None, similarity: None, gate: None, passed: false, micros: 0, note: Some("no chat client".into()) },
                Some(ch) => match ch.complete(&self.llm_prompt(), text) {
                    Err(e) => Step { rung: "llm", label: None, confidence: None, threshold: None, similarity: None, gate: None, passed: false, micros: t.elapsed().as_micros() as u64, note: Some(e) },
                    Ok(reply) => {
                        let label = self.parse_llm(&reply);
                        Step {
                            rung: "llm",
                            passed: label.is_some(),
                            label,
                            confidence: None,
                            threshold: None,
                            similarity: None,
                            gate: None,
                            micros: t.elapsed().as_micros() as u64,
                            note: Some(format!("{model}: {}", reply.trim())),
                        }
                    }
                },
            };
            steps.push(step);
        }

        let winner = steps.iter().find(|s| s.passed);
        let (label, rung, confidence) = match winner {
            Some(s) => (s.label.clone(), Some(s.rung), s.confidence),
            None => (None, None, None),
        };
        let decision = format!("{}\t{}\t{}", self.task, label.as_deref().unwrap_or(""), rung.unwrap_or(""));
        Answer {
            task: self.task.clone(),
            receipt: Receipt {
                ladder: self.digest.clone(),
                input: hex(text.as_bytes()),
                decision: hex(decision.as_bytes()),
            },
            label,
            rung,
            confidence,
            steps,
        }
    }
}
