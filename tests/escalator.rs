//! The escalator (บันไดเลื่อน): conformal acceptance per rung, the llm rung's letter
//! distribution, the review outcome, the rate guard. Offline — fakes stand in for Heimdall.

use reflex_study::embed::Embedder;
use reflex_study::heimdall::{Chat, Embed};
use reflex_study::ladder::{Example, Ladder, LadderConfig, RateGuard, conformal_qhat, conformal_set};
use reflex_study::tokenize::Tokenizer;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};

struct FakeEmbed;
impl Embed for FakeEmbed {
    fn model(&self) -> &str {
        "fake-ngram-256"
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let e = Embedder::new(Tokenizer::Unicode, 256);
        Ok(texts.iter().map(|t| e.embed(t)).collect())
    }
}

/// Returns first-token log-probabilities over option letters: the gold letter at ~0.97 for the
/// texts it knows, an even split between two letters for `split`, and counts every call.
struct FakeLetters {
    gold: HashMap<String, char>,
    split: (String, char, char),
    calls: AtomicUsize,
    completes: AtomicUsize,
}
impl Chat for FakeLetters {
    fn model(&self) -> &str {
        "fake-letters"
    }
    fn complete(&self, _s: &str, _u: &str) -> Result<String, String> {
        self.completes.fetch_add(1, Ordering::SeqCst);
        Ok("UNSURE".into())
    }
    fn top_logprobs(&self, _s: &str, user: &str) -> Result<Vec<(String, f32)>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if user == self.split.0 {
            return Ok(vec![(self.split.1.to_string(), (0.5f32).ln()), (format!(" {}", self.split.2), (0.49f32).ln()), ("Hello".into(), -9.0)]);
        }
        let g = *self.gold.get(user).unwrap_or(&'A');
        Ok(vec![(g.to_string(), (0.97f32).ln()), ("X".into(), -6.0), (format!("{})", if g == 'A' { 'B' } else { 'A' }), -4.0)])
    }
}

#[derive(Deserialize)]
struct Case {
    state: String,
    gold: Option<String>,
}

fn rows(jsonl: &str) -> Vec<Example> {
    jsonl
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let c: Case = serde_json::from_str(l).unwrap();
            c.gold.map(|label| Example { text: c.state, label })
        })
        .collect()
}

fn train_rows() -> Vec<Example> {
    let mut r = rows(include_str!("fixtures/th_frontdesk_cases.jsonl"));
    let corpus = reflex_study::corpus::load_dir(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/corpora/th-frontdesk"))).unwrap();
    for d in corpus {
        for doc in d.docs {
            r.push(Example { text: doc, label: d.name.clone() });
        }
    }
    r
}

/// The holdout's off-task rows (gold = null): never verified rows, never training rows.
fn off_task() -> Vec<String> {
    include_str!("fixtures/th_frontdesk_holdout.jsonl")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Case>(l).unwrap())
        .filter(|c| c.gold.is_none())
        .map(|c| c.state)
        .collect()
}

fn letters_for(ladder: &Ladder, verified: &[Example], split: &str) -> FakeLetters {
    let gold = verified
        .iter()
        .map(|e| (e.text.clone(), (b'A' + ladder.labels.binary_search(&e.label).unwrap() as u8) as char))
        .collect();
    FakeLetters { gold, split: (split.into(), 'B', 'C'), calls: AtomicUsize::new(0), completes: AtomicUsize::new(0) }
}

fn ladder() -> Ladder {
    Ladder::fit("frontdesk", &train_rows(), BTreeMap::new(), &LadderConfig::default(), Some(&FakeEmbed), Some("fake-letters")).unwrap()
}

#[test]
fn qhat_is_the_ceil_rank_score_and_refuses_when_rows_are_too_few() {
    let s: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
    assert_eq!(conformal_qhat(&s, 0.1), Some(0.9)); // ⌈10·0.9⌉ = 9th smallest
    assert_eq!(conformal_qhat(&s, 0.2), Some(0.8)); // ⌈10·0.8⌉ = 8th
    assert_eq!(conformal_qhat(&s, 0.05), None); // ⌈10·0.95⌉ = 10 > 9 rows
    // the set holds every label with 1 − p ≤ q̂, most probable first
    assert_eq!(conformal_set(&[0.2, 0.7, 0.1], 0.8), vec![1, 0]);
    assert_eq!(conformal_set(&[0.2, 0.7, 0.1], 0.5), vec![1]);
    assert!(conformal_set(&[0.2, 0.7, 0.1], 0.2).is_empty());
}

#[test]
fn sets_cover_the_true_label_at_the_promised_rate_on_exchangeable_data() {
    // A noisy 4-class classifier; calibrate on 500 rows, check coverage on 5000 fresh rows.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut unif = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut draw = |n: usize| -> Vec<(Vec<f32>, usize)> {
        (0..n)
            .map(|_| {
                let y = (unif() * 4.0) as usize % 4;
                let logits: Vec<f64> = (0..4).map(|c| if c == y { 1.2 } else { 0.0 } + 1.5 * (unif() - 0.5) * 2.0).collect();
                let z: f64 = logits.iter().map(|l| l.exp()).sum();
                (logits.iter().map(|l| (l.exp() / z) as f32).collect(), y)
            })
            .collect()
    };
    let cal = draw(500);
    let test = draw(5000);
    for alpha in [0.05, 0.1, 0.2] {
        let q = conformal_qhat(&cal.iter().map(|(p, y)| 1.0 - p[*y] as f64).collect::<Vec<_>>(), alpha).unwrap();
        let covered = test.iter().filter(|(p, y)| conformal_set(p, q).contains(y)).count() as f64 / test.len() as f64;
        assert!(covered >= 1.0 - alpha - 0.03, "α {alpha}: coverage {covered}");
        assert!(covered <= 1.0 - alpha + 0.05, "α {alpha}: coverage {covered} — sets far too wide");
    }
}

#[test]
fn letter_tokens_map_to_labels_and_strangers_are_ignored() {
    let l = ladder();
    let p = l.llm_distribution(&[("B".into(), (0.6f32).ln()), (" a".into(), (0.3f32).ln()), ("C)".into(), (0.1f32).ln()), ("Billing".into(), -0.1)]).unwrap();
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(p[1] > p[0] && p[0] > p[2], "{p:?}");
    assert!(l.llm_distribution(&[("Hello".into(), -0.1), ("ZZ".into(), -1.0)]).is_err());
    assert!(l.llm_letter_prompt().contains("A) appointment") && l.llm_letter_prompt().contains("C) records"));
}

#[test]
fn calibration_refuses_the_training_rows() {
    let mut l = ladder();
    let train = train_rows();
    let chat = letters_for(&l, &train, "");
    let e = l.calibrate_conformal(&train, 0.1, None, "answer", Some(&FakeEmbed), Some(&chat)).unwrap_err();
    assert!(e.contains("training rows"), "{e}");
}

#[test]
fn the_escalator_answers_singletons_and_sends_an_unsure_llm_set_for_review() {
    let verified = rows(include_str!("fixtures/th_frontdesk_holdout.jsonl"));
    let mut l = ladder();
    let chat = letters_for(&l, &verified, "");
    l.calibrate_conformal(&verified, 0.1, None, "review", Some(&FakeEmbed), Some(&chat)).unwrap();
    let cf = l.conformal.clone().unwrap();
    assert_eq!(cf.n, verified.len());
    assert!(cf.llm_qhat.is_some() && cf.lexical_qhat.is_some());
    assert!((cf.shares.values().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(chat.calls.load(Ordering::SeqCst), verified.len(), "one llm call per verified row");

    // A verified-like row answers at some rung, never as review.
    let a = l.decide(&verified[0].text, Some(&FakeEmbed), Some(&chat));
    assert!(a.label.is_some() && a.review.is_none(), "{a:?}");

    // An off-task row both gates pass up reaches the llm rung; when the llm splits B/C the
    // answer is review with exactly that set.
    let reaches_llm = off_task()
        .into_iter()
        .find(|t| l.decide(t, Some(&FakeEmbed), Some(&chat)).steps.iter().any(|s| s.rung == "llm"))
        .expect("some off-task row climbs to the llm rung");
    let chat = letters_for(&l, &verified, &reaches_llm);
    let a = l.decide(&reaches_llm, Some(&FakeEmbed), Some(&chat));
    assert_eq!(a.rung, Some("review"));
    assert_eq!(a.label, None);
    assert_eq!(a.review.as_deref(), Some(&["billing".to_string(), "records".to_string()][..]));
    assert_eq!(chat.completes.load(Ordering::SeqCst), 0, "the escalator never asks for generated text");

    // The same split under when_unsure = answer takes the top label instead.
    let mut practice = ladder();
    practice.calibrate_conformal(&verified, 0.1, None, "answer", Some(&FakeEmbed), Some(&chat)).unwrap();
    let a = practice.decide(&reaches_llm, Some(&FakeEmbed), Some(&chat));
    assert_eq!((a.rung, a.label.as_deref()), (Some("llm"), Some("billing")));
    assert!(a.review.is_none());
    // the calibrated ladder round-trips with its cutoffs
    let dir = std::env::temp_dir().join(format!("escalator-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("l.json");
    practice.save(&path).unwrap();
    let back = Ladder::load(&path).unwrap();
    assert_eq!(back.digest(), practice.digest());
    assert_eq!(back.conformal.unwrap().llm_qhat, practice.conformal.as_ref().unwrap().llm_qhat);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_ladder_without_the_escalator_serializes_and_decides_as_before() {
    let l = Ladder::fit("frontdesk", &train_rows(), BTreeMap::new(), &LadderConfig::default(), None, None).unwrap();
    let json = serde_json::to_string(&l).unwrap();
    assert!(!json.contains("conformal"), "v1 files must not grow the field");
    let a = l.decide("ขอใบเสร็จค่ารักษาใหม่ ฉบับเดิมหาย", None, None);
    assert!(a.review.is_none());
    assert!(!serde_json::to_string(&a).unwrap().contains("review"));
}

#[test]
fn the_rate_guard_flags_a_drifting_share_and_stays_quiet_inside_the_band() {
    let verified = rows(include_str!("fixtures/th_frontdesk_holdout.jsonl"));
    let mut l = ladder();
    let chat = letters_for(&l, &verified, "");
    l.calibrate_conformal(&verified, 0.1, None, "answer", Some(&FakeEmbed), Some(&chat)).unwrap();
    let cf = l.conformal.as_ref().unwrap();
    let top = cf.shares.iter().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0.clone();
    // Inside the band: replay the calibration mix exactly → no warning.
    let mut g = RateGuard::new(cf, 100);
    let mut mix: Vec<String> = Vec::new();
    for (k, v) in &cf.shares {
        mix.extend(std::iter::repeat_n(k.clone(), (v * 100.0).round() as usize));
    }
    mix.resize(100, top.clone());
    let warned: Vec<String> = mix.iter().filter_map(|o| g.observe(o)).collect();
    assert!(warned.is_empty(), "{warned:?}");
    // Every decision suddenly reaching review → flagged.
    let mut g = RateGuard::new(cf, 100);
    let w = (0..100).filter_map(|_| g.observe("review")).last().expect("drift is flagged");
    assert!(w.contains("review"), "{w}");
}

#[test]
fn the_llm_rung_can_take_its_own_alpha_and_bad_alphas_are_refused() {
    let verified = rows(include_str!("fixtures/th_frontdesk_holdout.jsonl"));
    let mut l = ladder();
    let chat = letters_for(&l, &verified, "");
    assert!(l.calibrate_conformal(&verified, 0.1, Some(1.5), "answer", Some(&FakeEmbed), Some(&chat)).is_err());
    assert!(l.calibrate_conformal(&verified, 0.0, None, "answer", Some(&FakeEmbed), Some(&chat)).is_err());
    l.calibrate_conformal(&verified, 0.05, Some(0.3), "review", Some(&FakeEmbed), Some(&chat)).unwrap();
    let cf = l.conformal.as_ref().unwrap();
    assert_eq!((cf.alpha, cf.alpha_llm), (0.05, Some(0.3)));
    // the same rows at one α: the llm cutoff can only be as loose or looser at the bigger α
    let mut same = ladder();
    same.calibrate_conformal(&verified, 0.05, None, "review", Some(&FakeEmbed), Some(&chat)).unwrap();
    let (strict, loose) = (same.conformal.as_ref().unwrap().llm_qhat, cf.llm_qhat);
    assert!(loose.unwrap_or(1.0) <= strict.unwrap_or(1.0), "{loose:?} vs {strict:?}");
    assert_eq!(same.conformal.as_ref().unwrap().lexical_qhat, cf.lexical_qhat, "the lower rungs keep α = 0.05");
}

#[test]
fn a_guide_reaches_both_llm_prompts_and_resets_calibration() {
    let verified = rows(include_str!("fixtures/th_frontdesk_holdout.jsonl"));
    let mut l = ladder();
    assert!(!serde_json::to_string(&l).unwrap().contains("\"guide\""), "no guide = no field");
    let chat = letters_for(&l, &verified, "");
    l.calibrate_conformal(&verified, 0.1, None, "answer", Some(&FakeEmbed), Some(&chat)).unwrap();
    let before = l.digest().to_string();
    l.set_guide(Some("Rule B1: a bare announcement is not a warning.".into()));
    assert!(l.llm_letter_prompt().contains("Rule B1") && l.llm_prompt().contains("Rule B1"));
    assert!(l.conformal.is_none(), "the llm cutoff was fitted under the old prompt");
    assert_ne!(l.digest(), before, "the guide is part of the ladder's identity");
    l.set_guide(Some("   ".into()));
    assert!(l.guide.is_none());
}

