//! The ladder's contracts, offline: fakes stand in for Heimdall (no test reaches a gateway).

use reflex_study::embed::Embedder;
use reflex_study::heimdall::{Chat, Embed};
use reflex_study::ladder::{Example, Ladder, LadderConfig};
use reflex_study::tokenize::Tokenizer;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A deterministic stand-in encoder: the hashed Thai n-gram embedding itself.
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

/// Answers a fixed reply and counts how often it was asked.
struct FakeChat {
    reply: &'static str,
    calls: AtomicUsize,
}
impl Chat for FakeChat {
    fn model(&self) -> &str {
        "fake-chat"
    }
    fn complete(&self, _system: &str, _user: &str) -> Result<String, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.reply.to_string())
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

fn fit(llm: Option<&str>) -> Ladder {
    Ladder::fit("frontdesk", &train_rows(), BTreeMap::new(), &LadderConfig { target_accuracy: 0.8, ..LadderConfig::default() }, Some(&FakeEmbed), llm).unwrap()
}

#[test]
fn fitting_is_deterministic_and_survives_a_round_trip() {
    let a = fit(None);
    assert_eq!(a.digest(), fit(None).digest());
    let path = std::env::temp_dir().join(format!("ladder_roundtrip_{}.json", std::process::id()));
    a.save(&path).unwrap();
    let b = Ladder::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(a.digest(), b.digest());
    for c in rows(include_str!("fixtures/th_frontdesk_holdout.jsonl")) {
        let (x, y) = (a.decide(&c.text, Some(&FakeEmbed), None), b.decide(&c.text, Some(&FakeEmbed), None));
        assert_eq!((x.label, x.rung, x.receipt.decision), (y.label, y.rung, y.receipt.decision));
    }
}

#[test]
fn the_lexical_rung_routes_thai_on_its_own() {
    let l = fit(None);
    let hold = rows(include_str!("fixtures/th_frontdesk_holdout.jsonl"));
    let right = hold.iter().filter(|c| l.lexical.labels[l.lexical.top(&reflex_study::features::lexical(Tokenizer::Unicode, &c.text)).0] == c.label).count();
    // Measured 2026-10-04 (`ladder eval`, lexical-only ladder): 15/15 — the modelless engine got 13/15.
    assert_eq!(right, 15, "forced lexical accuracy");
}

#[test]
fn an_upper_rung_only_sees_what_the_lower_rungs_passed_up() {
    let l = fit(Some("fake-chat"));
    let chat = FakeChat { reply: "records", calls: AtomicUsize::new(0) };
    let mut passed_up = 0;
    for c in rows(include_str!("fixtures/th_frontdesk_holdout.jsonl")) {
        let a = l.decide(&c.text, Some(&FakeEmbed), Some(&chat));
        let lower_passed = a.steps.iter().take_while(|s| s.rung != "llm").any(|s| s.passed);
        passed_up += !lower_passed as usize;
        assert_eq!(a.steps.iter().filter(|s| s.passed).count(), a.label.is_some() as usize, "at most one rung answers");
        assert_eq!(a.rung, a.steps.iter().find(|s| s.passed).map(|s| s.rung));
    }
    assert_eq!(chat.calls.load(Ordering::SeqCst), passed_up);
}

#[test]
fn a_missing_or_different_encoder_is_skipped_with_a_note_never_answered() {
    let l = fit(None);
    struct Other;
    impl Embed for Other {
        fn model(&self) -> &str {
            "some-other-encoder"
        }
        fn embed(&self, _: &[String]) -> Result<Vec<Vec<f32>>, String> {
            panic!("a mismatched encoder must not be called")
        }
    }
    for text in ["สูตรต้มยำกุ้งน้ำข้น", "พรุ่งนี้ฝนจะตกไหม"] {
        for embed in [None, Some(&Other as &dyn Embed)] {
            let a = l.decide(text, embed, None);
            if let Some(enc) = a.steps.iter().find(|s| s.rung == "encoder") {
                assert!(!enc.passed && enc.note.is_some(), "{enc:?}");
            }
        }
    }
}

#[test]
fn the_llm_reply_must_name_exactly_one_label() {
    let l = fit(Some("fake-chat"));
    assert_eq!(l.parse_llm("billing").as_deref(), Some("billing"));
    assert_eq!(l.parse_llm("  Billing.\n").as_deref(), Some("billing"));
    assert_eq!(l.parse_llm("`records`").as_deref(), Some("records"));
    assert_eq!(l.parse_llm("UNSURE"), None);
    assert_eq!(l.parse_llm("billing or records"), None);
    assert_eq!(l.parse_llm(""), None);
}

#[test]
fn off_task_text_is_passed_up_not_filed_under_a_label() {
    let l = fit(None);
    // Every off-task holdout row is unlike the training rows; with no llm rung, all must abstain.
    let off: Vec<String> = include_str!("fixtures/th_frontdesk_holdout.jsonl")
        .lines()
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| {
            let c: Case = serde_json::from_str(x).unwrap();
            c.gold.is_none().then_some(c.state)
        })
        .collect();
    assert_eq!(off.len(), 6);
    let answered: Vec<_> = off.iter().filter_map(|t| l.decide(t, None, None).label.map(|lab| (t, lab))).collect();
    assert!(answered.is_empty(), "off-task rows answered: {answered:?}");
}

#[test]
fn the_promotion_gate_refuses_a_ladder_trained_on_poisoned_teacher_labels() {
    use reflex_study::evolve;
    let good = fit(None);
    // Every "billing" row relabelled "records": what an upper rung that is reliably wrong
    // about one class would feed back.
    let poisoned: Vec<Example> = train_rows()
        .into_iter()
        .map(|e| Example { label: if e.label == "billing" { "records".into() } else { e.label }, ..e })
        .collect();
    let bad = Ladder::fit("frontdesk", &poisoned, BTreeMap::new(), &LadderConfig { target_accuracy: 0.8, ..LadderConfig::default() }, Some(&FakeEmbed), None).unwrap();
    let verified = rows(include_str!("fixtures/th_frontdesk_holdout.jsonl"));
    let p = evolve::compare(&good, &bad, &verified, Some(&FakeEmbed), None);
    assert!(!p.promoted, "{}", p.reason);
    assert!(evolve::compare(&good, &good, &verified, Some(&FakeEmbed), None).promoted, "an identical candidate is not worse");
}

#[test]
fn majority_needs_two_matching_voters_else_the_llm_decides() {
    use reflex_study::evolve::{Votes, majority};
    let v = |lex: &str, enc: Option<&str>, llm: Option<&str>| Votes {
        answer: Some(lex.into()), rung: Some("lexical"), confidence: Some(0.5),
        lexical: lex.into(), lexical_confidence: 0.5, encoder: enc.map(Into::into), llm: llm.map(Into::into),
    };
    assert_eq!(majority(&v("a", Some("a"), Some("b"))).as_deref(), Some("a"));
    assert_eq!(majority(&v("a", Some("b"), Some("b"))).as_deref(), Some("b"));
    assert_eq!(majority(&v("a", Some("b"), Some("c"))).as_deref(), Some("c"));
    assert_eq!(majority(&v("a", None, None)).as_deref(), Some("a"));
}

#[test]
fn a_remembered_example_answers_from_memory_at_once_and_stops_the_climb() {
    let mut l = fit(Some("fake-chat"));
    let before = l.digest().to_string();
    // Off-task texts the lexical rung passes up, taught to memory as one label: they must be
    // answered from memory immediately — no retraining, no llm call.
    let off_all: Vec<Example> = [include_str!("fixtures/th_frontdesk_cases.jsonl"), include_str!("fixtures/th_frontdesk_holdout.jsonl")]
        .iter()
        .flat_map(|f| f.lines())
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| {
            let c: Case = serde_json::from_str(x).unwrap();
            c.gold.is_none().then_some(Example { text: c.state, label: "records".into() })
        })
        .collect();
    let off: Vec<Example> = off_all.iter().filter(|e| !l.decide(&e.text, None, None).steps[0].passed).cloned().collect();
    // Measured 2026-10-05: the lexical gate lets 1 of the 16 off-task rows through
    // ("พรุ่งนี้ฝนจะตกที่เชียงใหม่ไหม" → appointment — "พรุ่งนี้" is in the appointment rows).
    assert_eq!((off.len(), off_all.len()), (15, 16));
    let mut taught = train_rows();
    taught.extend(off.iter().cloned());
    assert_eq!(l.remember(&taught, &FakeEmbed).unwrap(), taught.len());
    assert_ne!(l.digest(), before, "remembering changes the ladder's identity");
    let chat = FakeChat { reply: "billing", calls: AtomicUsize::new(0) };
    for e in &off {
        let a = l.decide(&e.text, Some(&FakeEmbed), Some(&chat));
        assert_eq!((a.rung, a.label.as_deref()), (Some("memory"), Some("records")), "{}", e.text);
        assert_eq!(a.steps.iter().filter(|s| s.passed).count(), 1);
    }
    assert_eq!(chat.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn split_training_sets_fit_each_rung_on_its_own_rows() {
    let rows = train_rows();
    let half = rows.len() / 2;
    let (a, b) = (rows[..].to_vec(), rows[half..].iter().chain(rows[..half].iter()).cloned().collect::<Vec<_>>());
    let l = Ladder::fit_split("frontdesk", &a, &b, BTreeMap::new(), &LadderConfig { target_accuracy: 0.8, ..LadderConfig::default() }, Some(&FakeEmbed), None).unwrap();
    assert!(l.encoder.as_ref().unwrap().fit.basis.contains("separate training set"));
    assert_eq!(l.trained_on, a.len() + b.len());
}

#[test]
fn nbsvm_and_auto_train_deterministically_and_record_the_choice() {
    use reflex_study::ladder::LexicalKind;
    let cfg = |k| LadderConfig { target_accuracy: 0.8, lexical: k, ..LadderConfig::default() };
    let rows = train_rows();
    let n1 = Ladder::fit("frontdesk", &rows, BTreeMap::new(), &cfg(LexicalKind::Nbsvm), Some(&FakeEmbed), None).unwrap();
    let n2 = Ladder::fit("frontdesk", &rows, BTreeMap::new(), &cfg(LexicalKind::Nbsvm), Some(&FakeEmbed), None).unwrap();
    assert_eq!(n1.digest(), n2.digest());
    assert_eq!(n1.lexical_features, "presence");
    let auto = Ladder::fit("frontdesk", &rows, BTreeMap::new(), &cfg(LexicalKind::Auto), Some(&FakeEmbed), None).unwrap();
    let c = auto.lexical_choice.as_ref().expect("auto records its choice");
    assert_eq!(c.candidates.len(), 5, "bag, nbsvm, and three distillation mixes");
    assert_eq!(c.chosen == "bag", c.lb95_vs_bag <= 0.0, "a challenger is kept only with LB95 > 0");
    // Whatever was chosen, the ladder still decides through it end to end.
    let a = auto.decide("ขอใบเสร็จค่ารักษาใหม่ ฉบับเดิมหาย", Some(&FakeEmbed), None);
    assert!(a.steps.first().is_some_and(|s| s.rung == "lexical"));
}

#[test]
fn the_default_lexical_rung_serializes_as_before() {
    let l = fit(None);
    let json = String::from_utf8(std::fs::read({
        let p = std::env::temp_dir().join(format!("ladder_default_{}.json", std::process::id()));
        l.save(&p).unwrap();
        p
    }).unwrap()).unwrap();
    assert!(!json.contains("lexical_features") && !json.contains("lexical_choice"), "Bag ladders keep the pre-Instinct file layout");
}
