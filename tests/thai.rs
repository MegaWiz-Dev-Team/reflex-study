//! Thai posture pins on corpora/th-frontdesk (deterministic engine → exact counts).
//!
//! Upstream's ASCII tokenizer turns Thai into the zero vector: routing is at chance and the
//! distance gate cannot tell in-corpus from off-corpus. The Unicode tokenizer (Thai character
//! n-grams, 1024 buckets) routes and gates. Calibration file = tuning; holdout = read once.
//! A change to any count here is a measured behaviour change: re-pin it deliberately.

use reflex_study::corpus::load_dir;
use reflex_study::engine::{Engine, EngineConfig};
use reflex_study::tokenize::Tokenizer;
use reflex_study::wire::{DecisionRequest, Kind, Question};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    state: String,
    prompt: String,
    options: Vec<String>,
    gold: Option<String>,
}

struct Tally {
    in_right: usize,
    in_n: usize,
    off_abstained: usize,
    off_n: usize,
}

fn run(cfg: EngineConfig, cases: &str) -> Tally {
    let corpus = load_dir(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/corpora/th-frontdesk"))).unwrap();
    let mut eng = Engine::build(corpus, cfg).unwrap();
    let mut t = Tally { in_right: 0, in_n: 0, off_abstained: 0, off_n: 0 };
    for line in cases.lines().filter(|l| !l.trim().is_empty()) {
        let c: Case = serde_json::from_str(line).unwrap();
        let req = DecisionRequest {
            state: c.state,
            questions: vec![Question { id: "q".into(), kind: Kind::Choice, prompt: c.prompt, options: c.options.clone(), criteria: None }],
        };
        let v = eng.solve(&req).unwrap().remove(0);
        match c.gold {
            Some(g) => {
                t.in_n += 1;
                t.in_right += (c.options[v.pick] == g) as usize;
            }
            None => {
                t.off_n += 1;
                t.off_abstained += v.abstained() as usize;
            }
        }
    }
    t
}

const CAL: &str = include_str!("fixtures/th_frontdesk_cases.jsonl");
const HOLDOUT: &str = include_str!("fixtures/th_frontdesk_holdout.jsonl");

fn thai_cfg() -> EngineConfig {
    EngineConfig {
        tokenizer: Tokenizer::Unicode,
        dim: 1024,
        // Fitted on the calibration file (best balanced split of max-cosine), never on holdout.
        gate_mid: 0.1511,
        ..EngineConfig::corpus_boot()
    }
}

#[test]
fn upstream_tokenizer_routes_thai_at_chance() {
    let cal = run(EngineConfig::corpus_boot(), CAL);
    assert_eq!((cal.in_right, cal.in_n), (12, 33)); // chance is 11/33
    let hold = run(EngineConfig::corpus_boot(), HOLDOUT);
    assert_eq!((hold.in_right, hold.in_n), (5, 15)); // chance is 5/15
}

#[test]
fn unicode_tokenizer_routes_and_gates_thai() {
    let cal = run(thai_cfg(), CAL);
    assert_eq!((cal.in_right, cal.in_n), (30, 33));
    assert_eq!((cal.off_abstained, cal.off_n), (9, 10));
    let hold = run(thai_cfg(), HOLDOUT);
    assert_eq!((hold.in_right, hold.in_n), (13, 15));
    assert_eq!((hold.off_abstained, hold.off_n), (4, 6));
}
