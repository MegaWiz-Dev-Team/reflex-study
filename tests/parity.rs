//! Parity with the official release binary (fixture recorded by scripts/record_parity.py).
//! With `Tokenizer::Ascii` and upstream's defaults, every recorded answer must reproduce:
//! same outcome, same routing reason, probabilities and confidence within 1e-6.

use reflex_study::corpus::load_dir;
use reflex_study::engine::{Engine, EngineConfig, demo_corpus};
use reflex_study::wire::{DecisionRequest, DecisionResponse};
use serde::Deserialize;

#[derive(Deserialize)]
struct Row {
    engine: String,
    request: DecisionRequest,
    response: DecisionResponse,
}

const TOL: f32 = 1e-6;

#[test]
fn reproduces_the_release_binary() {
    let text = include_str!("fixtures/parity.jsonl");
    let mut lines = text.lines();
    let header = lines.next().expect("header line");
    let rows: Vec<Row> = lines.map(|l| serde_json::from_str(l).expect("fixture row")).collect();
    assert!(rows.len() >= 100, "fixture too small: {}", rows.len());

    let mut demo = Engine::build(demo_corpus(), EngineConfig::default()).unwrap();
    let mut corpus = Engine::build(
        load_dir(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/corpora/first-corpus"))).unwrap(),
        EngineConfig::corpus_boot(),
    )
    .unwrap();

    let (mut answers, mut exact, mut worst) = (0usize, 0usize, 0.0f32);
    let mut failures = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let eng = if row.engine == "demo" { &mut demo } else { &mut corpus };
        let got = eng.decide(&row.request).expect("valid request");
        let want = &row.response;
        if got.routing != want.routing || got.calibration != want.calibration {
            failures.push(format!("row {i}: routing/calibration {:?} vs {:?}", got.routing, want.routing));
        }
        for (g, w) in got.answers.iter().zip(&want.answers) {
            answers += 1;
            let mut diffs: Vec<f32> = g.probabilities.iter().zip(&w.probabilities).map(|(a, b)| (a - b).abs()).collect();
            diffs.push((g.confidence - w.confidence).abs());
            let d = diffs.iter().copied().fold(0.0, f32::max);
            worst = worst.max(d);
            exact += (g == w) as usize;
            if g.outcome != w.outcome || g.probabilities.len() != w.probabilities.len() || d > TOL {
                failures.push(format!("row {i} [{}] q={}: got {:?} want {:?}", row.engine, g.question_id, g, w));
            }
        }
    }
    eprintln!("{header}\n{answers} answers · {exact} bit-identical · worst abs diff {worst:e}");
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}
