//! `reflex-study serve` — the `/decide`, `/feedback`, `/healthz` wire on loopback.
//! `reflex-study eval`  — run a labeled case file, report accuracy/abstain, fit thresholds.

use reflex_study::corpus::{self, DomainDocs};
use reflex_study::engine::{AbstainCause, Engine, EngineConfig, demo_corpus};
use reflex_study::http;
use reflex_study::threshold;
use reflex_study::tokenize::Tokenizer;
use reflex_study::wire::{DecisionRequest, Kind, Question};
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const USAGE: &str = "usage:
  reflex-study serve [--corpus DIR] [--bind 127.0.0.1:7341] [ENGINE FLAGS]
  reflex-study eval  --corpus DIR --cases FILE.jsonl [--rho 0.3] [--target 0.9] [ENGINE FLAGS]

ENGINE FLAGS
  --tokenizer ascii|unicode   ascii = upstream byte-for-byte (default); unicode = + Thai n-grams
  --dim N                     embedding buckets (default 256)
  --gate-mid X                distance-gate midpoint on max cosine (default 0.35)
  --idf                       weight features by IDF fitted from the corpus (not upstream)
  --gate-on-state             distance gate reads the state alone, not state + prompt (not upstream)
  --score-threshold X         default 0.35 for the demo engine, 0.0 with --corpus (upstream)

CASE FILE (one JSON object per line)
  {\"state\": \"...\", \"prompt\": \"...\", \"options\": [\"a\",\"b\"], \"gold\": \"a\"}
  gold = null marks an off-corpus case: the right answer is to abstain.";

struct Args {
    cmd: String,
    corpus: Option<String>,
    bind: String,
    cases: Option<String>,
    rho: f64,
    target: f64,
    cfg: EngineConfig,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or(USAGE)?;
    let mut a = Args {
        cmd,
        corpus: None,
        bind: "127.0.0.1:7341".into(),
        cases: None,
        rho: 0.30,
        target: 0.90,
        cfg: EngineConfig::default(),
    };
    let mut score_threshold = None;
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        let num = |s: String| s.parse::<f64>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--corpus" => a.corpus = Some(val()?),
            "--bind" => a.bind = val()?,
            "--cases" => a.cases = Some(val()?),
            "--rho" => a.rho = num(val()?)?,
            "--target" => a.target = num(val()?)?,
            "--tokenizer" => {
                a.cfg.tokenizer = Tokenizer::parse(&val()?).ok_or("--tokenizer is ascii or unicode")?
            }
            "--dim" => a.cfg.dim = val()?.parse().map_err(|e| format!("--dim: {e}"))?,
            "--gate-mid" => a.cfg.gate_mid = num(val()?)? as f32,
            "--idf" => a.cfg.idf = true,
            "--gate-on-state" => a.cfg.gate_on_state = true,
            "--score-threshold" => score_threshold = Some(num(val()?)? as f32),
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }
    if a.corpus.is_some() {
        a.cfg.score_threshold = EngineConfig::corpus_boot().score_threshold;
    }
    if let Some(t) = score_threshold {
        a.cfg.score_threshold = t;
    }
    Ok(a)
}

fn load(a: &Args) -> Result<(Vec<DomainDocs>, String), String> {
    match &a.corpus {
        Some(dir) => {
            let c = corpus::load_dir(std::path::Path::new(dir))?;
            let names = serde_json::to_string(&c.iter().map(|d| &d.name).collect::<Vec<_>>())
                .expect("names serialize");
            Ok((c, format!("{{\"domains\":{names}}}")))
        }
        None => Ok((demo_corpus(), "\"demo\"".into())),
    }
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let result = match a.cmd.as_str() {
        "serve" => serve(&a),
        "eval" => eval(&a),
        _ => Err(USAGE.into()),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

// ───────────────────────────── serve ─────────────────────────────

fn serve(a: &Args) -> Result<(), String> {
    let (docs, corpus_json) = load(a)?;
    let engine = Engine::build(docs, a.cfg.clone())?;
    eprintln!(
        "[reflex-study] domains {:?} · tokenizer {} · dim {} · score_threshold {} · distance_threshold {}",
        engine.domain_names(),
        a.cfg.tokenizer.as_str(),
        a.cfg.dim,
        a.cfg.score_threshold,
        a.cfg.distance_threshold
    );
    eprintln!("[reflex-study] listening on http://{}", a.bind);
    let engine = Mutex::new(engine);
    let handler = move |method: &str, path: &str, body: &[u8]| -> (u16, String) {
        match (method, path) {
            ("GET", "/healthz") => (
                200,
                format!(r#"{{"status":"ok","lanes":{{"modelless":"ready"}},"corpus":{corpus_json}}}"#),
            ),
            ("POST", "/decide") => match serde_json::from_slice::<DecisionRequest>(body) {
                Err(e) => (400, http::error_json(&e.to_string())),
                Ok(req) => match engine.lock().expect("engine lock").decide(&req) {
                    Ok(resp) => (200, serde_json::to_string(&resp).expect("response serializes")),
                    Err(e) => (422, http::error_json(&format!("request invalid: decision_wire: {e}"))),
                },
            },
            ("POST", "/feedback") => {
                #[derive(Deserialize)]
                struct Fb {
                    p: f32,
                    outcome: bool,
                }
                match serde_json::from_slice::<Fb>(body) {
                    Err(e) => (400, http::error_json(&e.to_string())),
                    Ok(f) => {
                        let refit = engine.lock().expect("engine lock").observe(f.p, f.outcome);
                        (200, format!(r#"{{"refit":{refit}}}"#))
                    }
                }
            }
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    };
    http::serve(&a.bind, http::allowed_origins("REFLEX_ALLOWED_ORIGIN"), Arc::new(handler))
}

// ───────────────────────────── eval ─────────────────────────────

#[derive(Deserialize)]
struct Case {
    state: String,
    #[serde(default = "default_prompt")]
    prompt: String,
    options: Vec<String>,
    /// The right option, or `null` when the right answer is to abstain (off-corpus).
    gold: Option<String>,
}

fn default_prompt() -> String {
    "Which one applies?".into()
}

/// Mann–Whitney AUC of `pos` over `neg` (ties count ½): how well one number separates the two.
fn auc(pos: &[f32], neg: &[f32]) -> f64 {
    let mut wins = 0.0;
    for p in pos {
        for n in neg {
            wins += if p > n { 1.0 } else if p == n { 0.5 } else { 0.0 };
        }
    }
    wins / (pos.len() * neg.len()) as f64
}

/// The gate midpoint that best splits in-corpus (pass) from off-corpus (abstain) max-cosines:
/// maximize the balanced rate, cut halfway between neighbours, widest gap on ties.
fn fit_gate_mid(in_sims: &[f32], off_sims: &[f32]) -> (f32, f64, f64) {
    let mut all: Vec<f32> = in_sims.iter().chain(off_sims).copied().collect();
    all.sort_by(f32::total_cmp);
    all.dedup();
    let mut best = (0.0f32, -1.0f64, -1.0f32, 0.0, 0.0);
    for w in all.windows(2) {
        let cut = (w[0] + w[1]) / 2.0;
        let pass = in_sims.iter().filter(|s| **s >= cut).count() as f64 / in_sims.len() as f64;
        let stop = off_sims.iter().filter(|s| **s < cut).count() as f64 / off_sims.len() as f64;
        let score = (pass + stop) / 2.0;
        let gap = w[1] - w[0];
        if score > best.1 || (score == best.1 && gap > best.2) {
            best = (cut, score, gap, pass, stop);
        }
    }
    (best.0, best.3, best.4)
}

fn eval(a: &Args) -> Result<(), String> {
    let path = a.cases.as_ref().ok_or("eval needs --cases FILE.jsonl")?;
    let (docs, _) = load(a)?;
    let mut engine = Engine::build(docs, a.cfg.clone())?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let cases: Vec<Case> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("{path}:{}: {e}", i + 1)))
        .collect::<Result<_, _>>()?;

    let mut in_obs: Vec<(f32, bool)> = Vec::new();
    let (mut in_answered, mut in_right_answered, mut in_right_forced) = (0, 0, 0);
    let (mut off_n, mut off_abstained) = (0, 0);
    let (mut in_sims, mut off_sims) = (Vec::new(), Vec::new());
    let mut lat = Vec::new();
    println!("{:<4} {:<8} {:<12} {:<12} {:>7} {:>6}  state", "#", "result", "gold", "forced pick", "conf", "maxcos");
    for (i, c) in cases.iter().enumerate() {
        let req = DecisionRequest {
            state: c.state.clone(),
            questions: vec![Question {
                id: "q".into(),
                kind: Kind::Choice,
                prompt: c.prompt.clone(),
                options: c.options.clone(),
                criteria: None,
            }],
        };
        let t = Instant::now();
        let v = engine.solve(&req).map_err(|e| format!("case {}: {e}", i + 1))?.remove(0);
        lat.push(t.elapsed());
        let pick = &c.options[v.pick];
        let result = match &c.gold {
            Some(g) => {
                let right = pick == g;
                in_sims.push(v.max_similarity);
                in_obs.push((v.confidence, right));
                in_right_forced += right as usize;
                if !v.abstained() {
                    in_answered += 1;
                    in_right_answered += right as usize;
                }
                match (v.abstained(), right) {
                    (false, true) => "right",
                    (false, false) => "WRONG",
                    (true, _) => "abstain",
                }
            }
            None => {
                off_n += 1;
                off_sims.push(v.max_similarity);
                if v.abstained() {
                    off_abstained += 1;
                    "abstain"
                } else {
                    "ANSWERED"
                }
            }
        };
        let short: String = c.state.chars().take(44).collect();
        let mark = match (&c.gold, v.cause) {
            (Some(g), _) if g != pick => " ✗",
            (_, AbstainCause::ScoreGate) => " [score gate]",
            _ => "",
        };
        println!(
            "{:<4} {:<8} {:<12} {:<12} {:>7.4} {:>6.3}  {short}{mark}",
            i + 1,
            result,
            c.gold.as_deref().unwrap_or("(off)"),
            pick,
            v.confidence,
            v.max_similarity,
        );
    }
    let n_in = in_obs.len();
    println!();
    println!(
        "tokenizer {} · dim {} · idf {} · gate on {} · gate_mid {} · score_threshold {}",
        a.cfg.tokenizer.as_str(),
        a.cfg.dim,
        a.cfg.idf,
        if a.cfg.gate_on_state { "state" } else { "state+prompt" },
        a.cfg.gate_mid,
        a.cfg.score_threshold
    );
    if n_in > 0 {
        println!(
            "in-corpus  n={n_in}: forced-pick accuracy {:.3} · answered {in_answered}/{n_in} · accuracy on answered {}",
            in_right_forced as f64 / n_in as f64,
            if in_answered > 0 { format!("{:.3}", in_right_answered as f64 / in_answered as f64) } else { "n/a".into() }
        );
    }
    if off_n > 0 {
        println!("off-corpus n={off_n}: abstained {off_abstained}/{off_n}");
    }
    let stats = |v: &[f32]| {
        let mut v = v.to_vec();
        v.sort_by(f32::total_cmp);
        if v.is_empty() { "n/a".to_string() } else { format!("min {:.3} · median {:.3} · max {:.3}", v[0], v[v.len() / 2], v[v.len() - 1]) }
    };
    println!("max cosine (routed domain) — in-corpus: {} | off-corpus: {}", stats(&in_sims), stats(&off_sims));
    if !in_sims.is_empty() && !off_sims.is_empty() {
        let (mid, pass, stop) = fit_gate_mid(&in_sims, &off_sims);
        println!(
            "gate separation AUC {:.3} · best midpoint on THESE cases {mid:.4} (in-corpus pass {:.2}, off-corpus abstain {:.2}; n_in={n_in}, n_off={off_n})",
            auc(&in_sims, &off_sims),
            pass,
            stop
        );
    }
    lat.sort();
    if !lat.is_empty() {
        println!("latency per decision (in-process) p50 {:?} · max {:?}", lat[lat.len() / 2], lat[lat.len() - 1]);
    }
    match threshold::percentile(&in_obs, a.rho) {
        Some(r) => println!("score threshold, percentile ρ={}: {}", a.rho, serde_json::to_string(&r).expect("serializes")),
        None => println!("score threshold, percentile ρ={}: none — {n_in} labeled in-corpus cases < {}", a.rho, threshold::THIN_SUPPORT_FLOOR),
    }
    match threshold::target_accuracy(&in_obs, a.target) {
        Some(r) => println!("score threshold, target accuracy {}: {}", a.target, serde_json::to_string(&r).expect("serializes")),
        None => println!("score threshold, target accuracy {}: none — {n_in} labeled in-corpus cases < {}", a.target, threshold::THIN_SUPPORT_FLOOR),
    }
    Ok(())
}
