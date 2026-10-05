//! `ladder train` — fit the rungs of one decision task from labeled rows.
//! `ladder eval`  — run a ladder over labeled cases: coverage and accuracy per rung.
//! `ladder serve` — `POST /v1/classify {"task", "text"}` on loopback for one or more ladders.

use reflex_study::heimdall::{self, Chat, Embed, HeimdallChat, HeimdallEmbed};
use reflex_study::http;
use reflex_study::ladder::{Example, Ladder, LadderConfig};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

const USAGE: &str = "usage:
  ladder train --task NAME --train ROWS.jsonl --out LADDER.json
               [--descriptions LABELS.json] [--target 0.9] [--folds 4]
               [--encoder BAAI/bge-m3 | --no-encoder] [--llm gemma-4-26b] [--embed-cache FILE]
               [--lexical bag|nbsvm|nbsvm-distill[:mix]|auto]   (auto = Ultra Instinct: gate-selected)
  ladder eval  --ladder LADDER.json --cases CASES.jsonl [--embed-cache FILE] [--no-llm]
  ladder serve --ladder LADDER.json [--ladder ...] [--bind 127.0.0.1:7342] [--embed-cache FILE]

ROWS / CASES: one JSON object per line, {\"text\": \"...\", \"label\": \"...\"}
  (in CASES, label null = the right answer is to abstain)
LABELS.json: {\"<label>\": \"<one-line meaning>\", ...} — the llm rung's prompt
The encoder and llm rungs call the local Heimdall gateway (HEIMDALL_API_URL, HEIMDALL_API_KEY).
The llm rung only ever calls the gateway's active local model (HEIMDALL_LOCAL_MODEL, default
gemma-4-26b): any other name would make Heimdall hot-swap the backend shared by every client.";

#[derive(Default)]
struct Args {
    cmd: String,
    task: Option<String>,
    train: Option<String>,
    out: Option<String>,
    descriptions: Option<String>,
    ladders: Vec<String>,
    cases: Option<String>,
    target: Option<f64>,
    folds: Option<usize>,
    encoder: Option<String>,
    llm: Option<String>,
    no_llm: bool,
    lexical: Option<String>,
    embed_cache: Option<PathBuf>,
    bind: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        cmd: it.next().ok_or(USAGE)?,
        encoder: Some(heimdall::DEFAULT_EMBED_MODEL.into()),
        ..Args::default()
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--task" => a.task = Some(val()?),
            "--train" => a.train = Some(val()?),
            "--out" => a.out = Some(val()?),
            "--descriptions" => a.descriptions = Some(val()?),
            "--ladder" => a.ladders.push(val()?),
            "--cases" => a.cases = Some(val()?),
            "--target" => a.target = Some(val()?.parse().map_err(|e| format!("--target: {e}"))?),
            "--folds" => a.folds = Some(val()?.parse().map_err(|e| format!("--folds: {e}"))?),
            "--encoder" => a.encoder = Some(val()?),
            "--no-encoder" => a.encoder = None,
            "--llm" => a.llm = Some(val()?),
            "--no-llm" => a.no_llm = true,
            "--lexical" => a.lexical = Some(val()?),
            "--embed-cache" => a.embed_cache = Some(PathBuf::from(val()?)),
            "--bind" => a.bind = Some(val()?),
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }
    Ok(a)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &str) -> Result<Vec<T>, String> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("{path}: {e}"))?
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("{path}:{}: {e}", i + 1)))
        .collect()
}

fn main() {
    let result = parse_args().and_then(|a| match a.cmd.as_str() {
        "train" => train(&a),
        "eval" => eval(&a),
        "serve" => serve(&a),
        _ => Err(USAGE.into()),
    });
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn train(a: &Args) -> Result<(), String> {
    let task = a.task.as_deref().ok_or("train needs --task")?;
    let rows: Vec<Example> = read_jsonl(a.train.as_deref().ok_or("train needs --train")?)?;
    let out = PathBuf::from(a.out.as_deref().ok_or("train needs --out")?);
    let descriptions: BTreeMap<String, String> = match &a.descriptions {
        Some(p) => serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?,
        None => BTreeMap::new(),
    };
    if let Some(m) = &a.llm
        && *m != heimdall::local_chat_model()
    {
        return Err(format!(
            "--llm {m}: the llm rung may only name the gateway's local model {} (another name hot-swaps the shared backend)",
            heimdall::local_chat_model()
        ));
    }
    let mut cfg = LadderConfig::default();
    cfg.target_accuracy = a.target.unwrap_or(cfg.target_accuracy);
    cfg.folds = a.folds.unwrap_or(cfg.folds);
    cfg.lexical = match a.lexical.as_deref() {
        None => reflex_study::ladder::LexicalKind::Bag,
        Some(o) => reflex_study::ladder::LexicalKind::parse(o).map_err(|e| format!("--lexical {e}"))?,
    };
    let embed = a.encoder.as_deref().map(|m| HeimdallEmbed::new(m, a.embed_cache.clone()));
    let t = std::time::Instant::now();
    let ladder = Ladder::fit(task, &rows, descriptions, &cfg, embed.as_ref().map(|e| e as &dyn Embed), a.llm.as_deref())?;
    ladder.save(&out)?;
    println!("task {task} · {} rows · labels {:?} · fit in {:.1?}", rows.len(), ladder.labels, t.elapsed());
    let show = |name: &str, fit: &reflex_study::ladder::RungFit| {
        println!(
            "  {name:8} threshold {:>8} · out-of-fold accuracy {:.3} · fitted on {} rows ({})",
            fit.threshold.map_or("never".into(), |t| format!("{t:.4}")),
            fit.oof_accuracy,
            fit.n,
            fit.basis
        );
        if let Some(r) = &fit.recommendation {
            println!(
                "           at that cutoff: answers {}/{} at accuracy {:.3}{}",
                r.n_pass,
                r.n,
                r.answered_accuracy,
                if r.met == Some(true) { "" } else { " — target NOT met, rung disabled" }
            );
        }
    };
    let gate = |g: &reflex_study::ladder::Gate| {
        println!(
            "           novelty gate: nearest-row cosine ≥ {} (ρ={} of {} training rows, out-of-fold)",
            g.cutoff.map_or("— (too few rows; rung never answers)".into(), |c| format!("{c:.4}")),
            g.rho,
            g.rows.len()
        );
    };
    show("lexical", &ladder.lexical_fit);
    if let Some(c) = &ladder.lexical_choice {
        println!("           Ultra Instinct: candidates {:?} · LB95 vs bag {:+.3} → {}", c.candidates.iter().map(|(k, a)| format!("{k} {a:.3}")).collect::<Vec<_>>(), c.lb95_vs_bag, c.chosen);
    }
    gate(&ladder.lexical_gate);
    if let Some(e) = &ladder.encoder {
        show("encoder", &e.fit);
        gate(&e.gate);
    }
    println!("  llm      {}", ladder.llm.as_deref().unwrap_or("(none)"));
    println!("wrote {} · digest {}", out.display(), ladder.digest());
    Ok(())
}

#[derive(Deserialize)]
struct Case {
    text: String,
    label: Option<String>,
}

struct Clients {
    embed: Option<HeimdallEmbed>,
    chat: Option<HeimdallChat>,
}

impl Clients {
    fn for_ladders(ladders: &[Ladder], cache: Option<PathBuf>, no_llm: bool) -> Self {
        let embed = ladders.iter().find_map(|l| l.encoder.as_ref()).map(|e| HeimdallEmbed::new(&e.model, cache));
        let chat = if no_llm { None } else { ladders.iter().find_map(|l| l.llm.as_deref()).map(HeimdallChat::new) };
        Self { embed, chat }
    }
    fn embed(&self) -> Option<&dyn Embed> {
        self.embed.as_ref().map(|e| e as &dyn Embed)
    }
    fn chat(&self) -> Option<&dyn Chat> {
        self.chat.as_ref().map(|c| c as &dyn Chat)
    }
}

fn eval(a: &Args) -> Result<(), String> {
    let ladder = Ladder::load(&PathBuf::from(a.ladders.first().ok_or("eval needs --ladder")?))?;
    let cases: Vec<Case> = read_jsonl(a.cases.as_deref().ok_or("eval needs --cases")?)?;
    let clients = Clients::for_ladders(std::slice::from_ref(&ladder), a.embed_cache.clone(), a.no_llm);
    // per rung: (answered, right); plus latency samples
    let mut per: BTreeMap<&str, (usize, usize, Vec<u64>)> = BTreeMap::new();
    let (mut n_in, mut answered, mut right, mut off, mut off_abstained) = (0, 0, 0, 0, 0);
    println!("{:<4} {:<8} {:<9} {:<22} {:<22} {:>7}  text", "#", "result", "rung", "gold", "label", "conf");
    for (i, c) in cases.iter().enumerate() {
        let ans = ladder.decide(&c.text, clients.embed(), clients.chat());
        for s in &ans.steps {
            per.entry(s.rung).or_default().2.push(s.micros);
        }
        let result = match (&c.label, &ans.label) {
            (Some(g), Some(l)) => {
                n_in += 1;
                answered += 1;
                let ok = g == l;
                right += ok as usize;
                let e = per.entry(ans.rung.unwrap()).or_default();
                e.0 += 1;
                e.1 += ok as usize;
                if ok { "right" } else { "WRONG" }
            }
            (Some(_), None) => {
                n_in += 1;
                "abstain"
            }
            (None, None) => {
                off += 1;
                off_abstained += 1;
                "abstain"
            }
            (None, Some(_)) => {
                off += 1;
                "ANSWERED"
            }
        };
        let short: String = c.text.chars().take(40).collect();
        println!(
            "{:<4} {:<8} {:<9} {:<22} {:<22} {:>7}  {short}",
            i + 1,
            result,
            ans.rung.unwrap_or("-"),
            c.label.as_deref().unwrap_or("(off)"),
            ans.label.as_deref().unwrap_or("-"),
            ans.confidence.map_or("-".into(), |c| format!("{c:.3}")),
        );
    }
    println!("\nladder {} · task {} · digest {}", a.ladders[0], ladder.task, &ladder.digest()[..16]);
    println!(
        "labeled n={n_in}: answered {answered} ({:.0}%) · right {right} · accuracy on answered {}",
        100.0 * answered as f64 / n_in.max(1) as f64,
        if answered > 0 { format!("{:.3}", right as f64 / answered as f64) } else { "n/a".into() }
    );
    if off > 0 {
        println!("off-task n={off}: abstained {off_abstained}");
    }
    for (rung, (ans, ok, mut lat)) in per {
        lat.sort_unstable();
        println!(
            "  {rung:8} reached {:3} · answered {ans:3} · right {ok:3} · p50 {} µs",
            lat.len(),
            lat.get(lat.len() / 2).copied().unwrap_or(0)
        );
    }
    Ok(())
}

fn serve(a: &Args) -> Result<(), String> {
    if a.ladders.is_empty() {
        return Err("serve needs at least one --ladder".into());
    }
    let ladders: Vec<Ladder> = a.ladders.iter().map(|p| Ladder::load(&PathBuf::from(p))).collect::<Result<_, _>>()?;
    let clients = Clients::for_ladders(&ladders, a.embed_cache.clone(), a.no_llm);
    let bind = a.bind.clone().unwrap_or_else(|| "127.0.0.1:7342".into());
    for l in &ladders {
        eprintln!(
            "[ladder] task {} · labels {} · rungs lexical{}{} · digest {}",
            l.task,
            l.labels.len(),
            if l.encoder.is_some() { "→encoder" } else { "" },
            if l.llm.is_some() { "→llm" } else { "" },
            &l.digest()[..16]
        );
    }
    eprintln!("[ladder] listening on http://{bind}");
    let health = serde_json::json!({
        "status": "ok",
        "tasks": ladders.iter().map(|l| serde_json::json!({"task": l.task, "labels": l.labels, "digest": l.digest()})).collect::<Vec<_>>(),
    })
    .to_string();
    let handler = move |method: &str, path: &str, body: &[u8]| -> (u16, String) {
        #[derive(Deserialize)]
        struct Req {
            task: String,
            text: String,
        }
        match (method, path) {
            ("GET", "/healthz") => (200, health.clone()),
            ("POST", "/v1/classify") => match serde_json::from_slice::<Req>(body) {
                Err(e) => (400, http::error_json(&e.to_string())),
                Ok(r) => match ladders.iter().find(|l| l.task == r.task) {
                    None => (422, http::error_json(&format!("unknown task {:?}", r.task))),
                    Some(l) => (200, serde_json::to_string(&l.decide(&r.text, clients.embed(), clients.chat())).expect("answer serializes")),
                },
            },
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    };
    http::serve(&bind, http::allowed_origins("REFLEX_ALLOWED_ORIGIN"), Arc::new(handler))
}
