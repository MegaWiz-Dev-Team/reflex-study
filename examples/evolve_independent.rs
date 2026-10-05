//! The update loop with independent voters: upper-rung labels are SERVED, never trained on.
//! Only the seed rows and human-audited rows train the ladder, so the lexical and encoder
//! rungs never learn each other's (or the llm's) mistakes, and the 2-of-3 vote keeps its power.
//!
//! usage: evolve_independent RUN_DIR POOLS.jsonl [ROUNDS=3] [AUDIT_SHARE=0.1]
//! (same RUN_DIR layout as `evolve_rounds`)
//!
//! - verified-random: audit every 10th row the upper rungs answered (the earlier gated policy's audit);
//! - verified-active: audit the rows whose three voters disagree, least lexical confidence first.
//!
//! Each round's ladder is read on the test rows twice: as served, and with the 2-of-3 vote over
//! the least-confident share of lower-rung answers — the share chosen on the VERIFIED rows.

use reflex_study::evolve;
use reflex_study::heimdall::{CachedChat, Chat, Embed, HeimdallChat, HeimdallEmbed, local_chat_model};
use reflex_study::ladder::{Example, Ladder, LadderConfig};
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Gen {
    id: u64,
    text: String,
    label: String,
}

#[derive(Deserialize)]
struct PoolRow {
    pool: u32,
    text: String,
    label: String,
}

fn jsonl<T: for<'de> Deserialize<'de>>(p: &PathBuf) -> Vec<T> {
    std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let run = PathBuf::from(&a[1]);
    let pools: Vec<PoolRow> = jsonl(&PathBuf::from(&a[2]));
    let rounds: u32 = a.get(3).map_or(3, |x| x.parse().unwrap());
    let audit_share: f64 = a.get(4).map_or(0.1, |x| x.parse().unwrap());

    let corpus: BTreeMap<String, Vec<String>> = serde_json::from_slice(&std::fs::read(run.join("corpus.json")).unwrap()).unwrap();
    let seed: Vec<Example> = corpus.iter().flat_map(|(l, ts)| ts.iter().map(|t| Example { text: t.clone(), label: l.clone() })).collect();
    let generated: Vec<Gen> = jsonl(&run.join("generated.jsonl"));
    let verified: Vec<Example> = generated.iter().filter(|g| g.id % 2 == 1).map(|g| Example { text: g.text.clone(), label: g.label.clone() }).collect();
    let test: Vec<Example> = generated.iter().filter(|g| g.id % 2 == 0).map(|g| Example { text: g.text.clone(), label: g.label.clone() }).collect();
    let descriptions: BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(run.join("ladder/descriptions.json")).unwrap()).unwrap();
    let hidden: HashMap<&str, &str> = pools.iter().map(|p| (p.text.as_str(), p.label.as_str())).collect();

    let embed = HeimdallEmbed::new("BAAI/bge-m3", Some(run.join("ladder/embed-cache.json")));
    let chat = CachedChat::new(HeimdallChat::new(&local_chat_model()), run.join("ladder/chat-cache.json"));
    let (em, ch): (Option<&dyn Embed>, Option<&dyn Chat>) = (Some(&embed), Some(&chat));
    let cfg = LadderConfig::default();
    let llm = local_chat_model();
    let fit = |rows: &[Example]| Ladder::fit("student_action", rows, descriptions.clone(), &cfg, em, Some(&llm)).unwrap();
    let saved = run.join("evolve/ladders");
    std::fs::create_dir_all(&saved).unwrap();

    let read = |l: &Ladder, tag: &str| -> serde_json::Value {
        let r = evolve::read(l, &verified, &test, em, ch);
        println!("  {tag:22} {}", r.line());
        json!({"tag": tag, "reading": r, "digest": l.digest()})
    };

    let l0 = fit(&seed);
    let mut report = vec![read(&l0, "round 0 (seed)")];
    for policy in ["verified-random", "verified-active"] {
        let mut current = l0.clone();
        let mut audited: Vec<Example> = Vec::new();
        for r in 1..=rounds {
            let texts: Vec<String> = pools.iter().filter(|p| p.pool == r).map(|p| p.text.clone()).collect();
            let budget = (texts.len() as f64 * audit_share).round() as usize;
            let picked: Vec<String> = if policy == "verified-random" {
                evolve::harvest(&current, &texts, em, ch).into_iter().step_by(10).map(|h| h.text).collect()
            } else {
                let mut v: Vec<(bool, f32, String)> = texts
                    .iter()
                    .map(|t| {
                        let x = evolve::votes(&current, t, em, ch);
                        let unanimous = x.encoder.as_ref() == Some(&x.lexical) && x.llm.as_ref() == Some(&x.lexical);
                        (unanimous, x.lexical_confidence, t.clone())
                    })
                    .collect();
                v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
                v.into_iter().take(budget).map(|x| x.2).collect()
            };
            let disagreeing = picked.len();
            audited.extend(picked.iter().map(|t| Example { text: t.clone(), label: hidden[t.as_str()].to_string() }));
            let mut rows = seed.clone();
            rows.extend(audited.iter().cloned());
            let candidate = fit(&rows);
            let gate = evolve::compare(&current, &candidate, &verified, em, ch);
            current = candidate; // verified-only rows cannot poison; the gate verdict is recorded, not enforced
            current.save(&saved.join(format!("{policy}-r{r}.json"))).unwrap();
            println!("{policy} round {r}: audited {disagreeing} (total {}) · train rows {} · gate would say: {} ({})", audited.len(), rows.len(), gate.promoted, gate.reason);
            let mut v = read(&current, &format!("{policy} r{r}"));
            v["policy"] = json!(policy);
            v["round"] = json!(r);
            v["audited_total"] = json!(audited.len());
            v["gate"] = json!({"promoted": gate.promoted, "reason": gate.reason});
            report.push(v);
        }
    }
    println!("{}", serde_json::to_string(&report).unwrap());
}
