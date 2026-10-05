//! Three rounds of the update loop, two policies, one fixed test set — does feeding upper-rung
//! answers back into the lower rungs help, and does it plant silent errors?
//!
//! usage: evolve_rounds RUN_DIR POOLS.jsonl [ROUNDS=3] [AUDIT_EVERY=10]
//!
//! RUN_DIR holds `corpus.json` (seed rows), `generated.jsonl` (odd ids = the verified set,
//! even ids = test), `ladder/descriptions.json`, and the embed/chat caches. POOLS rows are
//! `{"pool": r, "text", "label"}`; the label stays hidden from the loop — it is read only to
//! play the human auditor and to measure teacher quality.
//!
//! - naive: every harvested upper-rung label is trained in; the new ladder always replaces the old.
//! - gated: every AUDIT_EVERY-th harvested row is corrected to its true label (the human
//!   audit); the candidate replaces the current ladder only if `evolve::compare` passes on the
//!   verified set.

use reflex_study::evolve::{self, Harvest};
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
    std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let run = PathBuf::from(&a[1]);
    let pools: Vec<PoolRow> = jsonl(&PathBuf::from(&a[2]));
    let rounds: u32 = a.get(3).map_or(3, |x| x.parse().unwrap());
    let audit_every: usize = a.get(4).map_or(10, |x| x.parse().unwrap());

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

    let l0 = fit(&seed);
    let saved = run.join("evolve/ladders");
    std::fs::create_dir_all(&saved).unwrap();
    l0.save(&saved.join("r0.json")).unwrap();
    let (s0, _) = evolve::score(&l0, &test, em, ch);
    println!("seed {} rows · verified {} · test {} · pools {} · audit every {audit_every}", seed.len(), verified.len(), test.len(), pools.len());
    println!("round 0 (seed ladder) test: right {}/{} · by rung {:?} · silent errors {} · llm calls {}", s0.right, s0.n, s0.by_rung, s0.silent_errors, s0.llm_calls);
    let mut report = vec![json!({"policy": "both", "round": 0, "test": s0})];

    for policy in ["naive", "gated"] {
        let mut current = l0.clone();
        let mut accepted: Vec<Example> = Vec::new();
        for r in 1..=rounds {
            let texts: Vec<String> = pools.iter().filter(|p| p.pool == r).map(|p| p.text.clone()).collect();
            let h: Vec<Harvest> = evolve::harvest(&current, &texts, em, ch);
            let teacher_right = h.iter().filter(|x| hidden[x.text.as_str()] == x.label).count();
            let mut corrected = 0;
            for (i, x) in h.iter().enumerate() {
                let mut label = x.label.clone();
                if policy == "gated" && i % audit_every == 0 {
                    let truth = hidden[x.text.as_str()].to_string();
                    corrected += (truth != label) as usize;
                    label = truth;
                }
                accepted.push(Example { text: x.text.clone(), label });
            }
            let mut rows = seed.clone();
            rows.extend(accepted.iter().cloned());
            let candidate = fit(&rows);
            let (promoted, reason) = if policy == "naive" {
                (true, "naive: always".to_string())
            } else {
                let p = evolve::compare(&current, &candidate, &verified, em, ch);
                (p.promoted, p.reason)
            };
            if promoted {
                current = candidate;
            }
            current.save(&saved.join(format!("{policy}-r{r}.json"))).unwrap();
            let (s, _) = evolve::score(&current, &test, em, ch);
            println!(
                "{policy:5} round {r}: harvested {} (teacher right {teacher_right}, audited fixes {corrected}) · train rows {} · promoted {promoted} ({reason}) · TEST right {}/{} · by rung {:?} · silent errors {} (lexical {}) · llm calls {}",
                h.len(),
                rows.len(),
                s.right,
                s.n,
                s.by_rung,
                s.silent_errors,
                s.lexical_errors,
                s.llm_calls
            );
            report.push(json!({
                "policy": policy, "round": r, "harvested": h.len(), "teacher_right": teacher_right,
                "audited_fixes": corrected, "train_rows": rows.len(), "promoted": promoted, "reason": reason,
                "test": s, "ladder_digest": current.digest(),
            }));
        }
    }
    println!("{}", serde_json::to_string(&report).unwrap());
}
