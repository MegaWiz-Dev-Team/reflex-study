//! Two more ways to feed upper-rung answers back without planting silent errors.
//!
//! usage: evolve_tri_memory RUN_DIR POOLS.jsonl [ROUNDS=3] [PAIR_ERROR_MAX=0.1]
//! (same RUN_DIR layout as `evolve_rounds`; ladders saved to RUN_DIR/evolve/ladders/)
//!
//! - tri-training (Zhou & Li 2005), no human audit: a row teaches the lexical rung only when the
//!   encoder and the llm agree on it, and teaches the encoder only when the lexical rung and the
//!   llm agree. A round's rows for a rung are taken only if that agreeing pair's error rate,
//!   measured on the VERIFIED rows, is at most PAIR_ERROR_MAX. Both heads are refitted on their
//!   own sets (`Ladder::fit_split`); the new ladder always replaces the old (like naive).
//! - memory: the heads stay as trained on the seed; the seed rows and every audited row (every
//!   10th row the upper rungs answered — the gated/verified-random audit) go into the encoder's
//!   memory via `Ladder::remember`. Nothing is retrained.

use reflex_study::evolve::{self, Votes};
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

/// Among verified rows where the two voters agree, the share they get wrong (`None` if they
/// never agree).
fn pair_error(v: &[Votes], gold: &[Example], a: fn(&Votes) -> Option<&String>, b: fn(&Votes) -> Option<&String>) -> Option<f64> {
    let agreed: Vec<bool> = v.iter().zip(gold).filter_map(|(x, g)| match (a(x), b(x)) {
        (Some(p), Some(q)) if p == q => Some(*p == g.label),
        _ => None,
    }).collect();
    (!agreed.is_empty()).then(|| agreed.iter().filter(|ok| !**ok).count() as f64 / agreed.len() as f64)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let run = PathBuf::from(&a[1]);
    let pools: Vec<PoolRow> = jsonl(&PathBuf::from(&a[2]));
    let rounds: u32 = a.get(3).map_or(3, |x| x.parse().unwrap());
    let pair_max: f64 = a.get(4).map_or(0.1, |x| x.parse().unwrap());

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
    let saved = run.join("evolve/ladders");
    std::fs::create_dir_all(&saved).unwrap();
    let mut report = Vec::new();

    let l0 = Ladder::fit("student_action", &seed, descriptions.clone(), &cfg, em, Some(&llm)).unwrap();

    // ── tri-training (thresholds out-of-fold on the training rows, then on the verified rows) ──
    for calibration in ["oof", "verified"] {
    let tag = if calibration == "oof" { "tri" } else { "tri-cal" };
    let (mut lex_rows, mut enc_rows) = (seed.clone(), seed.clone());
    let mut current = l0.clone();
    if calibration == "verified" {
        current.recalibrate(&verified, em).unwrap();
        let reading = evolve::read(&current, &verified, &test, em, ch);
        println!("tri-cal round 0 (seed ladder, verified thresholds)\n  {tag} r0  {}", reading.line());
        report.push(json!({"policy": tag, "round": 0, "reading": reading, "digest": current.digest()}));
    }
    for r in 1..=rounds {
        let texts: Vec<String> = pools.iter().filter(|p| p.pool == r).map(|p| p.text.clone()).collect();
        let vp: Vec<Votes> = texts.iter().map(|t| evolve::votes(&current, t, em, ch)).collect();
        let vd: Vec<Votes> = verified.iter().map(|x| evolve::votes(&current, &x.text, em, ch)).collect();
        let e_enc_llm = pair_error(&vd, &verified, |v| v.encoder.as_ref(), |v| v.llm.as_ref());
        let e_lex_llm = pair_error(&vd, &verified, |v| Some(&v.lexical), |v| v.llm.as_ref());
        let for_lex: Vec<Example> = texts.iter().zip(&vp).filter_map(|(t, v)| match (&v.encoder, &v.llm) {
            (Some(p), Some(q)) if p == q => Some(Example { text: t.clone(), label: p.clone() }),
            _ => None,
        }).collect();
        let for_enc: Vec<Example> = texts.iter().zip(&vp).filter_map(|(t, v)| match &v.llm {
            Some(q) if *q == v.lexical => Some(Example { text: t.clone(), label: q.clone() }),
            _ => None,
        }).collect();
        let take_lex = e_enc_llm.is_some_and(|e| e <= pair_max);
        let take_enc = e_lex_llm.is_some_and(|e| e <= pair_max);
        let right = |rows: &[Example]| rows.iter().filter(|x| hidden[x.text.as_str()] == x.label).count();
        if take_lex {
            lex_rows.extend(for_lex.iter().cloned());
        }
        if take_enc {
            enc_rows.extend(for_enc.iter().cloned());
        }
        current = Ladder::fit_split("student_action", &lex_rows, &enc_rows, descriptions.clone(), &cfg, em, Some(&llm)).unwrap();
        if calibration == "verified" {
            current.recalibrate(&verified, em).unwrap();
        }
        current.save(&saved.join(format!("{tag}-r{r}.json"))).unwrap();
        println!(
            "{tag} round {r}: for lexical {} (right {}), pair error enc+llm {:?} → taken {take_lex} · for encoder {} (right {}), pair error lex+llm {:?} → taken {take_enc} · rows lexical {} encoder {}",
            for_lex.len(), right(&for_lex), e_enc_llm.map(|e| (e * 1000.0).round() / 1000.0), for_enc.len(), right(&for_enc), e_lex_llm.map(|e| (e * 1000.0).round() / 1000.0), lex_rows.len(), enc_rows.len()
        );
        let reading = evolve::read(&current, &verified, &test, em, ch);
        println!("  {tag} r{r}  {}", reading.line());
        report.push(json!({"policy": tag, "round": r, "for_lexical": [for_lex.len(), right(&for_lex)], "for_encoder": [for_enc.len(), right(&for_enc)],
            "pair_error_enc_llm": e_enc_llm, "pair_error_lex_llm": e_lex_llm, "taken": [take_lex, take_enc], "reading": reading, "digest": current.digest()}));
    }
    }

    // ── memory ──
    let mut current = l0.clone();
    let n = current.remember(&seed, &embed).unwrap();
    current.save(&saved.join("memory-r0.json")).unwrap();
    let reading = evolve::read(&current, &verified, &test, em, ch);
    let cutoff = |l: &Ladder| l.encoder.as_ref().and_then(|e| e.memory.as_ref()).and_then(|m| m.cutoff);
    println!("memory round 0: {n} seed rows remembered, cutoff {:?}\n  mem r0   {}", cutoff(&current), reading.line());
    report.push(json!({"policy": "memory", "round": 0, "memory": n, "cutoff": cutoff(&current), "reading": reading, "digest": current.digest()}));
    for r in 1..=rounds {
        let texts: Vec<String> = pools.iter().filter(|p| p.pool == r).map(|p| p.text.clone()).collect();
        let audited: Vec<Example> = evolve::harvest(&current, &texts, em, ch)
            .into_iter()
            .step_by(10)
            .map(|h| Example { label: hidden[h.text.as_str()].to_string(), text: h.text })
            .collect();
        let n = current.remember(&audited, &embed).unwrap();
        current.save(&saved.join(format!("memory-r{r}.json"))).unwrap();
        let reading = evolve::read(&current, &verified, &test, em, ch);
        println!("memory round {r}: +{} audited → {n} remembered, cutoff {:?}\n  mem r{r}   {}", audited.len(), cutoff(&current), reading.line());
        report.push(json!({"policy": "memory", "round": r, "audited": audited.len(), "memory": n, "cutoff": cutoff(&current), "reading": reading, "digest": current.digest()}));
    }
    println!("{}", serde_json::to_string(&report).unwrap());
}
