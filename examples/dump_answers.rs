//! Each row's answer from one ladder, as JSONL, for scoring outside Rust.
//!
//! usage: dump_answers LADDER.json ROWS.jsonl OUT.jsonl RUN_DIR
//! ROWS: `{"text", "label"}`; OUT per row: gold, label, rung, confidence, and per-step
//! `[rung, passed, micros]`. Embeddings and llm replies come from RUN_DIR's caches.

use reflex_study::heimdall::{CachedChat, Chat, Embed, HeimdallChat, HeimdallEmbed, local_chat_model};
use reflex_study::ladder::{Example, Ladder};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ladder = Ladder::load(&PathBuf::from(&a[1])).unwrap();
    let rows: Vec<Example> = std::fs::read_to_string(&a[2]).unwrap().lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect();
    let run = PathBuf::from(&a[4]);
    let embed = HeimdallEmbed::new("BAAI/bge-m3", Some(run.join("ladder/embed-cache.json")));
    let chat = CachedChat::new(HeimdallChat::new(&local_chat_model()), run.join("ladder/chat-cache.json"));
    let mut out = std::fs::File::create(&a[3]).unwrap();
    for r in &rows {
        let ans = ladder.decide(&r.text, Some(&embed as &dyn Embed), Some(&chat as &dyn Chat));
        let steps: Vec<_> = ans.steps.iter().map(|s| json!([s.rung, s.passed, s.micros])).collect();
        writeln!(out, "{}", json!({"gold": r.label, "label": ans.label, "rung": ans.rung, "confidence": ans.confidence, "steps": steps, "ladder": ladder.digest()})).unwrap();
    }
}
