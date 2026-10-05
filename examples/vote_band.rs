//! Every rung's own answer for every row, so a "vote when unsure" policy can be read offline.
//!
//! usage: vote_band LADDER.json ROWS.jsonl OUT.jsonl RUN_DIR
//! ROWS: `{"text", "label"}`. OUT, per row: gold, the ladder's own answer and rung, and the
//! lexical / encoder / llm answers each computed regardless of the gates.

use reflex_study::features;
use reflex_study::heimdall::{CachedChat, Chat, Embed, HeimdallChat, HeimdallEmbed, local_chat_model};
use reflex_study::ladder::{Example, Ladder};
use reflex_study::tokenize::Tokenizer;
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
    let tok = Tokenizer::parse(&ladder.tokenizer).unwrap();
    let enc = ladder.encoder.as_ref().unwrap();
    let mut out = std::fs::File::create(&a[3]).unwrap();
    for r in &rows {
        let ans = ladder.decide(&r.text, Some(&embed as &dyn Embed), Some(&chat as &dyn Chat));
        let x = features::lexical(tok, &r.text);
        let (lc, ls) = ladder.lexical.top(&x);
        let (lsim, _) = ladder.lexical_gate.check(&x);
        let e = features::dense(&embed.embed(&[r.text.clone()]).unwrap()[0]);
        let (ec, es) = enc.head.top(&e);
        let llm = chat.complete(&ladder.llm_prompt(), &r.text).ok().and_then(|reply| ladder.parse_llm(&reply));
        writeln!(out, "{}", json!({
            "text": r.text, "gold": r.label,
            "answer": ans.label, "rung": ans.rung,
            "lexical": ladder.labels[lc], "lexical_conf": ls, "lexical_sim": lsim, "lexical_threshold": ladder.lexical_fit.threshold,
            "encoder": ladder.labels[ec], "encoder_conf": es,
            "llm": llm,
        })).unwrap();
    }
}
