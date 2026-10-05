//! The two model calls the upper rungs make, both through the local Heimdall gateway
//! (`HEIMDALL_API_URL`, default `http://127.0.0.1:8080/v1`; key from `HEIMDALL_API_KEY`).
//! Model names carry no provider prefix, so Heimdall serves them on this machine: bge-m3 on its
//! fastembed CPU route, the chat model on the local backend. Nothing here talks to a cloud API.
//!
//! The traits let tests and offline evals substitute a fake; tests never reach a live gateway.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

pub const DEFAULT_EMBED_MODEL: &str = "BAAI/bge-m3";
/// The name the shared local backend is loaded and called under. Heimdall compares the
/// requested model to the active one as an EXACT string and runs `hotswap.sh` (restarting mlx
/// for every client) on any mismatch — the same weights under another name included. Measured
/// 4 Oct 2026: calls naming `mlx-community/gemma-4-26b-a4b-it-4bit` ping-ponged swaps with the
/// production clients' `gemma-4-26b` and returned 502 during each restart.
pub const DEFAULT_CHAT_MODEL: &str = "gemma-4-26b";

/// The local model this machine's gateway serves (`HEIMDALL_LOCAL_MODEL`, default
/// [`DEFAULT_CHAT_MODEL`]). [`HeimdallChat`] refuses any other name rather than swap it.
pub fn local_chat_model() -> String {
    std::env::var("HEIMDALL_LOCAL_MODEL").unwrap_or_else(|_| DEFAULT_CHAT_MODEL.into())
}

pub trait Embed: Send + Sync {
    /// Model identity, recorded in a ladder so a different encoder cannot be loaded silently.
    fn model(&self) -> &str;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

pub trait Chat: Send + Sync {
    fn model(&self) -> &str;
    fn complete(&self, system: &str, user: &str) -> Result<String, String>;
}

fn base_url() -> String {
    std::env::var("HEIMDALL_API_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into())
}

fn post(path: &str, body: Value) -> Result<Value, String> {
    let key = std::env::var("HEIMDALL_API_KEY").map_err(|_| "HEIMDALL_API_KEY is not set".to_string())?;
    let url = format!("{}/{path}", base_url().trim_end_matches('/'));
    ureq::post(&url)
        .set("Authorization", &format!("Bearer {key}"))
        .send_json(body)
        .map_err(|e| format!("{url}: {e}"))?
        .into_json::<Value>()
        .map_err(|e| format!("{url}: {e}"))
}

/// bge-m3 (or another encoder) through Heimdall, with an optional on-disk cache keyed by
/// BLAKE3(model, text) so re-training and re-evaluating do not re-embed.
pub struct HeimdallEmbed {
    model: String,
    cache_path: Option<PathBuf>,
    cache: Mutex<HashMap<String, Vec<f32>>>,
}

impl HeimdallEmbed {
    pub fn new(model: &str, cache_path: Option<PathBuf>) -> Self {
        let cache = cache_path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Self {
            model: model.into(),
            cache_path,
            cache: Mutex::new(cache),
        }
    }

    fn key(&self, text: &str) -> String {
        blake3::hash(format!("{}\0{text}", self.model).as_bytes()).to_hex().to_string()
    }
}

impl Embed for HeimdallEmbed {
    fn model(&self) -> &str {
        &self.model
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut cache = self.cache.lock().expect("embed cache lock");
        let missing: Vec<&String> = texts.iter().filter(|t| !cache.contains_key(&self.key(t))).collect();
        for chunk in missing.chunks(16) {
            let d = post("embeddings", json!({"model": self.model, "input": chunk}))?;
            let mut data: Vec<&Value> = d["data"].as_array().ok_or("embeddings: no data array")?.iter().collect();
            data.sort_by_key(|e| e["index"].as_u64());
            if data.len() != chunk.len() {
                return Err(format!("embeddings: asked {} got {}", chunk.len(), data.len()));
            }
            for (t, e) in chunk.iter().zip(data) {
                let v: Vec<f32> = serde_json::from_value(e["embedding"].clone()).map_err(|e| e.to_string())?;
                cache.insert(self.key(t), v);
            }
        }
        if !missing.is_empty()
            && let Some(p) = &self.cache_path
        {
            std::fs::write(p, serde_json::to_vec(&*cache).expect("cache serializes")).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        Ok(texts.iter().map(|t| cache[&self.key(t)].clone()).collect())
    }
}

/// A local chat model through Heimdall, temperature 0.
pub struct HeimdallChat {
    model: String,
}

impl HeimdallChat {
    pub fn new(model: &str) -> Self {
        Self { model: model.into() }
    }
}

impl Chat for HeimdallChat {
    fn model(&self) -> &str {
        &self.model
    }

    fn complete(&self, system: &str, user: &str) -> Result<String, String> {
        let local = local_chat_model();
        if self.model != local {
            return Err(format!(
                "refused: {} is not the gateway's local model {local} — calling it would hot-swap the shared backend",
                self.model
            ));
        }
        let d = post(
            "chat/completions",
            json!({
                "model": self.model,
                "temperature": 0,
                "max_tokens": 16,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }),
        )?;
        d["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("chat: no content in {d}"))
    }
}

/// A [`Chat`] with an on-disk reply cache keyed by BLAKE3(model, system, user): the same
/// question is never asked twice, and a re-run replays the recorded replies exactly.
pub struct CachedChat<C: Chat> {
    inner: C,
    path: PathBuf,
    cache: Mutex<HashMap<String, String>>,
}

impl<C: Chat> CachedChat<C> {
    pub fn new(inner: C, path: PathBuf) -> Self {
        let cache = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        Self { inner, path, cache: Mutex::new(cache) }
    }
}

impl<C: Chat> Chat for CachedChat<C> {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn complete(&self, system: &str, user: &str) -> Result<String, String> {
        let key = blake3::hash(format!("{}\0{system}\0{user}", self.inner.model()).as_bytes()).to_hex().to_string();
        if let Some(r) = self.cache.lock().expect("chat cache lock").get(&key) {
            return Ok(r.clone());
        }
        let reply = self.inner.complete(system, user)?;
        let mut cache = self.cache.lock().expect("chat cache lock");
        cache.insert(key, reply.clone());
        std::fs::write(&self.path, serde_json::to_vec(&*cache).expect("cache serializes")).map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_local_chat_model_is_refused_before_any_request() {
        // Points at a dead port: had the guard let the call through, the error would be a
        // connection error, not the refusal.
        unsafe { std::env::set_var("HEIMDALL_API_URL", "http://127.0.0.1:1/v1") };
        let e = HeimdallChat::new("mlx-community/gemma-4-26b-a4b-it-4bit").complete("s", "u").unwrap_err();
        assert!(e.starts_with("refused:"), "{e}");
    }
}
