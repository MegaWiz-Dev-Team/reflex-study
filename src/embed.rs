//! Hashed-feature embedding: feature events → `dim` buckets with the sign trick, then
//! L2-normalized. A pure function of the text; nothing is learned.

use crate::tokenize::Tokenizer;
use std::collections::{HashMap, HashSet};

/// Upstream's width (256 buckets; 64 washed the distance gate out at their birth measurement).
pub const EMBED_DIM: usize = 256;

#[derive(Clone, Debug)]
pub struct Embedder {
    pub tokenizer: Tokenizer,
    pub dim: usize,
    /// Optional per-feature IDF weights fitted from the corpus documents (not upstream).
    idf: Option<Idf>,
}

/// Smoothed IDF `ln((N+1)/(df+1)) + 1` per feature hash; a feature no document carries gets
/// the maximum weight `ln(N+1) + 1`.
#[derive(Clone, Debug)]
struct Idf {
    weights: HashMap<u64, f32>,
    unseen: f32,
}

impl Embedder {
    pub fn new(tokenizer: Tokenizer, dim: usize) -> Self {
        assert!(dim > 0, "embedding width must be positive");
        Self { tokenizer, dim, idf: None }
    }

    /// Fit IDF weights from the documents (deterministic: a pure function of the corpus).
    pub fn with_idf(mut self, docs: &[&str]) -> Self {
        let mut df: HashMap<u64, u32> = HashMap::new();
        for doc in docs {
            let mut seen = HashSet::new();
            self.tokenizer.features(doc, |h, _| {
                seen.insert(h);
            });
            for h in seen {
                *df.entry(h).or_default() += 1;
            }
        }
        let n = docs.len() as f32;
        self.idf = Some(Idf {
            weights: df
                .into_iter()
                .map(|(h, d)| (h, ((n + 1.0) / (d as f32 + 1.0)).ln() + 1.0))
                .collect(),
            unseen: (n + 1.0).ln() + 1.0,
        });
        self
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        self.tokenizer.features(text, |h, w| {
            let bucket = (h as usize) % self.dim;
            let sign = if h >> 63 == 1 { -1.0 } else { 1.0 };
            let w = match &self.idf {
                Some(idf) => w * idf.weights.get(&h).copied().unwrap_or(idf.unseen),
                None => w,
            };
            v[bucket] += w * sign;
        });
        // A zero vector stays zero ("no signal", never NaN).
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            let inv = 1.0 / n;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
        v
    }
}

/// Sequential dot product (the order matters for bit-parity with upstream).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// Unit centroid of `rows`: mean, then divided by its norm (the routing direction).
pub fn unit_centroid(rows: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; dim];
    for r in rows {
        for (a, x) in acc.iter_mut().zip(r) {
            *a += x;
        }
    }
    let k = rows.len() as f32;
    for x in acc.iter_mut() {
        *x /= k;
    }
    let n = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in acc.iter_mut() {
            *x /= n;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_unit_norm() {
        let e = Embedder::new(Tokenizer::Ascii, EMBED_DIM);
        let a = e.embed("refund the customer invoice balance");
        let b = e.embed("refund the customer invoice balance");
        assert_eq!(a, b);
        let n: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn thai_is_the_zero_vector_in_ascii_mode_only() {
        let s = "ขอคืนเงินค่าบริการที่ถูกเรียกเก็บซ้ำ";
        let ascii = Embedder::new(Tokenizer::Ascii, EMBED_DIM).embed(s);
        assert!(ascii.iter().all(|x| *x == 0.0));
        let uni = Embedder::new(Tokenizer::Unicode, EMBED_DIM).embed(s);
        assert!(uni.iter().any(|x| *x != 0.0));
    }
}
