//! Sparse feature bags for the trained rungs: hashed n-gram events (the same tokenizer as the
//! engine) folded into `SPARSE_VOCAB` buckets, or a dense embedding, both L2-normalized.

use crate::tokenize::Tokenizer;
use std::collections::BTreeMap;

/// Hash space of the lexical rung (Instinct's width).
pub const SPARSE_VOCAB: u64 = 1 << 17;

/// `(feature id, value)` sorted by id, ids unique (shared with the ultra-instinct crate).
pub use ultra_instinct::Sparse;

fn normalized(m: BTreeMap<u32, f32>) -> Sparse {
    let n = m.values().map(|v| v * v).sum::<f32>().sqrt();
    m.into_iter().map(|(k, v)| (k, if n > 0.0 { v / n } else { v })).collect()
}

/// Weighted event counts of `text`, L2-normalized. Empty text (or, in `Ascii` mode, Thai) → empty.
pub fn lexical(tokenizer: Tokenizer, text: &str) -> Sparse {
    let mut m = BTreeMap::new();
    tokenizer.features(text, |h, w| *m.entry((h % SPARSE_VOCAB) as u32).or_insert(0.0) += w);
    normalized(m)
}

/// A dense embedding as a sparse row over ids `0..len`, L2-normalized.
pub fn dense(v: &[f32]) -> Sparse {
    normalized(v.iter().enumerate().map(|(i, x)| (i as u32, *x)).collect())
}

/// Binary presence of each hashed event (value 1.0, ids sorted, duplicates collapsed) — the
/// NBSVM input, where the per-class log-count ratio carries the weight instead of the count.
pub fn presence(tokenizer: Tokenizer, text: &str) -> Sparse {
    let mut m = BTreeMap::new();
    tokenizer.features(text, |h, _| {
        m.insert((h % SPARSE_VOCAB) as u32, 1.0f32);
    });
    m.into_iter().collect()
}
