//! Corpus-distance gate: how close is the question to the nearest document of the routed
//! domain? `confidence = sigmoid(scale · (max_cosine − mid))`. Under `threshold` → abstain.

use crate::embed::dot;
use crate::sigmoid;

/// Upstream midpoint: in-corpus max-cosine ≈ 0.68–0.70, off-corpus ≈ 0.00–0.02 at their birth
/// measurement (English, 256 buckets). A different tokenizer moves both populations — measure.
pub const GATE_MID: f32 = 0.35;
/// Upstream slope.
pub const GATE_SCALE: f32 = 8.0;

/// Re-normalize (upstream normalizes again inside the gate; kept for bit-parity).
fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.iter().map(|x| x * inv).collect()
    } else {
        v.to_vec()
    }
}

pub struct DistanceGate {
    rows: Vec<Vec<f32>>,
    mid: f32,
    scale: f32,
}

impl DistanceGate {
    pub fn new(rows: &[Vec<f32>], mid: f32, scale: f32) -> Self {
        assert!(mid.is_finite() && scale.is_finite() && scale > 0.0);
        Self {
            rows: rows.iter().map(|r| unit(r)).collect(),
            mid,
            scale,
        }
    }

    /// Max cosine to any registered document (−1 for an empty gate).
    pub fn max_similarity(&self, q: &[f32]) -> f32 {
        let q = unit(q);
        self.rows.iter().fold(-1.0f32, |m, r| m.max(dot(r, &q)))
    }

    pub fn confidence(&self, q: &[f32]) -> f32 {
        sigmoid(self.scale * (self.max_similarity(q) - self.mid))
    }
}
