//! A study re-implementation of the Reflex modelless decision engine
//! (<https://reflex.gist.rs>, upstream source MIT: gist-rs/riir-reflex + katopz/katgpt-rs).
//!
//! The pipeline, per question:
//!
//! 1. embed: state + prompt (+ criteria) as a hashed bag of words and word bigrams;
//! 2. route: the domain whose document centroid is closest;
//! 3. score each option: LZ4 compressed-length delta against the routed domain's corpus,
//!    plus the state's cosine to the option's own domain centroid when options name domains;
//! 4. normalize: sigmoid per option, then divide by the sum;
//! 5. confidence: 1 − normalized entropy (≤ 8 options) or the max probability;
//! 6. abstain: confidence under the score threshold, or the state far from the routed corpus.
//!
//! `Tokenizer::Ascii` reproduces upstream exactly (tests/parity.rs checks it against the
//! release binary). `Tokenizer::Unicode` is the extension this study adds: upstream drops
//! every non-ASCII byte at token edges, so Thai text embeds to the zero vector and always
//! abstains; the Unicode mode turns Thai runs into character n-grams instead.

pub mod calibrate;
pub mod corpus;
pub mod drafter;
pub mod embed;
pub mod engine;
pub mod evolve;
pub mod features;
pub mod gate;
pub mod heimdall;
pub mod http;
pub mod ladder;
pub mod threshold;
pub mod tokenize;
pub mod wire;

/// The lexical head (one-vs-all logistic, NBSVM, soft targets) — from the ultra-instinct crate.
pub use ultra_instinct::linear;

/// Sigmoid in the two-branch form (no overflow for large |x|).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
