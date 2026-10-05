//! The compression drafter: "the corpus is the model".
//!
//! `delta(ctx, option) = |lz4(corpus + ctx)| − |lz4(corpus + ctx + option)|` — how few bytes the
//! option costs once the domain's documents and the request are already in the window. Higher
//! means more predictable. Upstream measured that a short option barely moves the length of a
//! ~300-byte context, so this term alone ties and the centroid cosine does the ranking.

use lz4_flex::block::{compress_into, get_maximum_output_size};

pub struct Drafter {
    corpus: Vec<u8>,
    buf: Vec<u8>,
    out: Vec<u8>,
}

impl Drafter {
    pub fn new(corpus: Vec<u8>) -> Self {
        Self {
            corpus,
            buf: Vec::new(),
            out: Vec::new(),
        }
    }

    pub fn corpus_len(&self) -> usize {
        self.corpus.len()
    }

    fn compressed_len(&mut self, ctx: &[u8], option: &[u8]) -> usize {
        self.buf.clear();
        self.buf.extend_from_slice(&self.corpus);
        self.buf.extend_from_slice(ctx);
        self.buf.extend_from_slice(option);
        let bound = get_maximum_output_size(self.buf.len());
        if self.out.len() < bound {
            self.out.resize(bound, 0);
        }
        compress_into(&self.buf, &mut self.out).expect("output sized by get_maximum_output_size")
    }

    /// Compressed length of `corpus + ctx` — compute once per question, reuse per option.
    pub fn baseline(&mut self, ctx: &[u8]) -> usize {
        self.compressed_len(ctx, &[])
    }

    /// The option's delta against a precomputed [`Self::baseline`].
    pub fn delta(&mut self, baseline: usize, ctx: &[u8], option: &[u8]) -> i32 {
        baseline as i32 - self.compressed_len(ctx, option) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_seen_in_the_corpus_costs_less_than_unseen_text() {
        let mut d = Drafter::new(
            b"rollback is one command when a deploy regresses the error budget".to_vec(),
        );
        let ctx = b"the deploy regressed. ";
        let base = d.baseline(ctx);
        let seen = d.delta(base, ctx, b"rollback is one command");
        let unseen = d.delta(base, ctx, b"qzx vmk plf wty brrr");
        assert!(seen > unseen, "seen {seen} vs unseen {unseen}");
    }
}
