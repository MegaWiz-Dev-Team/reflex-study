//! Text → hashed feature events `(hash, weight)`.
//!
//! Two modes:
//!
//! - [`Tokenizer::Ascii`] is upstream's tokenizer, byte for byte: split on ASCII whitespace,
//!   trim every byte at the token edges that is not an ASCII letter or digit, hash each word
//!   (FNV-1a, ASCII lowercased inside the hash) plus each adjacent word pair at weight 0.5.
//!   A Thai clause is non-ASCII end to end, so the trim empties it and it contributes nothing.
//! - [`Tokenizer::Unicode`] keeps every ASCII token's events identical to `Ascii` and adds:
//!   letters and digits of any script survive the edge trim, non-ASCII words are lowercased,
//!   and a run of Thai characters (no spaces between words) becomes overlapping character
//!   trigrams (weight 1.0) and bigrams (weight 0.5). A few Thai spellings are normalized first.

/// Word-hash salt (upstream constant).
pub const WORD_SALT: u64 = 0x3456_7890_1234_5678;
/// Word-pair hash salt (upstream constant).
pub const BIGRAM_SALT: u64 = 0x0f1e_2d3c_4b5a_6978;
/// Word-pair weight (upstream constant).
pub const BIGRAM_WEIGHT: f32 = 0.5;
/// Thai character-trigram salt.
const THAI_TRI_SALT: u64 = 0x7a3c_51e9_0d24_b86f;
/// Thai character-bigram salt.
const THAI_BI_SALT: u64 = 0x2b9d_04f7_c61e_8a35;
/// Thai character-trigram weight.
const THAI_TRI_WEIGHT: f32 = 1.0;
/// Thai character-bigram weight.
const THAI_BI_WEIGHT: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tokenizer {
    /// Upstream's tokenizer (parity mode).
    #[default]
    Ascii,
    /// Upstream's events for ASCII text, plus Thai character n-grams and non-ASCII words.
    Unicode,
}

impl Tokenizer {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ascii" => Some(Self::Ascii),
            "unicode" => Some(Self::Unicode),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ascii => "ascii",
            Self::Unicode => "unicode",
        }
    }

    /// Call `emit(hash, weight)` once per feature event of `text`.
    pub fn features(self, text: &str, mut emit: impl FnMut(u64, f32)) {
        match self {
            Self::Ascii => ascii_features(text, &mut emit),
            Self::Unicode => unicode_features(text, &mut emit),
        }
    }
}

/// FNV-1a 64 with ASCII uppercase folded to lowercase inside the hash (upstream's
/// `fnv1a_word`; the fold also applies to the bytes of a word-pair key).
#[inline]
pub fn fnv1a(bytes: &[u8], salt: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ salt;
    for &b in bytes {
        let b = if b.is_ascii_uppercase() { b + 32 } else { b };
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn trim_ascii(raw: &[u8]) -> &[u8] {
    let keep = |b: u8| b.is_ascii_alphanumeric();
    let mut a = 0;
    let mut z = raw.len();
    while a < z && !keep(raw[a]) {
        a += 1;
    }
    while z > a && !keep(raw[z - 1]) {
        z -= 1;
    }
    &raw[a..z]
}

/// One word event, plus the pair event with the previous word.
fn word(t: &[u8], prev: &mut Option<u64>, emit: &mut impl FnMut(u64, f32)) {
    let h = fnv1a(t, WORD_SALT);
    emit(h, 1.0);
    if let Some(p) = *prev {
        let mut pair = [0u8; 16];
        pair[..8].copy_from_slice(&p.to_le_bytes());
        pair[8..].copy_from_slice(&h.to_le_bytes());
        emit(fnv1a(&pair, BIGRAM_SALT), BIGRAM_WEIGHT);
    }
    *prev = Some(h);
}

fn ascii_features(text: &str, emit: &mut impl FnMut(u64, f32)) {
    let mut prev = None;
    for raw in text.as_bytes().split(|b| b.is_ascii_whitespace()) {
        let t = trim_ascii(raw);
        // A punctuation-only token is skipped WITHOUT breaking the pair chain.
        if !t.is_empty() {
            word(t, &mut prev, emit);
        }
    }
}

/// Thai letters, vowels and tone marks (U+0E01–U+0E3A, U+0E40–U+0E4E). Thai digits are
/// normalized to ASCII before this is asked; ฿ and the Thai punctuation marks are not letters.
#[inline]
pub fn is_thai(c: char) -> bool {
    matches!(c, '\u{0E01}'..='\u{0E3A}' | '\u{0E40}'..='\u{0E4E}')
}

#[inline]
fn is_separator(c: char) -> bool {
    c.is_ascii_whitespace() || c == '\u{200B}' || (!c.is_ascii() && c.is_whitespace())
}

#[inline]
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || (!c.is_ascii() && c.is_alphanumeric())
}

/// The handful of compatibility folds that matter here: Thai digits → ASCII digits,
/// NIKHAHIT + SARA AA → SARA AM, full-width ASCII → ASCII, and the invisible joiners dropped.
/// (Full NFKC would split SARA AM the other way; one direction applied everywhere is what counts.)
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '\u{0E50}'..='\u{0E59}' => out.push((b'0' + (c as u32 - 0x0E50) as u8) as char),
            '\u{0E4D}' if it.peek() == Some(&'\u{0E32}') => {
                it.next();
                out.push('\u{0E33}');
            }
            '\u{FF01}'..='\u{FF5E}' => out.push(char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)),
            '\u{3000}' => out.push(' '),
            '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{00AD}' => {}
            _ => out.push(c),
        }
    }
    out
}

fn trim_word(s: &str) -> &str {
    s.trim_matches(|c: char| !is_word_char(c))
}

/// A non-Thai word in Unicode mode: ASCII words hash exactly as in `Ascii` mode.
fn unicode_word(t: &str, prev: &mut Option<u64>, emit: &mut impl FnMut(u64, f32)) {
    if t.is_ascii() {
        word(t.as_bytes(), prev, emit);
    } else {
        word(t.to_lowercase().as_bytes(), prev, emit);
    }
}

fn thai_ngrams(run: &str, emit: &mut impl FnMut(u64, f32)) {
    let idx: Vec<usize> = run
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(run.len()))
        .collect();
    let n = idx.len() - 1;
    if n == 1 {
        emit(fnv1a(run.as_bytes(), THAI_BI_SALT), THAI_BI_WEIGHT);
        return;
    }
    for i in 0..n.saturating_sub(1) {
        emit(fnv1a(run[idx[i]..idx[i + 2]].as_bytes(), THAI_BI_SALT), THAI_BI_WEIGHT);
    }
    for i in 0..n.saturating_sub(2) {
        emit(fnv1a(run[idx[i]..idx[i + 3]].as_bytes(), THAI_TRI_SALT), THAI_TRI_WEIGHT);
    }
}

fn unicode_features(text: &str, emit: &mut impl FnMut(u64, f32)) {
    let text = normalize(text);
    let mut prev = None;
    for raw in text.split(is_separator) {
        if !raw.chars().any(is_thai) {
            let t = trim_word(raw);
            if !t.is_empty() {
                unicode_word(t, &mut prev, emit);
            }
            continue;
        }
        // Mixed or pure Thai: walk maximal Thai / non-Thai runs.
        let mut start = 0;
        let mut in_thai = None;
        for (i, c) in raw.char_indices().chain(std::iter::once((raw.len(), ' '))) {
            let thai = i < raw.len() && is_thai(c);
            if in_thai != Some(thai) {
                if let Some(was_thai) = in_thai {
                    let run = &raw[start..i];
                    if was_thai {
                        thai_ngrams(run, emit);
                        // No word-pair event spans a Thai run.
                        prev = None;
                    } else {
                        let t = trim_word(run);
                        if !t.is_empty() {
                            unicode_word(t, &mut prev, emit);
                        }
                    }
                }
                start = i;
                in_thai = Some(thai);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(tok: Tokenizer, s: &str) -> Vec<(u64, f32)> {
        let mut v = Vec::new();
        tok.features(s, |h, w| v.push((h, w)));
        v
    }

    #[test]
    fn ascii_text_gives_the_same_events_in_both_modes() {
        let s = "Our deploy regressed the error budget after the rollout — run the ROLLBACK, \
                 and verify the health endpoints (don't skip e-mail).";
        assert_eq!(events(Tokenizer::Ascii, s), events(Tokenizer::Unicode, s));
    }

    #[test]
    fn thai_is_empty_in_ascii_mode_and_not_in_unicode_mode() {
        let s = "ลูกค้าโดนตัดบัตรซ้ำสองครั้ง ขอคืนเงิน";
        assert!(events(Tokenizer::Ascii, s).is_empty());
        let u = events(Tokenizer::Unicode, s);
        // 27 chars → 26 bigrams + 25 trigrams; 9 chars → 8 bigrams + 7 trigrams.
        assert_eq!(u.len(), 26 + 25 + 8 + 7);
    }

    #[test]
    fn mixed_token_hashes_the_ascii_part_as_a_word() {
        let mut ascii_word = Vec::new();
        Tokenizer::Ascii.features("CT", |h, w| ascii_word.push((h, w)));
        let mixed = events(Tokenizer::Unicode, "ตรวจCTสมอง");
        assert!(mixed.contains(&ascii_word[0]));
    }

    #[test]
    fn normalization_folds_thai_digits_and_sara_am() {
        assert_eq!(normalize("๑๒๓ น\u{0E4D}\u{0E32}"), "123 น\u{0E33}");
        assert_eq!(
            events(Tokenizer::Unicode, "น้ำ ๒"),
            events(Tokenizer::Unicode, "น้\u{0E4D}\u{0E32} 2")
        );
    }
}
