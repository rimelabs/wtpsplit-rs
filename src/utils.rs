//! Utility functions for wtpsplit

use crate::constants::PRIMES;

/// Sigmoid activation function
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid to a slice of values in place
pub fn sigmoid_inplace(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = sigmoid(*v);
    }
}

/// Encode text as Unicode code points (ordinals)
pub fn encode_text(text: &str) -> Vec<i64> {
    text.chars().map(|c| c as i64).collect()
}

/// Hash encode character ordinals for WtP models
///
/// This implements the same hashing scheme as CANINE, using multiple hash
/// functions with prime multipliers.
///
/// # Arguments
/// * `ordinals` - Character ordinals (Unicode code points)
/// * `num_hashes` - Number of hash functions to use
/// * `num_buckets` - Number of hash buckets
///
/// # Returns
/// A 2D array of shape (len, num_hashes) containing hash IDs
pub fn hash_encode(ordinals: &[i64], num_hashes: usize, num_buckets: i64) -> Vec<Vec<i64>> {
    assert!(
        num_hashes <= PRIMES.len(),
        "num_hashes must be <= {}",
        PRIMES.len()
    );

    ordinals
        .iter()
        .map(|&ord| {
            (0..num_hashes)
                .map(|i| {
                    let shard_id = (ord + 1) * PRIMES[i];
                    shard_id.rem_euclid(num_buckets)
                })
                .collect()
        })
        .collect()
}

/// Convert split indices to sentences
///
/// Given text and indices where sentences end, reconstruct the sentences.
///
/// # Arguments
/// * `text` - The original text
/// * `indices` - Character indices where sentences end
/// * `strip_whitespace` - Whether to strip whitespace from sentences
///
/// # Returns
/// Vector of sentence strings
pub fn indices_to_sentences(text: &str, indices: &[usize], strip_whitespace: bool) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut offset = 0;

    for &idx in indices {
        let mut end_idx = idx + 1;

        // Skip trailing whitespace for next sentence start
        while end_idx < chars.len() && chars[end_idx].is_whitespace() {
            end_idx += 1;
        }

        let sentence: String = chars[offset..end_idx].iter().collect();
        let sentence = if strip_whitespace {
            sentence.trim().to_string()
        } else {
            sentence
        };

        if !sentence.is_empty() {
            sentences.push(sentence);
        }

        offset = end_idx;
    }

    // Handle the last sentence (after the last split point)
    if offset < chars.len() {
        let last_sentence: String = chars[offset..].iter().collect();
        let last_sentence = if strip_whitespace {
            last_sentence.trim().to_string()
        } else {
            last_sentence
        };

        if !last_sentence.is_empty() {
            sentences.push(last_sentence);
        }
    }

    // Handle case when indices is empty - return the whole text
    if indices.is_empty() && !text.is_empty() {
        let sentence = if strip_whitespace {
            text.trim().to_string()
        } else {
            text.to_string()
        };
        if !sentence.is_empty() {
            return vec![sentence];
        }
    }

    sentences
}

/// Map token-level probabilities to character-level probabilities
///
/// For subword models, we need to map predictions from tokens back to characters.
/// Each token's probability is assigned to the last character of that token.
///
/// # Arguments
/// * `text_len` - Length of the original text in characters
/// * `token_logits` - Logits for each token (shape: num_tokens x num_labels)
/// * `offsets` - Character offset mapping for each token (start, end)
/// * `num_labels` - Number of output labels
///
/// # Returns
/// Character-level logits (shape: text_len x num_labels)
pub fn token_to_char_probs(
    text_len: usize,
    token_logits: &[Vec<f32>],
    offsets: &[(usize, usize)],
    num_labels: usize,
) -> Vec<Vec<f32>> {
    // Initialize with -inf (very low probability)
    let mut char_probs = vec![vec![f32::NEG_INFINITY; num_labels]; text_len];

    for (token_idx, &(start, end)) in offsets.iter().enumerate() {
        if start < end && end > 0 && end <= text_len && token_idx < token_logits.len() {
            // Assign token's probability to the last character of the token
            let char_idx = end - 1;
            char_probs[char_idx] = token_logits[token_idx].clone();
        }
    }

    char_probs
}

/// Remove spaces from text and track their positions
///
/// This is used for the `remove_whitespace_before_inference` option.
///
/// # Returns
/// Tuple of (text without spaces, original positions of spaces)
pub fn remove_spaces(text: &str) -> (String, Vec<usize>) {
    let mut result = String::new();
    let mut space_positions = Vec::new();

    for c in text.chars() {
        if c == ' ' {
            space_positions.push(result.len() + space_positions.len());
        } else {
            result.push(c);
        }
    }

    (result, space_positions)
}

/// Reinsert space probabilities into the probability array
pub fn reinsert_space_probs(probs: &[f32], space_positions: &[usize]) -> Vec<f32> {
    let mut result = probs.to_vec();

    for &pos in space_positions {
        if pos <= result.len() {
            result.insert(pos, 0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_encode_text() {
        let encoded = encode_text("abc");
        assert_eq!(encoded, vec![97, 98, 99]);
    }

    #[test]
    fn test_hash_encode() {
        let ordinals = vec![97, 98, 99]; // "abc"
        let hashed = hash_encode(&ordinals, 8, 8192);
        assert_eq!(hashed.len(), 3);
        assert_eq!(hashed[0].len(), 8);
        // All hash values should be in range [0, num_buckets)
        for row in &hashed {
            for &h in row {
                assert!(h >= 0 && h < 8192);
            }
        }
    }

    #[test]
    fn test_indices_to_sentences() {
        let text = "Hello world. This is a test.";
        let indices = vec![11]; // After "Hello world."
        let sentences = indices_to_sentences(text, &indices, false);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello world. ");
        assert_eq!(sentences[1], "This is a test.");
    }

    #[test]
    fn test_indices_to_sentences_strip() {
        let text = "Hello world.  This is a test.";
        let indices = vec![11]; // After "Hello world."
        let sentences = indices_to_sentences(text, &indices, true);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test.");
    }

    #[test]
    fn test_remove_spaces() {
        let (text, positions) = remove_spaces("hello world test");
        assert_eq!(text, "helloworldtest");
        assert_eq!(positions, vec![5, 11]);
    }

    /// The character indices that received a token's logits, with the value.
    fn written(char_probs: &[Vec<f32>]) -> Vec<(usize, f32)> {
        char_probs
            .iter()
            .enumerate()
            .filter(|(_, p)| p[0].is_finite())
            .map(|(i, p)| (i, p[0]))
            .collect()
    }

    /// One distinct logit per token, so every write can be traced to its token.
    fn logits(num_tokens: usize) -> Vec<Vec<f32>> {
        (0..num_tokens).map(|i| vec![(i + 1) as f32]).collect()
    }

    #[test]
    fn test_token_to_char_probs_ascii() {
        // Pure ASCII: byte offsets and character offsets coincide, so this case
        // behaves exactly as before.
        let text = "The cat sat.";
        assert_eq!(text.len(), text.chars().count());
        let offsets = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (6, 7),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ];
        let char_probs = token_to_char_probs(text.chars().count(), &logits(offsets.len()), &offsets, 1);
        assert_eq!(char_probs.len(), 12);
        assert_eq!(
            written(&char_probs),
            vec![
                (0, 1.0),
                (1, 2.0),
                (2, 3.0),
                (4, 4.0),
                (5, 5.0),
                (6, 6.0),
                (8, 7.0),
                (9, 8.0),
                (10, 9.0),
                (11, 10.0),
            ]
        );
    }

    #[test]
    fn test_token_to_char_probs_curly_apostrophe() {
        // U+2019 RIGHT SINGLE QUOTATION MARK is three bytes, so from there on the
        // byte offsets run two ahead of the character offsets.
        let text = "it\u{2019}d rain soon.";
        assert_eq!(text.len(), 17);
        assert_eq!(text.chars().count(), 15);
        let offsets = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
        ];
        let char_probs = token_to_char_probs(text.chars().count(), &logits(offsets.len()), &offsets, 1);
        assert_eq!(char_probs.len(), 15);
        assert_eq!(
            written(&char_probs),
            vec![
                (0, 1.0),
                (1, 2.0),
                (2, 3.0),
                (3, 4.0),
                (5, 5.0),
                (6, 6.0),
                (7, 7.0),
                (8, 8.0),
                (10, 9.0),
                (11, 10.0),
                (12, 11.0),
                (13, 12.0),
                (14, 13.0),
            ]
        );
        // The sentence-final period is character 14, and it is written.
        assert_eq!(char_probs[14][0], 13.0);
    }

    #[test]
    fn test_token_to_char_probs_em_dash() {
        // Same story for U+2014 EM DASH.
        let text = "a\u{2014}b test.";
        assert_eq!(text.len(), 11);
        assert_eq!(text.chars().count(), 9);
        let offsets = vec![(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)];
        let char_probs = token_to_char_probs(text.chars().count(), &logits(offsets.len()), &offsets, 1);
        assert_eq!(
            written(&char_probs),
            vec![
                (0, 1.0),
                (1, 2.0),
                (2, 3.0),
                (4, 4.0),
                (5, 5.0),
                (6, 6.0),
                (7, 7.0),
                (8, 8.0),
            ]
        );
    }

    #[test]
    fn test_token_to_char_probs_chinese() {
        // Every character is three bytes, so byte offsets are three times too
        // large; with character offsets all twelve tokens are written.
        let text = "\u{8fd9}\u{662f}\u{7b2c}\u{4e00}\u{53e5}\u{3002}\u{8fd9}\u{662f}\u{7b2c}\u{4e8c}\u{53e5}\u{3002}";
        assert_eq!(text.len(), 36);
        assert_eq!(text.chars().count(), 12);
        let offsets: Vec<(usize, usize)> = (0..12).map(|i| (i, i + 1)).collect();
        let char_probs = token_to_char_probs(text.chars().count(), &logits(offsets.len()), &offsets, 1);
        assert_eq!(char_probs.len(), 12);
        let expected: Vec<(usize, f32)> = (0..12).map(|i| (i, (i + 1) as f32)).collect();
        assert_eq!(written(&char_probs), expected);
        // Both sentence-final ideographic full stops are reachable.
        assert_eq!(char_probs[5][0], 6.0);
        assert_eq!(char_probs[11][0], 12.0);
    }

    #[test]
    fn test_token_to_char_probs_byte_offsets_overflow_char_len() {
        // The offsets `Tokenizer::encode` (as opposed to `encode_char_offsets`)
        // returns for the same Chinese text: `end` exceeds the character count
        // for two thirds of the tokens, so their logits are dropped by the
        // bounds check, and the remainder land on the wrong character. This is
        // the behaviour the tokenization call site must not reintroduce.
        let text = "\u{8fd9}\u{662f}\u{7b2c}\u{4e00}\u{53e5}\u{3002}\u{8fd9}\u{662f}\u{7b2c}\u{4e8c}\u{53e5}\u{3002}";
        let byte_offsets: Vec<(usize, usize)> = (0..12).map(|i| (i * 3, i * 3 + 3)).collect();
        let char_probs = token_to_char_probs(
            text.chars().count(),
            &logits(byte_offsets.len()),
            &byte_offsets,
            1,
        );
        assert_eq!(
            written(&char_probs),
            vec![(2, 1.0), (5, 2.0), (8, 3.0), (11, 4.0)]
        );
    }

    #[test]
    fn test_token_to_char_probs_edge_cases() {
        // Empty text yields no characters to write to.
        assert!(token_to_char_probs(0, &[], &[], 1).is_empty());
        // Special tokens carry (0, 0) offsets and must never be written.
        let char_probs = token_to_char_probs(3, &logits(3), &[(0, 0), (0, 1), (0, 0)], 1);
        assert_eq!(written(&char_probs), vec![(0, 2.0)]);
    }
}
