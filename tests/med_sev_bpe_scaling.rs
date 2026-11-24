mod common;

use common::gguf_builder::minimal_tokenizer_gguf;
use inkling::gguf::Gguf;
use inkling::tokenizer::Tokenizer;
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Naive O(L^2) reference BPE matching the exact original semantics:
/// 1. Whole-piece lookup in `tok_to_id` (excluding `tok_type == 3`).
/// 2. Repeatedly scan left-to-right for the strictly lowest-rank adjacent merge
///    (`rank < best.rank`, breaking ties by leftmost index) and splice it in place.
struct NaiveBpe {
    id_to_tok: Vec<String>,
    tok_to_id: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_to_id: [u32; 256],
}

impl NaiveBpe {
    fn from_gguf(g: &Gguf) -> Self {
        use inkling::gguf::Value;
        let id_to_tok = match &g.kv["tokenizer.ggml.tokens"] {
            Value::AS(v) => v.clone(),
            _ => panic!("no tokens"),
        };
        let tok_type = match &g.kv["tokenizer.ggml.token_type"] {
            Value::AI(v) => v.clone(),
            _ => panic!("no token types"),
        };
        let merges_s = match &g.kv["tokenizer.ggml.merges"] {
            Value::AS(v) => v,
            _ => panic!("no merges"),
        };
        let mut tok_to_id = HashMap::with_capacity(id_to_tok.len() * 2);
        for (i, t) in id_to_tok.iter().enumerate() {
            if tok_type.get(i) != Some(&3) {
                tok_to_id.insert(t.clone(), i as u32);
            }
        }
        let mut merges = HashMap::with_capacity(merges_s.len() * 2);
        for (rank, m) in merges_s.iter().enumerate() {
            let (a, b) = m.split_once(' ').expect("bad merge");
            if let (Some(&ia), Some(&ib), Some(&im)) = (
                tok_to_id.get(a),
                tok_to_id.get(b),
                tok_to_id.get(&format!("{a}{b}")),
            ) {
                merges.insert((ia, ib), (rank as u32, im));
            }
        }
        let mut byte_to_id = [0u32; 256];
        let mut n = 0u32;
        for b in 0..256u32 {
            let keep =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let cp = if keep {
                b
            } else {
                n += 1;
                255 + n
            };
            let ch = char::from_u32(cp).unwrap();
            byte_to_id[b as usize] = *tok_to_id.get(&ch.to_string()).expect("byte token missing");
        }
        Self {
            id_to_tok,
            tok_to_id,
            merges,
            byte_to_id,
        }
    }

    fn bpe_piece(&self, piece: &str) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }
        let mapped: String = piece
            .bytes()
            .map(|b| self.id_to_tok[self.byte_to_id[b as usize] as usize].as_str())
            .collect();
        if let Some(&id) = self.tok_to_id.get(&mapped) {
            return vec![id];
        }
        let mut sym: Vec<u32> = piece
            .bytes()
            .map(|b| self.byte_to_id[b as usize])
            .collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for j in 0..sym.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(sym[j], sym[j + 1])) {
                    if best.is_none_or(|b| rank < b.0) {
                        best = Some((rank, j, merged));
                    }
                }
            }
            match best {
                Some((_, j, m)) => {
                    sym[j] = m;
                    sym.remove(j + 1);
                }
                None => break,
            }
        }
        sym
    }
}

fn build_pair(extra_tokens: &[(&str, i32)], merges: &[&str]) -> (Tokenizer, NaiveBpe) {
    let raw = minimal_tokenizer_gguf(extra_tokens, merges);
    let g = Gguf::from_bytes(&raw).expect("parse synthetic gguf");
    let tk = Tokenizer::load(&g);
    let naive = NaiveBpe::from_gguf(&g);
    (tk, naive)
}

fn decode_all(tk: &Tokenizer, ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|&id| tk.decode_bytes(id)).collect()
}

/// Build a tokenizer with cascading repeated-'a' merges, alternating 'ab' merges,
/// and non-merging distractor pairs, used for both correctness and scaling tests.
fn build_scaling_tokenizer() -> (Tokenizer, NaiveBpe) {
    build_pair(
        &[
            ("aa", 1),
            ("aaaa", 1),
            ("aaaaaaaa", 1),
            ("ab", 1),
            ("abab", 1),
            ("abababab", 1),
            ("ba", 1),
            ("aba", 1),
            ("ababa", 1),
            ("cd", 1),
            ("xy", 1),
            ("zz", 1),
            // Special token that must NEVER be produced by bpe()
            ("aaaaaaaaa", 3),
        ],
        &[
            "a a",
            "aa aa",
            "aaaa aaaa",
            "a b",
            "ab ab",
            "abab abab",
            "b a",
            "a ba",
            "aba ba",
            "c d",
            "x y",
            "z z",
            "aaaaaaaa a",
        ],
    )
}

#[test]
fn bpe_exact_equivalence_tricky_tie_breaking_and_cascades() {
    // Competing merges with non-monotonic ranks that trigger stale heap entries
    // and leftmost tie-breaking.
    let (tk, naive) = build_pair(
        &[
            ("aa", 1),
            ("aaa", 1),
            ("aaaa", 1),
            ("ab", 1),
            ("ba", 1),
            ("bc", 1),
            ("xa", 1),
            ("abc", 1),
            ("aba", 1),
            ("ababa", 1),
            ("xabc", 1),
            ("Ġa", 1),
            ("Ġab", 1),
            // UTF-8 multibyte bytes for 'é' (0xC3 0xA9 -> Ã ©)
            ("Ã©", 1),
            ("Ã©Ã©", 1),
            // Control token that must not be emitted when parse_special=false
            ("abac", 3),
        ],
        &[
            "b c",
            "x a",
            "xa bc",
            "a bc",
            "a b",
            "a a",
            "aa a",
            "aa aa",
            "b a",
            "a ba",
            "aba ba",
            "Ġ a",
            "Ġa b",
            "Ã ©",
            "Ã© Ã©",
            "aba c",
        ],
    );

    let single_pieces = [
        "",
        "a",
        "aa",
        "aaa",
        "aaaa",
        "aaaaa",
        "aaaaaa",
        "aaaaaaa",
        "aaaaaaaa",
        "xabc",
        "xabcd",
        "abcbcd",
        "abababa",
        "ababababa",
        "abac",
        "é",
        "éé",
        "ééé",
        "éééé",
        "ééééé",
    ];

    for piece in single_pieces {
        let got = tk.encode(piece, false);
        let expected = naive.bpe_piece(piece);
        assert_eq!(got, expected, "mismatch on single piece {piece:?}");
        assert!(
            got.iter().all(|&id| !tk.is_special(id)),
            "special token leaked for {piece:?}"
        );
        assert_eq!(decode_all(&tk, &got), piece.as_bytes());
    }

    // Multi-piece space-separated input (each " word" is one pre-tokenizer piece)
    let multi = "xabc abababa aaaaaaa éééé abac";
    let expected_multi: Vec<u32> = ["xabc", " abababa", " aaaaaaa", " éééé", " abac"]
        .into_iter()
        .flat_map(|p| naive.bpe_piece(p))
        .collect();
    assert_eq!(tk.encode(multi, false), expected_multi);
}

#[test]
fn bpe_scaling_50k_chars_completes_in_o_n_log_n() {
    let (tk, _naive) = build_scaling_tokenizer();

    // Tight time budget that O(L log L) beats easily and O(L^2) violates by >40x.
    #[cfg(debug_assertions)]
    let budget = Duration::from_millis(250);
    #[cfg(not(debug_assertions))]
    let budget = Duration::from_millis(50);

    // Watchdog prevents an O(L^2) regression from hanging the test runner for minutes.
    let watchdog = Duration::from_secs(2);

    let run_timed = |input: String| -> (Vec<u32>, Duration) {
        let raw = minimal_tokenizer_gguf(
            &[
                ("aa", 1),
                ("aaaa", 1),
                ("aaaaaaaa", 1),
                ("ab", 1),
                ("abab", 1),
                ("abababab", 1),
                ("cd", 1),
                ("xy", 1),
            ],
            &[
                "a a",
                "aa aa",
                "aaaa aaaa",
                "a b",
                "ab ab",
                "abab abab",
                "c d",
                "x y",
            ],
        );
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let g = Gguf::from_bytes(&raw).expect("parse synthetic gguf");
            let tk = Tokenizer::load(&g);
            let t0 = Instant::now();
            let ids = tk.encode(&input, false);
            let elapsed = t0.elapsed();
            let _ = tx.send((ids, elapsed));
        });
        match rx.recv_timeout(watchdog) {
            Ok((ids, elapsed)) => {
                handle.join().expect("encoder thread panicked");
                (ids, elapsed)
            }
            Err(_) => {
                panic!(
                    "Tokenizer::encode exceeded watchdog ({watchdog:?}) on 50,000-char input \
                     — quadratic O(L^2) BPE regression!"
                );
            }
        }
    };

    // Case 1: 50,000 repeated 'a's -> cascades a->aa->aaaa->aaaaaaaa (6,250 tokens)
    let input_a = "a".repeat(50_000);
    let (ids_a, elapsed_a) = run_timed(input_a.clone());
    assert!(
        elapsed_a < budget,
        "encode('a' * 50_000) took {elapsed_a:?}, expected < {budget:?}"
    );
    assert_eq!(ids_a.len(), 6_250, "50,000 'a's should merge into 6,250 'aaaaaaaa' tokens");
    assert_eq!(decode_all(&tk, &ids_a), input_a.as_bytes());

    // Case 2: 50,000 chars of alternating "ab" -> cascades ab->abab->abababab (6,250 tokens)
    let input_ab = "ab".repeat(25_000);
    let (ids_ab, elapsed_ab) = run_timed(input_ab.clone());
    assert!(
        elapsed_ab < budget,
        "encode('ab' * 25_000) took {elapsed_ab:?}, expected < {budget:?}"
    );
    assert_eq!(ids_ab.len(), 6_250, "25,000 'ab's should merge into 6,250 'abababab' tokens");
    assert_eq!(decode_all(&tk, &ids_ab), input_ab.as_bytes());

    // Case 3: Odd length 50,001 'a's -> 6,250 'aaaaaaaa' + 1 'a' (tests leftmost tie-breaking tail)
    let input_odd = "a".repeat(50_001);
    let (ids_odd, elapsed_odd) = run_timed(input_odd.clone());
    assert!(
        elapsed_odd < budget,
        "encode('a' * 50_001) took {elapsed_odd:?}, expected < {budget:?}"
    );
    assert_eq!(ids_odd.len(), 6_251);
    assert_eq!(ids_odd[..6_250], ids_a[..]);
    assert_eq!(tk.decode_bytes(ids_odd[6_250]), b"a");
    assert_eq!(decode_all(&tk, &ids_odd), input_odd.as_bytes());
}

/// Build a random valid BPE merge table over alphabet {'a', 'b', 'c', 'd'} from a
/// sequence of merge choices. Each merge combines two existing vocab symbols into a new
/// symbol, exercising arbitrary rank orderings, ties, and cascades.
fn build_random_merge_table(steps: &[(usize, usize)]) -> (Vec<String>, Vec<String>) {
    let mut pool: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    let mut extra_tokens: Vec<String> = Vec::new();
    let mut merges: Vec<String> = Vec::new();
    let mut seen_pairs: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();

    for &(ia, ib) in steps {
        let a = pool[ia % pool.len()].clone();
        let b = pool[ib % pool.len()].clone();
        if a.len() + b.len() > 16 || !seen_pairs.insert((a.clone(), b.clone())) {
            continue;
        }
        let merged = format!("{a}{b}");
        if !pool.contains(&merged) {
            pool.push(merged.clone());
            extra_tokens.push(merged);
        }
        merges.push(format!("{a} {b}"));
    }
    (extra_tokens, merges)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// Bit-exact equivalence between Tokenizer::encode(s, false) and naive O(L^2) BPE
    /// on arbitrary single-piece strings with repeated chars, Unicode, and competing merges.
    #[test]
    fn prop_bpe_matches_naive_on_fixed_rich_merges(
        pieces in prop::collection::vec("[a-d]{1,32}", 1..6),
    ) {
        let (tk, naive) = build_pair(
            &[
                ("aa", 1), ("bb", 1), ("ab", 1), ("ba", 1), ("bc", 1), ("cb", 1),
                ("cd", 1), ("da", 1), ("aaa", 1), ("aba", 1), ("abc", 1), ("bab", 1),
                ("aaaa", 1), ("abab", 1), ("abcd", 1), ("Ġa", 1), ("Ġab", 1), ("Ġaa", 1),
                ("bcda", 3), // special token must never be emitted
            ],
            &[
                "b c", "a b", "a a", "b a", "c b", "c d", "d a", "b b",
                "a ba", "ab c", "aa a", "b ab", "aa aa", "ab ab", "abc d",
                "Ġ a", "Ġa b", "Ġa a", "bc da",
            ],
        );

        let full = pieces.join(" ");
        let got = tk.encode(&full, false);

        // Each lowercase ASCII piece (with leading space for pieces[1..]) is one pre-tokenizer piece.
        let mut expected = Vec::new();
        for (idx, p) in pieces.iter().enumerate() {
            if idx == 0 {
                expected.extend(naive.bpe_piece(p));
            } else {
                expected.extend(naive.bpe_piece(&format!(" {p}")));
            }
        }

        prop_assert_eq!(&got, &expected, "mismatch on {:?}", full);
        prop_assert!(got.iter().all(|&id| !tk.is_special(id)));
        prop_assert_eq!(decode_all(&tk, &got), full.as_bytes());
    }

    /// Bit-exact equivalence over arbitrary randomly generated merge tables & strings,
    /// stressing stale heap entries, tied ranks, and deep cascades.
    #[test]
    fn prop_bpe_matches_naive_on_arbitrary_merge_tables(
        steps in prop::collection::vec((0usize..32, 0usize..32), 1..24),
        input in "[a-d]{0,64}",
    ) {
        let (extra_owned, merges_owned) = build_random_merge_table(&steps);
        let extra_refs: Vec<(&str, i32)> = extra_owned.iter().map(|s| (s.as_str(), 1)).collect();
        let merge_refs: Vec<&str> = merges_owned.iter().map(|s| s.as_str()).collect();
        let (tk, naive) = build_pair(&extra_refs, &merge_refs);

        let got = tk.encode(&input, false);
        let expected = naive.bpe_piece(&input);
        prop_assert_eq!(&got, &expected, "mismatch for input={:?}, merges={:?}", input, merges_owned);
        prop_assert_eq!(decode_all(&tk, &got), input.as_bytes());
    }

    /// Bit-exact equivalence on multi-byte Unicode lowercase letter pieces (single piece in GPT-4o split).
    #[test]
    fn prop_bpe_matches_naive_on_unicode_pieces(
        chars in prop::collection::vec(prop_oneof![Just('é'), Just('ü'), Just('α'), Just('β'), Just('a')], 1..40),
    ) {
        let (tk, naive) = build_pair(
            &[
                ("Ã©", 1),
                ("Ã¼", 1),
                ("Î±", 1),
                ("Î²", 1),
                ("Ã©Ã©", 1),
                ("Î±Î²", 1),
                ("Ã©a", 1),
                ("aÃ©", 1),
                ("Ã©Ã¼", 1),
            ],
            &[
                "Ã ©",
                "Ã ¼",
                "Î ±",
                "Î ²",
                "Ã© Ã©",
                "Î± Î²",
                "Ã© a",
                "a Ã©",
                "Ã© Ã¼",
            ],
        );

        let s: String = chars.into_iter().collect();
        let got = tk.encode(&s, false);
        let expected = naive.bpe_piece(&s);
        prop_assert_eq!(&got, &expected, "unicode mismatch on {:?}", s);
        prop_assert_eq!(decode_all(&tk, &got), s.as_bytes());
    }
}
