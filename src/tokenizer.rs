//! Byte-level BPE tokenizer for Inkling 2 (GPT-4o / o200k style pre-tokenizer), built from GGUF vocab + merges.

use crate::gguf::{Gguf, Value};
use crate::unicode_tables::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub struct Tokenizer {
    id_to_tok: Vec<String>,
    tok_type: Vec<i64>,
    tok_to_id: HashMap<String, u32>,
    max_tok_bytes: usize,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    specials: Vec<(String, u32)>,
    special_first_byte: [bool; 256],
    byte_to_id: [u32; 256],
    uni_to_byte: HashMap<char, u8>,
    pub eos: u32,
}

fn uflags(c: u32) -> u8 {
    let i = RUN_START.partition_point(|&s| s <= c);
    RUN_FLAGS[i - 1]
}
#[inline] fn is_l(f: u8) -> bool { f & (U_LU | U_LL | U_LT | U_LM | U_LO) != 0 }
#[inline] fn is_n(f: u8) -> bool { f & U_N != 0 }
#[inline] fn is_ws(f: u8) -> bool { f & U_WS != 0 }
#[inline] fn is_ucls(f: u8) -> bool { f & (U_LU | U_LT | U_LM | U_LO | U_M) != 0 }
#[inline] fn is_lcls(f: u8) -> bool { f & (U_LL | U_LM | U_LO | U_M) != 0 }

impl Tokenizer {
    pub fn load(g: &Gguf) -> Tokenizer {
        let id_to_tok = match &g.kv["tokenizer.ggml.tokens"] { Value::AS(v) => v.clone(), _ => panic!("no tokens") };
        let tok_type = match &g.kv["tokenizer.ggml.token_type"] { Value::AI(v) => v.clone(), _ => panic!("no token types") };
        let merges_s = match &g.kv["tokenizer.ggml.merges"] { Value::AS(v) => v, _ => panic!("no merges") };
        let mut tok_to_id = HashMap::with_capacity(id_to_tok.len() * 2);
        let mut max_tok_bytes = 0usize;
        for (i, t) in id_to_tok.iter().enumerate() {
            // Exclude special/control tokens (tok_type == 3) from BPE lookup and merges so
            // `encode(..., false)` can never emit control token IDs.
            if tok_type.get(i) != Some(&3) {
                max_tok_bytes = max_tok_bytes.max(t.len());
                tok_to_id.insert(t.clone(), i as u32);
            }
        }
        let mut merges = HashMap::with_capacity(merges_s.len() * 2);
        for (rank, m) in merges_s.iter().enumerate() {
            let (a, b) = m.split_once(' ').expect("bad merge");
            if let (Some(&ia), Some(&ib), Some(&im)) = (tok_to_id.get(a), tok_to_id.get(b), tok_to_id.get(&format!("{a}{b}"))) {
                merges.insert((ia, ib), (rank as u32, im));
            }
        }
        let mut specials: Vec<(String, u32)> = id_to_tok.iter().enumerate()
            .filter(|(i, t)| tok_type.get(*i) == Some(&3) && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32)).collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let mut special_first_byte = [false; 256];
        for (s, _) in &specials {
            special_first_byte[s.as_bytes()[0] as usize] = true;
        }
        // GPT-2 byte <-> unicode mapping
        let mut byte_to_id = [0u32; 256];
        let mut uni_to_byte = HashMap::new();
        let mut n = 0u32;
        for b in 0..256u32 {
            let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let cp = if keep { b } else { n += 1; 255 + n };
            let ch = char::from_u32(cp).unwrap();
            byte_to_id[b as usize] = *tok_to_id.get(&ch.to_string()).expect("byte token missing");
            uni_to_byte.insert(ch, b as u8);
        }
        let eos = g.u("tokenizer.ggml.eos_token_id") as u32;
        Tokenizer { id_to_tok, tok_type, tok_to_id, max_tok_bytes, merges, specials, special_first_byte, byte_to_id, uni_to_byte, eos }
    }

    pub fn is_special(&self, id: u32) -> bool { self.tok_type.get(id as usize) == Some(&3) }

    /// Look up the token ID of a special/control token (`tok_type == 3`) by exact string.
    pub fn special_id(&self, tok: &str) -> Option<u32> {
        self.specials.iter().find(|(s, _)| s == tok).map(|&(_, id)| id)
    }

    /// Raw bytes of one token (special tokens rendered literally; out-of-range IDs return empty).
    pub fn decode_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.id_to_tok.get(id as usize) else {
            return Vec::new();
        };
        if self.is_special(id) { return t.as_bytes().to_vec(); }
        let mut out = Vec::with_capacity(t.len());
        for ch in t.chars() {
            match self.uni_to_byte.get(&ch) {
                Some(b) => out.push(*b),
                None => { let mut buf = [0u8; 4]; out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()); }
            }
        }
        out
    }

    /// GPT-4o pre-tokenizer regex, hand-matched:
    ///  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    /// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    /// |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
    fn split(&self, s: &[char]) -> Vec<(usize, usize)> {
        let n = s.len();
        let f: Vec<u8> = s.iter().map(|&c| uflags(c as u32)).collect();
        let lower = |c: char| c.to_ascii_lowercase();
        let contraction = |e: usize| -> usize {
            if e < n && s[e] == '\'' && e + 1 < n {
                let a = lower(s[e + 1]);
                let b = if e + 2 < n { lower(s[e + 2]) } else { '\0' };
                if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') { return e + 3; }
                if matches!(a, 's' | 't' | 'm' | 'd') { return e + 2; }
            }
            e
        };
        let is_p = |i: usize| i < n && s[i] != '\r' && s[i] != '\n' && !is_l(f[i]) && !is_n(f[i]);
        let body1 = |k: usize| -> Option<usize> {
            let mut u = k;
            while u < n && is_ucls(f[u]) { u += 1; }
            let mut t = u + 1;
            while t > k {
                t -= 1;
                if t < n && is_lcls(f[t]) {
                    let mut e = t;
                    while e < n && is_lcls(f[e]) { e += 1; }
                    return Some(e);
                }
            }
            None
        };
        let body2 = |k: usize| -> Option<usize> {
            let mut u = k;
            while u < n && is_ucls(f[u]) { u += 1; }
            if u == k { return None; }
            let mut e = u;
            while e < n && is_lcls(f[e]) { e += 1; }
            Some(e)
        };
        let mut spans = Vec::new();
        let mut i = 0;
        while i < n {
            let mut end: Option<usize> = None;
            for alt in 0..2 {
                let body = |k: usize| if alt == 0 { body1(k) } else { body2(k) };
                if is_p(i) {
                    if let Some(e) = body(i + 1) { end = Some(contraction(e)); break; }
                }
                if let Some(e) = body(i) { end = Some(contraction(e)); break; }
            }
            if end.is_none() && is_n(f[i]) {
                let mut e = i;
                while e < n && e < i + 3 && is_n(f[e]) { e += 1; }
                end = Some(e);
            }
            if end.is_none() {
                let cls = |j: usize| j < n && !is_ws(f[j]) && !is_l(f[j]) && !is_n(f[j]);
                let mut k = i;
                if s[k] == ' ' && cls(k + 1) { k += 1; }
                if cls(k) {
                    let mut e = k;
                    while cls(e) { e += 1; }
                    while e < n && matches!(s[e], '\r' | '\n' | '/') { e += 1; }
                    end = Some(e);
                }
            }
            if end.is_none() && is_ws(f[i]) {
                let mut w = i;
                while w < n && is_ws(f[w]) { w += 1; }
                let last = (i..w).rev().find(|&j| s[j] == '\r' || s[j] == '\n');
                end = Some(if let Some(l) = last { l + 1 } else if w == n { w } else if w - i >= 2 { w - 1 } else { w });
            }
            let e = match end { Some(e) if e > i => e, _ => i + 1 };
            spans.push((i, e));
            i = e;
        }
        spans
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let bytes = piece.as_bytes();
        if bytes.is_empty() {
            return;
        }
        if bytes.len() == 1 {
            out.push(self.byte_to_id[bytes[0] as usize]);
            return;
        }
        // fast path: whole piece is a token (every byte maps to >= 1 UTF-8 byte, so
        // pieces longer than `max_tok_bytes` can never match a vocabulary entry)
        if bytes.len() <= self.max_tok_bytes {
            let mut mapped = String::with_capacity(bytes.len() * 2);
            for &b in bytes {
                mapped.push_str(&self.id_to_tok[self.byte_to_id[b as usize] as usize]);
            }
            if let Some(&id) = self.tok_to_id.get(&mapped) {
                out.push(id);
                return;
            }
        }
        let n = bytes.len();
        // Zero-allocation stack path for normal-length words (<= 32 bytes); avoids 5 heap
        // allocations per short piece while keeping O(L log L) scaling for longer pieces.
        if n <= 32 {
            let mut buf = [0u32; 32];
            for (i, &b) in bytes.iter().enumerate() {
                buf[i] = self.byte_to_id[b as usize];
            }
            let mut len = n;
            while len > 1 {
                let mut best: Option<(u32, usize, u32)> = None;
                for j in 0..len - 1 {
                    if let Some(&(rank, merged)) = self.merges.get(&(buf[j], buf[j + 1])) {
                        if best.is_none_or(|b| rank < b.0) {
                            best = Some((rank, j, merged));
                        }
                    }
                }
                match best {
                    Some((_, j, m)) => {
                        buf[j] = m;
                        buf.copy_within(j + 2..len, j + 1);
                        len -= 1;
                    }
                    None => break,
                }
            }
            out.extend_from_slice(&buf[..len]);
            return;
        }
        let mut sym: Vec<u32> = bytes.iter().map(|&b| self.byte_to_id[b as usize]).collect();
        const NONE: usize = usize::MAX;
        let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { NONE } else { i - 1 }).collect();
        let mut next: Vec<usize> = (0..n).map(|i| if i + 1 == n { NONE } else { i + 1 }).collect();
        let mut alive = vec![true; n];
        // Min-heap by (rank, left_idx): alive node indices remain strictly increasing along `next`,
        // so ties break strictly leftmost, matching the exact original left-to-right scan.
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(&(rank, _)) = self.merges.get(&(sym[i], sym[i + 1])) {
                heap.push(Reverse((rank, i)));
            }
        }
        while let Some(Reverse((rank, left))) = heap.pop() {
            if !alive[left] {
                continue;
            }
            let right = next[left];
            if right == NONE || !alive[right] {
                continue;
            }
            let Some(&(cur_rank, merged)) = self.merges.get(&(sym[left], sym[right])) else {
                continue;
            };
            if cur_rank != rank {
                continue;
            }
            sym[left] = merged;
            alive[right] = false;
            let rr = next[right];
            next[left] = rr;
            if rr != NONE {
                prev[rr] = left;
            }
            let ll = prev[left];
            if ll != NONE {
                if let Some(&(r, _)) = self.merges.get(&(sym[ll], sym[left])) {
                    heap.push(Reverse((r, ll)));
                }
            }
            if rr != NONE {
                if let Some(&(r, _)) = self.merges.get(&(sym[left], sym[rr])) {
                    heap.push(Reverse((r, left)));
                }
            }
        }
        let mut idx = 0;
        while idx != NONE {
            out.push(sym[idx]);
            idx = next[idx];
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() { return; }
        let cps: Vec<char> = text.chars().collect();
        for (a, b) in self.split(&cps) {
            let piece: String = cps[a..b].iter().collect();
            self.bpe(&piece, out);
        }
    }

    /// Encode text; when `parse_special`, literal control tokens (`tok_type == 3`) become their ids.
    /// When `!parse_special`, no token ID with `tok_type == 3` is ever emitted.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if !parse_special {
            self.encode_plain(text, &mut out);
            return out;
        }
        let mut plain_start = 0;
        let bytes = text.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            if self.special_first_byte[bytes[pos] as usize] && text.is_char_boundary(pos) {
                if let Some((s, id)) = self.specials.iter().find(|(s, _)| text[pos..].starts_with(s.as_str())) {
                    self.encode_plain(&text[plain_start..pos], &mut out);
                    out.push(*id);
                    pos += s.len();
                    plain_start = pos;
                    continue;
                }
            }
            pos += 1;
        }
        self.encode_plain(&text[plain_start..], &mut out);
        out
    }
}
