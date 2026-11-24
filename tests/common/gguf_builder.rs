#![allow(dead_code)]
//! Programmatic GGUF v3 binary builder for valid and intentionally malformed test fixtures.

use std::io::Write;

pub const GGUF_TYPE_UINT8: u32 = 0;
pub const GGUF_TYPE_INT8: u32 = 1;
pub const GGUF_TYPE_UINT16: u32 = 2;
pub const GGUF_TYPE_INT16: u32 = 3;
pub const GGUF_TYPE_UINT32: u32 = 4;
pub const GGUF_TYPE_INT32: u32 = 5;
pub const GGUF_TYPE_FLOAT32: u32 = 6;
pub const GGUF_TYPE_BOOL: u32 = 7;
pub const GGUF_TYPE_STRING: u32 = 8;
pub const GGUF_TYPE_ARRAY: u32 = 9;
pub const GGUF_TYPE_UINT64: u32 = 10;
pub const GGUF_TYPE_INT64: u32 = 11;
pub const GGUF_TYPE_FLOAT64: u32 = 12;

#[derive(Clone, Debug)]
pub enum KvVal {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrI64(Vec<i64>),
    ArrBool(Vec<bool>),
    ArrF32(Vec<f32>),
    /// Arbitrary raw encoded value bytes (with explicit type tag) for malformed fixtures.
    Raw { ty: u32, payload: Vec<u8> },
}

#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub name: String,
    pub ne: Vec<u64>,
    pub ty: u32,
    pub offset_override: Option<u64>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct GgufBuilder {
    pub magic: [u8; 4],
    pub version: u32,
    pub alignment: u64,
    pub kvs: Vec<(String, KvVal)>,
    pub tensors: Vec<TensorSpec>,
    pub truncate_to: Option<usize>,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self {
            magic: *b"GGUF",
            version: 3,
            alignment: 32,
            kvs: Vec::new(),
            tensors: Vec::new(),
            truncate_to: None,
        }
    }

    pub fn kv(mut self, key: impl Into<String>, val: KvVal) -> Self {
        self.kvs.push((key.into(), val));
        self
    }

    pub fn tensor(mut self, name: impl Into<String>, ne: Vec<u64>, ty: u32, data: Vec<u8>) -> Self {
        self.tensors.push(TensorSpec {
            name: name.into(),
            ne,
            ty,
            offset_override: None,
            data,
        });
        self
    }

    pub fn tensor_with_offset(
        mut self,
        name: impl Into<String>,
        ne: Vec<u64>,
        ty: u32,
        offset: u64,
        data: Vec<u8>,
    ) -> Self {
        self.tensors.push(TensorSpec {
            name: name.into(),
            ne,
            ty,
            offset_override: Some(offset),
            data,
        });
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());

        for (k, v) in &self.kvs {
            write_str(&mut out, k);
            write_kv(&mut out, v);
        }

        let align = self.alignment.max(1) as usize;
        let mut data_buf = Vec::new();
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            while data_buf.len() % align != 0 {
                data_buf.push(0);
            }
            let off = t.offset_override.unwrap_or(data_buf.len() as u64);
            offsets.push(off);
            data_buf.extend_from_slice(&t.data);
        }

        for (t, off) in self.tensors.iter().zip(offsets) {
            write_str(&mut out, &t.name);
            out.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
            for &dim in &t.ne {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&t.ty.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
        }

        let pad_align = self.alignment.max(1) as usize;
        while out.len() % pad_align != 0 {
            out.push(0);
        }
        out.extend_from_slice(&data_buf);

        if let Some(n) = self.truncate_to {
            out.truncate(n);
        }
        out
    }

    pub fn write_temp(&self) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("create temp gguf");
        f.write_all(&self.build()).expect("write temp gguf");
        f.flush().expect("flush temp gguf");
        f
    }
}

/// Build a minimal valid GGUF containing a complete 256-byte GPT-2 byte-level vocabulary,
/// optional merges, and standard Inkling control tokens (tok_type == 3).
pub fn minimal_tokenizer_gguf(extra_tokens: &[(&str, i32)], merges: &[&str]) -> Vec<u8> {
    let mut tokens: Vec<String> = Vec::new();
    let mut types: Vec<i32> = Vec::new();

    // GPT-2 byte <-> unicode table (all 256 single-byte tokens)
    let mut n = 0u32;
    for b in 0..256u32 {
        let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let cp = if keep {
            b
        } else {
            n += 1;
            255 + n
        };
        tokens.push(char::from_u32(cp).unwrap().to_string());
        types.push(1);
    }

    // Standard Inkling control tokens (tok_type = 3)
    for ctrl in [
        "<|end_message|>",
        "<|message_system|>",
        "<|message_user|>",
        "<|message_model|>",
        "<|content_text|>",
        "<|content_thinking|>",
        "<|content_model_end_sampling|>",
    ] {
        tokens.push(ctrl.to_string());
        types.push(3);
    }

    for &(tok, ty) in extra_tokens {
        tokens.push(tok.to_string());
        types.push(ty);
    }

    let eos_id = 256u32; // <|end_message|>
    GgufBuilder::new()
        .kv("tokenizer.ggml.tokens", KvVal::ArrStr(tokens))
        .kv("tokenizer.ggml.token_type", KvVal::ArrI32(types))
        .kv(
            "tokenizer.ggml.merges",
            KvVal::ArrStr(merges.iter().map(|s| s.to_string()).collect()),
        )
        .kv("tokenizer.ggml.eos_token_id", KvVal::U32(eos_id))
        .build()
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_kv(out: &mut Vec<u8>, v: &KvVal) {
    match v {
        KvVal::U8(x) => {
            out.extend_from_slice(&GGUF_TYPE_UINT8.to_le_bytes());
            out.push(*x);
        }
        KvVal::I8(x) => {
            out.extend_from_slice(&GGUF_TYPE_INT8.to_le_bytes());
            out.push(*x as u8);
        }
        KvVal::U16(x) => {
            out.extend_from_slice(&GGUF_TYPE_UINT16.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::I16(x) => {
            out.extend_from_slice(&GGUF_TYPE_INT16.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::U32(x) => {
            out.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::I32(x) => {
            out.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::F32(x) => {
            out.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::Bool(x) => {
            out.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
            out.push(*x as u8);
        }
        KvVal::Str(s) => {
            out.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            write_str(out, s);
        }
        KvVal::U64(x) => {
            out.extend_from_slice(&GGUF_TYPE_UINT64.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::I64(x) => {
            out.extend_from_slice(&GGUF_TYPE_INT64.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::F64(x) => {
            out.extend_from_slice(&GGUF_TYPE_FLOAT64.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        KvVal::ArrStr(xs) => {
            out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            out.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for s in xs {
                write_str(out, s);
            }
        }
        KvVal::ArrI32(xs) => {
            out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            out.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        KvVal::ArrI64(xs) => {
            out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            out.extend_from_slice(&GGUF_TYPE_INT64.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        KvVal::ArrBool(xs) => {
            out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            out.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.push(*x as u8);
            }
        }
        KvVal::ArrF32(xs) => {
            out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            out.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        KvVal::Raw { ty, payload } => {
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(payload);
        }
    }
}
