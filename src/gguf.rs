//! Zero-copy GGUF reader: mmaps the file and parses metadata + the tensor table.

use crate::engine::{
    BLOCK, E, EXT, FF, HD, LOOP_END, LOOP_START, NEXP, NH, NKV, NREL, N_PASS, N_PHYS, TOPK,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::os::unix::io::AsRawFd;

pub const T_F32: u32 = 0;
pub const T_F16: u32 = 1;
pub const T_Q4_0: u32 = 2;
pub const T_Q5_0: u32 = 6;
pub const T_Q8_0: u32 = 8;
pub const T_Q4_K: u32 = 12;
pub const T_Q5_K: u32 = 13;
pub const T_Q6_K: u32 = 14;
pub const T_BF16: u32 = 30;

fn invalid_data(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// Checked lookup of `(bytes per block, elements per block)` for a ggml tensor type.
pub fn try_type_block(t: u32) -> Result<(usize, usize)> {
    match t {
        T_F32 => Ok((4, 1)),
        T_F16 | T_BF16 => Ok((2, 1)),
        T_Q4_0 => Ok((18, 32)),
        T_Q5_0 => Ok((22, 32)),
        T_Q8_0 => Ok((34, 32)),
        T_Q4_K => Ok((144, 256)),
        T_Q5_K => Ok((176, 256)),
        T_Q6_K => Ok((210, 256)),
        _ => Err(invalid_data(format!("unsupported ggml type {t}"))),
    }
}

/// (bytes per block, elements per block)
pub fn type_block(t: u32) -> (usize, usize) {
    try_type_block(t).unwrap_or_else(|e| panic!("{e}"))
}

pub fn type_name(t: u32) -> &'static str {
    match t {
        T_F32 => "F32",
        T_F16 => "F16",
        T_BF16 => "BF16",
        T_Q4_0 => "Q4_0",
        T_Q5_0 => "Q5_0",
        T_Q8_0 => "Q8_0",
        T_Q4_K => "Q4_K",
        T_Q5_K => "Q5_K",
        T_Q6_K => "Q6_K",
        _ => "?",
    }
}

/// Non-finite weight or quantization scale offender found during a GGUF finiteness audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightOffender {
    pub tensor: String,
    pub ty: u32,
    pub non_finite: u64,
}

/// Summary of a load-time finiteness scan across all tensors in a [`Gguf`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightAuditReport {
    pub tensors_checked: usize,
    pub blocks_checked: u64,
    pub non_finite_count: u64,
    pub offenders: Vec<WeightOffender>,
}

impl WeightAuditReport {
    pub fn is_clean(&self) -> bool {
        self.non_finite_count == 0
    }
}

#[inline]
fn is_f32_non_finite(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000
}

#[inline]
fn is_f16_non_finite(bits: u16) -> bool {
    (bits & 0x7C00) == 0x7C00
}

#[inline]
fn is_bf16_non_finite(bits: u16) -> bool {
    (bits & 0x7F80) == 0x7F80
}

/// Scan a single tensor's raw bytes for non-finite values or quantization scale headers.
/// Returns `(blocks_checked, non_finite_count)`.
///
/// Quantized block layouts match `src/kernels.metal` / GGML:
/// - `Q4_0` (18B), `Q5_0` (22B), `Q8_0` (34B): fp16 `d` at `0..2`
/// - `Q4_K` (144B), `Q5_K` (176B): fp16 `d` at `0..2`, fp16 `dmin` at `2..4`
/// - `Q6_K` (210B): fp16 `d` at `208..210`
fn audit_tensor_bytes(ty: u32, data: &[u8]) -> (u64, u64) {
    let mut bad = 0u64;
    match ty {
        T_F32 => {
            let chunks = data.as_chunks::<4>().0;
            for &c in chunks {
                bad += is_f32_non_finite(u32::from_le_bytes(c)) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_F16 => {
            let chunks = data.as_chunks::<2>().0;
            for &c in chunks {
                bad += is_f16_non_finite(u16::from_le_bytes(c)) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_BF16 => {
            let chunks = data.as_chunks::<2>().0;
            for &c in chunks {
                bad += is_bf16_non_finite(u16::from_le_bytes(c)) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_Q4_0 => {
            let chunks = data.as_chunks::<18>().0;
            for b in chunks {
                bad += is_f16_non_finite(u16::from_le_bytes([b[0], b[1]])) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_Q5_0 => {
            let chunks = data.as_chunks::<22>().0;
            for b in chunks {
                bad += is_f16_non_finite(u16::from_le_bytes([b[0], b[1]])) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_Q8_0 => {
            let chunks = data.as_chunks::<34>().0;
            for b in chunks {
                bad += is_f16_non_finite(u16::from_le_bytes([b[0], b[1]])) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_Q4_K => {
            let chunks = data.as_chunks::<144>().0;
            for b in chunks {
                let d = u16::from_le_bytes([b[0], b[1]]);
                let dmin = u16::from_le_bytes([b[2], b[3]]);
                bad += is_f16_non_finite(d) as u64 + is_f16_non_finite(dmin) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_Q5_K => {
            let chunks = data.as_chunks::<176>().0;
            for b in chunks {
                let d = u16::from_le_bytes([b[0], b[1]]);
                let dmin = u16::from_le_bytes([b[2], b[3]]);
                bad += is_f16_non_finite(d) as u64 + is_f16_non_finite(dmin) as u64;
            }
            (chunks.len() as u64, bad)
        }
        T_Q6_K => {
            let chunks = data.as_chunks::<210>().0;
            for b in chunks {
                bad += is_f16_non_finite(u16::from_le_bytes([b[208], b[209]])) as u64;
            }
            (chunks.len() as u64, bad)
        }
        _ => (0, 0),
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    AI(Vec<i64>),
    #[allow(dead_code)]
    AF(Vec<f64>),
    AS(Vec<String>),
    AB(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub ne: Vec<u64>,
    pub ty: u32,
    /// byte offset from the start of the mapped file
    pub file_off: u64,
    pub nbytes: u64,
}

impl Tensor {
    pub fn try_row_bytes(&self) -> Result<u64> {
        let (bs, be) = try_type_block(self.ty)?;
        let &d0 = self
            .ne
            .first()
            .ok_or_else(|| invalid_data(format!("tensor {} has zero rank", self.name)))?;
        let be_u64 = be as u64;
        if !d0.is_multiple_of(be_u64) {
            return Err(invalid_data(format!(
                "tensor {} row width {d0} is not divisible by block size {be}",
                self.name
            )));
        }
        (d0 / be_u64)
            .checked_mul(bs as u64)
            .ok_or_else(|| invalid_data(format!("tensor {} row_bytes overflow", self.name)))
    }

    pub fn row_bytes(&self) -> u64 {
        self.try_row_bytes().unwrap_or(0)
    }
}

enum Backing {
    Mmap { _file: File, map: *mut u8, map_len: usize },
    Owned(Box<[u8]>),
}

pub struct Gguf {
    backing: Backing,
    pub file_len: usize,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, Tensor>,
}

// SAFETY: `Backing::Mmap` wraps a read-only `PROT_READ` memory mapping whose backing file
// stays alive for the lifetime of `Gguf` and is unmapped only in `Drop`.
unsafe impl Send for Gguf {}
// SAFETY: `Backing::Mmap` is immutable (`PROT_READ`) and never mutated after creation.
unsafe impl Sync for Gguf {}

impl Drop for Gguf {
    fn drop(&mut self) {
        if let Backing::Mmap { map, map_len, .. } = self.backing {
            // SAFETY: `map` and `map_len` came from a successful `libc::mmap` in `Gguf::open`
            // with `map_len > 0` and are unmapped exactly once here.
            unsafe {
                libc::munmap(map as *mut _, map_len);
            }
        }
    }
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn rem(&self) -> usize {
        self.b.len().saturating_sub(self.p)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| invalid_data("GGUF reader offset overflow"))?;
        if end > self.b.len() {
            return Err(invalid_data("unexpected EOF while parsing GGUF"));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn str(&mut self) -> Result<String> {
        let len_u64 = self.u64()?;
        let n = usize::try_from(len_u64)
            .map_err(|_| invalid_data("GGUF string length exceeds address space"))?;
        let bytes = self.take(n)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn scalar_primitive(&mut self, t: u32) -> Result<Value> {
        match t {
            0 => Ok(Value::U(self.u8()? as u64)),
            1 => Ok(Value::I(self.u8()? as i8 as i64)),
            2 => Ok(Value::U(self.u16()? as u64)),
            3 => Ok(Value::I(self.u16()? as i16 as i64)),
            4 => Ok(Value::U(self.u32()? as u64)),
            5 => Ok(Value::I(self.u32()? as i32 as i64)),
            6 => Ok(Value::F(f32::from_bits(self.u32()?) as f64)),
            7 => Ok(Value::B(self.u8()? != 0)),
            8 => Ok(Value::S(self.str()?)),
            10 => Ok(Value::U(self.u64()?)),
            11 => Ok(Value::I(self.u64()? as i64)),
            12 => Ok(Value::F(f64::from_bits(self.u64()?))),
            _ => Err(invalid_data(format!("bad gguf value type {t}"))),
        }
    }

    fn scalar(&mut self, t: u32) -> Result<Value> {
        if t != 9 {
            return self.scalar_primitive(t);
        }
        let et = self.u32()?;
        let n_u64 = self.u64()?;
        if et == 9 {
            return Err(invalid_data("nested GGUF arrays are not supported"));
        }
        let min_elem_bytes: usize = match et {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            8 | 10 | 11 | 12 => 8,
            _ => return Err(invalid_data(format!("bad gguf array element type {et}"))),
        };
        let n = usize::try_from(n_u64)
            .map_err(|_| invalid_data("GGUF array length exceeds address space"))?;
        let min_bytes = n
            .checked_mul(min_elem_bytes)
            .ok_or_else(|| invalid_data("GGUF array byte size overflow"))?;
        if min_bytes > self.rem() {
            return Err(invalid_data("GGUF array length exceeds remaining file bytes"));
        }
        match et {
            8 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.str()?);
                }
                Ok(Value::AS(v))
            }
            7 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.u8()? != 0);
                }
                Ok(Value::AB(v))
            }
            6 | 12 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    match self.scalar_primitive(et)? {
                        Value::F(f) => v.push(f),
                        _ => unreachable!(),
                    }
                }
                Ok(Value::AF(v))
            }
            _ => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    match self.scalar_primitive(et)? {
                        Value::U(u) => v.push(u as i64),
                        Value::I(i) => v.push(i),
                        _ => unreachable!(),
                    }
                }
                Ok(Value::AI(v))
            }
        }
    }
}

impl Gguf {
    fn parse_parts(bytes: &[u8]) -> Result<(HashMap<String, Value>, HashMap<String, Tensor>)> {
        const MIN_KV_BYTES: usize = 8 + 4 + 1;
        const MIN_TENSOR_BYTES: usize = 8 + 4 + 4 + 8;

        let file_len = bytes.len();
        let mut r = Rd { b: bytes, p: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(invalid_data("not a GGUF file"));
        }
        let _ver = r.u32()?;
        let nt_u64 = r.u64()?;
        let nkv_u64 = r.u64()?;

        let nt = usize::try_from(nt_u64)
            .map_err(|_| invalid_data("GGUF tensor count exceeds address space"))?;
        let nkv = usize::try_from(nkv_u64)
            .map_err(|_| invalid_data("GGUF kv count exceeds address space"))?;

        let min_kv_bytes = nkv
            .checked_mul(MIN_KV_BYTES)
            .ok_or_else(|| invalid_data("GGUF kv count overflow"))?;
        let min_t_bytes = nt
            .checked_mul(MIN_TENSOR_BYTES)
            .ok_or_else(|| invalid_data("GGUF tensor count overflow"))?;
        let min_header_bytes = min_kv_bytes
            .checked_add(min_t_bytes)
            .ok_or_else(|| invalid_data("GGUF header count overflow"))?;
        if min_header_bytes > r.rem() {
            return Err(invalid_data("GGUF header counts exceed remaining file bytes"));
        }

        let mut kv = HashMap::with_capacity(nkv);
        for _ in 0..nkv {
            let k = r.str()?;
            let t = r.u32()?;
            let v = r.scalar(t)?;
            kv.insert(k, v);
        }

        let align = match kv.get("general.alignment") {
            None => 32u64,
            Some(Value::U(a)) if *a > 0 && *a <= u32::MAX as u64 => *a,
            Some(Value::I(a)) if *a > 0 && *a <= u32::MAX as i64 => *a as u64,
            Some(_) => {
                return Err(invalid_data(
                    "general.alignment must be a positive integer within u32 range",
                ))
            }
        };

        if min_t_bytes > r.rem() {
            return Err(invalid_data("GGUF tensor table exceeds remaining file bytes"));
        }
        let mut tl = Vec::with_capacity(nt);
        for _ in 0..nt {
            let name = r.str()?;
            let nd = r.u32()? as usize;
            if nd == 0 {
                return Err(invalid_data(format!("tensor {name} has zero rank")));
            }
            let ne_bytes = nd
                .checked_mul(8)
                .ok_or_else(|| invalid_data(format!("tensor {name} rank overflow")))?;
            if ne_bytes > r.rem() {
                return Err(invalid_data(format!(
                    "tensor {name} dimensions exceed remaining file bytes"
                )));
            }
            let mut ne = Vec::with_capacity(nd);
            for _ in 0..nd {
                ne.push(r.u64()?);
            }
            let ty = r.u32()?;
            let off = r.u64()?;
            let (bs, be) = try_type_block(ty)?;
            let mut n: u64 = 1;
            for &d in &ne {
                n = n.checked_mul(d).ok_or_else(|| {
                    invalid_data(format!("tensor {name} element count overflow"))
                })?;
            }
            let be_u64 = be as u64;
            let bs_u64 = bs as u64;
            if !ne[0].is_multiple_of(be_u64) || !n.is_multiple_of(be_u64) {
                return Err(invalid_data(format!(
                    "tensor {name} shape {:?} is not divisible by block size {be}",
                    ne
                )));
            }
            let nbytes = (n / be_u64)
                .checked_mul(bs_u64)
                .ok_or_else(|| invalid_data(format!("tensor {name} byte size overflow")))?;
            tl.push(Tensor {
                name,
                ne,
                ty,
                file_off: off,
                nbytes,
            });
        }

        let data_off = (r.p as u64)
            .checked_next_multiple_of(align)
            .ok_or_else(|| invalid_data("GGUF data offset alignment overflow"))?;
        let file_len_u64 = file_len as u64;
        let mut tensors = HashMap::with_capacity(tl.len());
        for mut t in tl {
            t.file_off = t
                .file_off
                .checked_add(data_off)
                .ok_or_else(|| invalid_data(format!("tensor {} offset overflow", t.name)))?;
            let end = t
                .file_off
                .checked_add(t.nbytes)
                .ok_or_else(|| invalid_data(format!("tensor {} end offset overflow", t.name)))?;
            if end > file_len_u64 {
                return Err(invalid_data(format!("tensor {} out of bounds", t.name)));
            }
            tensors.insert(t.name.clone(), t);
        }
        Ok((kv, tensors))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Gguf> {
        let owned: Box<[u8]> = bytes.into();
        let (kv, tensors) = Self::parse_parts(&owned)?;
        let file_len = owned.len();
        Ok(Gguf {
            backing: Backing::Owned(owned),
            file_len,
            kv,
            tensors,
        })
    }

    pub fn open(path: &str) -> Result<Gguf> {
        let file = File::open(path)?;
        let file_len_u64 = file.metadata()?.len();
        if file_len_u64 == 0 {
            return Err(invalid_data("empty GGUF file"));
        }
        let file_len = usize::try_from(file_len_u64)
            .map_err(|_| invalid_data("GGUF file too large for address space"))?;
        let page = 16384usize;
        let map_len = file_len
            .checked_next_multiple_of(page)
            .ok_or_else(|| invalid_data("GGUF mmap length overflow"))?;
        // SAFETY: `file` is an open file descriptor, `map_len` is non-zero and page-aligned,
        // and the returned pointer is checked against `MAP_FAILED` before use.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(Error::last_os_error());
        }
        let map = map as *mut u8;
        // Construct `Gguf` before parsing so `Drop` always calls `munmap` if parsing fails.
        let mut g = Gguf {
            backing: Backing::Mmap {
                _file: file,
                map,
                map_len,
            },
            file_len,
            kv: HashMap::new(),
            tensors: HashMap::new(),
        };
        let (kv, tensors) = Self::parse_parts(g.as_bytes())?;
        g.kv = kv;
        g.tensors = tensors;
        Ok(g)
    }

    pub fn raw_map(&self) -> Option<(*mut u8, usize)> {
        match &self.backing {
            Backing::Mmap { map, map_len, .. } => Some((*map, *map_len)),
            Backing::Owned(_) => None,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            Backing::Mmap { map, .. } => {
                // SAFETY: `map` came from a successful `mmap` of length `map_len >= self.file_len`
                // and remains valid and read-only for the lifetime of `&self`.
                unsafe { std::slice::from_raw_parts(*map, self.file_len) }
            }
            Backing::Owned(buf) => buf,
        }
    }

    pub fn try_u(&self, k: &str) -> Result<u64> {
        match self.kv.get(k) {
            Some(Value::U(u)) => Ok(*u),
            Some(Value::I(i)) if *i >= 0 => Ok(*i as u64),
            Some(Value::B(b)) => Ok(*b as u64),
            v => Err(invalid_data(format!("kv {k} is not a non-negative integer: {v:?}"))),
        }
    }

    pub fn try_f(&self, k: &str) -> Result<f64> {
        match self.kv.get(k) {
            Some(Value::F(f)) if f.is_finite() => Ok(*f),
            Some(Value::U(u)) => Ok(*u as f64),
            Some(Value::I(i)) => Ok(*i as f64),
            v => Err(invalid_data(format!("kv {k} is not a finite number: {v:?}"))),
        }
    }

    pub fn u(&self, k: &str) -> u64 {
        match self.kv.get(k) {
            Some(Value::U(u)) => *u,
            Some(Value::I(i)) => *i as u64,
            Some(Value::B(b)) => *b as u64,
            v => panic!("kv {k} not an integer: {v:?}"),
        }
    }

    pub fn f(&self, k: &str) -> f64 {
        match self.kv.get(k) {
            Some(Value::F(f)) => *f,
            _ => self.u(k) as f64,
        }
    }

    pub fn tensor(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    pub fn opt(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn bytes(&self, t: &Tensor) -> &[u8] {
        &self.as_bytes()[t.file_off as usize..(t.file_off + t.nbytes) as usize]
    }

    /// Parallel load-time finiteness audit across all tensors in this GGUF.
    ///
    /// For floating-point tensors (`F32`, `F16`, `BF16`) every element is checked; for
    /// quantized tensors (`Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`) only the fp16
    /// block scale header(s) (`d`, and `dmin` for `Q4_K`/`Q5_K`) are inspected without
    /// dequantizing payload bytes. Offenders are returned sorted deterministically by name.
    pub fn audit_weights_finiteness(&self) -> WeightAuditReport {
        let mut list: Vec<&Tensor> = self.tensors.values().collect();
        let tensors_checked = list.len();
        if tensors_checked == 0 {
            return WeightAuditReport {
                tensors_checked: 0,
                blocks_checked: 0,
                non_finite_count: 0,
                offenders: Vec::new(),
            };
        }

        // Largest tensors first (LPT) so multi-threaded workers stay balanced across MoE weights.
        list.sort_unstable_by_key(|a| std::cmp::Reverse(a.nbytes));

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8);
        let n_threads = tensors_checked.min(max_threads);

        let mut blocks_checked = 0u64;
        let mut non_finite_count = 0u64;
        let mut offenders = Vec::new();

        if n_threads <= 1 {
            for t in list {
                let (blocks, bad) = audit_tensor_bytes(t.ty, self.bytes(t));
                blocks_checked += blocks;
                if bad > 0 {
                    non_finite_count += bad;
                    offenders.push(WeightOffender {
                        tensor: t.name.clone(),
                        ty: t.ty,
                        non_finite: bad,
                    });
                }
            }
        } else {
            let next_idx = std::sync::atomic::AtomicUsize::new(0);
            let list_ref: &[&Tensor] = &list;
            let partials: Vec<(u64, u64, Vec<WeightOffender>)> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(n_threads);
                for _ in 0..n_threads {
                    handles.push(s.spawn(|| {
                        let mut local_blocks = 0u64;
                        let mut local_bad = 0u64;
                        let mut local_offenders = Vec::new();
                        loop {
                            let idx =
                                next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if idx >= list_ref.len() {
                                break;
                            }
                            let t = list_ref[idx];
                            let (blocks, bad) = audit_tensor_bytes(t.ty, self.bytes(t));
                            local_blocks += blocks;
                            if bad > 0 {
                                local_bad += bad;
                                local_offenders.push(WeightOffender {
                                    tensor: t.name.clone(),
                                    ty: t.ty,
                                    non_finite: bad,
                                });
                            }
                        }
                        (local_blocks, local_bad, local_offenders)
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("weight audit worker panicked"))
                    .collect()
            });
            for (b, bad, mut offs) in partials {
                blocks_checked += b;
                non_finite_count += bad;
                offenders.append(&mut offs);
            }
        }

        offenders.sort_by(|a, b| a.tensor.cmp(&b.tensor));
        WeightAuditReport {
            tensors_checked,
            blocks_checked,
            non_finite_count,
            offenders,
        }
    }

    /// Validate that this GGUF conforms to all `inkling2_loop` architecture invariants,
    /// tensor ranks/dimensions, quantization types, byte bounds, and vocabulary sizes
    /// required by `Engine` and its Metal kernels.
    pub fn validate_inkling_model(&self) -> Result<()> {
        let arch = match self.kv.get("general.architecture") {
            Some(Value::S(s)) => s.as_str(),
            _ => return Err(invalid_data("missing or invalid general.architecture")),
        };
        if arch != "inkling2_loop" {
            return Err(invalid_data(format!("unsupported architecture {arch}")));
        }
        let a = |k: &str| format!("inkling2_loop.{k}");
        let expect_u = |k: &str, expected: usize| -> Result<()> {
            let full = a(k);
            let got = self.try_u(&full)?;
            if got != expected as u64 {
                return Err(invalid_data(format!(
                    "kv {full} mismatch: expected {expected}, got {got}"
                )));
            }
            Ok(())
        };

        expect_u("embedding_length", E)?;
        expect_u("block_count", N_PHYS)?;
        expect_u("expert_count", NEXP)?;
        expect_u("expert_used_count", TOPK)?;
        expect_u("loop.start", LOOP_START)?;
        expect_u("loop.end", LOOP_END)?;
        expect_u("loop.count", N_PASS)?;
        expect_u("attn_res.block_size", BLOCK)?;
        expect_u("attention.sliding_window", EXT)?;
        expect_u("attention.relative_extent", EXT)?;

        let swa_pat = match self.kv.get(&a("attention.sliding_window_pattern")) {
            Some(Value::AB(v)) if v.len() >= N_PHYS => v,
            Some(Value::AB(v)) => {
                return Err(invalid_data(format!(
                    "attention.sliding_window_pattern too short: {} < {N_PHYS}",
                    v.len()
                )));
            }
            _ => return Err(invalid_data("missing or invalid attention.sliding_window_pattern")),
        };

        const MIN_RMS_EPS: f32 = 1e-12;
        let eps = self.try_f(&a("attention.layer_norm_rms_epsilon"))?;
        let eps_f32 = eps as f32;
        if !eps_f32.is_finite() || eps_f32 < MIN_RMS_EPS {
            return Err(invalid_data(format!(
                "attention.layer_norm_rms_epsilon ({eps}) must be finite in f32 and >= {MIN_RMS_EPS:e}"
            )));
        }
        let _log_alpha = self.try_f(&a("attention.log_scale_alpha"))?;
        let log_floor = self.try_f(&a("attention.log_scale_floor"))?;
        if log_floor <= 0.0 {
            return Err(invalid_data("attention.log_scale_floor must be > 0"));
        }
        let situ_beta = self.try_f(&a("activation.situ_beta"))?;
        if situ_beta == 0.0 {
            return Err(invalid_data("activation.situ_beta must be non-zero"));
        }
        let _situ_lbeta = self.try_f(&a("activation.situ_linear_beta"))?;
        let _wscale = self.try_f(&a("expert_weights_scale"))?;
        let softcap = self.try_f(&a("final_logit_softcapping"))?;
        let softcap_f32 = softcap as f32;
        if !softcap_f32.is_finite() || softcap_f32 < 0.0 {
            return Err(invalid_data(format!(
                "final_logit_softcapping ({softcap}) must be finite in f32 and >= 0"
            )));
        }

        let n_vocab = self.try_u(&a("vocab_size_unpadded"))?;
        if n_vocab == 0 {
            return Err(invalid_data("vocab_size_unpadded must be > 0"));
        }

        let check_bounds_and_size = |t: &Tensor| -> Result<()> {
            let (bs, be) = try_type_block(t.ty)?;
            let mut n: u64 = 1;
            for &d in &t.ne {
                n = n.checked_mul(d).ok_or_else(|| {
                    invalid_data(format!("tensor {} element count overflow", t.name))
                })?;
            }
            let expected_nbytes = (n / be as u64)
                .checked_mul(bs as u64)
                .ok_or_else(|| invalid_data(format!("tensor {} byte size overflow", t.name)))?;
            if t.nbytes != expected_nbytes {
                return Err(invalid_data(format!(
                    "tensor {} nbytes mismatch: expected {expected_nbytes}, got {}",
                    t.name, t.nbytes
                )));
            }
            let end = t
                .file_off
                .checked_add(t.nbytes)
                .ok_or_else(|| invalid_data(format!("tensor {} end offset overflow", t.name)))?;
            if end > self.file_len as u64 {
                return Err(invalid_data(format!("tensor {} out of bounds", t.name)));
            }
            let rb = t.try_row_bytes()?;
            if rb > u32::MAX as u64 {
                return Err(invalid_data(format!(
                    "tensor {} row_bytes {rb} exceeds u32::MAX",
                    t.name
                )));
            }
            Ok(())
        };

        let check = |name: &str, allowed_types: &[u32], allowed_shapes: &[&[u64]]| -> Result<&Tensor> {
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| invalid_data(format!("missing required tensor {name}")))?;
            if !allowed_types.contains(&t.ty) {
                return Err(invalid_data(format!(
                    "tensor {name} has unsupported type {} ({})",
                    t.ty,
                    type_name(t.ty)
                )));
            }
            if !allowed_shapes.contains(&t.ne.as_slice()) {
                return Err(invalid_data(format!(
                    "tensor {name} has invalid shape {:?}",
                    t.ne
                )));
            }
            check_bounds_and_size(t)?;
            Ok(t)
        };

        let e = E as u64;
        let nh = NH as u64;
        let nkv = NKV as u64;
        let hd = HD as u64;
        let nrel = NREL as u64;
        let ext = EXT as u64;
        let nexp = NEXP as u64;
        let ff = FF as u64;

        for (i, &swa) in swa_pat.iter().take(N_PHYS).enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}");
            if swa {
                check(&n("attn_norm.weight"), &[T_F32], &[&[e]])?;
                if self.tensors.contains_key(&n("attn_res_query.weight")) {
                    check(&n("attn_res_query.weight"), &[T_F32], &[&[e], &[e, 1]])?;
                }
                if self.tensors.contains_key(&n("attn_k_conv_proj.weight")) {
                    check(&n("attn_k_conv_proj.weight"), &[T_Q4_K, T_Q6_K], &[&[e, nkv * 4]])?;
                }
                if self.tensors.contains_key(&n("attn_v_conv_proj.weight")) {
                    check(&n("attn_v_conv_proj.weight"), &[T_Q4_K, T_Q6_K], &[&[e, nkv * 4]])?;
                }
            } else {
                if self.tensors.contains_key(&n("attn_norm.weight")) {
                    return Err(invalid_data(format!(
                        "full-attention layer {i} must not have attn_norm.weight"
                    )));
                }
                check(&n("attn_q_inp_norm.weight"), &[T_F32], &[&[e]])?;
                check(&n("attn_k_inp_norm.weight"), &[T_F32], &[&[e]])?;
                check(&n("attn_v_inp_norm.weight"), &[T_F32], &[&[e]])?;
                check(&n("attn_res_query.weight"), &[T_F32], &[&[e, 3]])?;
                check(&n("attn_k_conv_proj.weight"), &[T_Q4_K, T_Q6_K], &[&[e, nkv * 4]])?;
                check(&n("attn_v_conv_proj.weight"), &[T_Q4_K, T_Q6_K], &[&[e, nkv * 4]])?;
            }

            check(&n("attn_q.weight"), &[T_Q4_K], &[&[e, nh * hd]])?;
            check(&n("attn_gate.weight"), &[T_Q4_K], &[&[e, nh * hd]])?;
            check(&n("attn_k.weight"), &[T_Q4_K], &[&[e, nkv * hd]])?;
            check(&n("attn_v.weight"), &[T_Q4_K, T_Q6_K], &[&[e, nkv * hd]])?;
            check(&n("attn_output.weight"), &[T_Q4_K], &[&[nh * hd, e]])?;
            check(&n("attn_rel.weight"), &[T_Q4_K], &[&[e, nh * nrel]])?;
            check(&n("attn_sink_proj.weight"), &[T_Q4_K], &[&[e, nh]])?;

            check(&n("ffn_res_query.weight"), &[T_F32], &[&[e], &[e, 1]])?;
            check(&n("ffn_norm.weight"), &[T_F32], &[&[e]])?;
            check(&n("attn_q_clip.weight"), &[T_F32], &[&[hd, nh], &[nh * hd]])?;
            check(&n("attn_rel_clip.weight"), &[T_F32], &[&[nrel, nh], &[nh * nrel]])?;
            check(&n("attn_rel_bias.weight"), &[T_F32], &[&[nrel, ext], &[ext * nrel]])?;
            check(
                &n("attn_k_conv.weight"),
                &[T_F32],
                &[&[4, nkv * hd], &[4, 1, nkv * hd], &[4, hd, nkv], &[4 * nkv * hd]],
            )?;
            check(
                &n("attn_v_conv.weight"),
                &[T_F32],
                &[&[4, nkv * hd], &[4, 1, nkv * hd], &[4, hd, nkv], &[4 * nkv * hd]],
            )?;
            check(
                &n("attn_post_conv.weight"),
                &[T_F32],
                &[&[4, e], &[4, 1, e], &[4 * e]],
            )?;
            check(
                &n("ffn_post_conv.weight"),
                &[T_F32],
                &[&[4, e], &[4, 1, e], &[4 * e]],
            )?;
            check(&n("ffn_gate_inp.weight"), &[T_F32], &[&[e, nexp]])?;
            check(&n("exp_probs_b.bias"), &[T_F32], &[&[nexp]])?;

            let gu = check(&n("ffn_gate_up_exps.weight"), &[T_Q4_K], &[&[e, 2 * ff, nexp]])?;
            let gu_exp_bytes = gu
                .try_row_bytes()?
                .checked_mul(gu.ne[1])
                .ok_or_else(|| invalid_data(format!("{} expert_bytes overflow", gu.name)))?;
            if gu_exp_bytes > u32::MAX as u64 {
                return Err(invalid_data(format!(
                    "{} expert_bytes exceeds u32::MAX",
                    gu.name
                )));
            }

            let dn = check(&n("ffn_down_exps.weight"), &[T_Q8_0, T_Q5_0], &[&[ff, e, nexp]])?;
            let dn_exp_bytes = dn
                .try_row_bytes()?
                .checked_mul(dn.ne[1])
                .ok_or_else(|| invalid_data(format!("{} expert_bytes overflow", dn.name)))?;
            if dn_exp_bytes > u32::MAX as u64 {
                return Err(invalid_data(format!(
                    "{} expert_bytes exceeds u32::MAX",
                    dn.name
                )));
            }
        }

        for i in LOOP_START..LOOP_END {
            let n = |s: &str| format!("blk.{i}.{s}");
            check(&n("loop_msg_query.weight"), &[T_F32], &[&[e, 4]])?;
            for c in ["q", "k", "v", "ffn"] {
                check(&n(&format!("loop_{c}_ctrl_down.weight")), &[T_F32], &[&[2 * e, 64]])?;
                check(&n(&format!("loop_{c}_ctrl_gain.weight")), &[T_F32], &[&[64, e]])?;
                check(&n(&format!("loop_{c}_ctrl_forget.weight")), &[T_F32], &[&[64, e]])?;
            }
        }

        check("token_embd_norm.weight", &[T_F32], &[&[e]])?;
        check("output_norm.weight", &[T_F32], &[&[e]])?;
        check("output_res_query.weight", &[T_F32], &[&[e], &[e, 1]])?;

        let tok_embd = self
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| invalid_data("missing required tensor token_embd.weight"))?;
        if tok_embd.ty != T_Q4_K {
            return Err(invalid_data(format!(
                "token_embd.weight has unsupported type {}",
                type_name(tok_embd.ty)
            )));
        }
        if tok_embd.ne.len() != 2 || tok_embd.ne[0] != e || tok_embd.ne[1] == 0 || tok_embd.ne[1] > u32::MAX as u64 {
            return Err(invalid_data(format!(
                "token_embd.weight has invalid shape {:?}",
                tok_embd.ne
            )));
        }
        check_bounds_and_size(tok_embd)?;

        let output = self
            .tensors
            .get("output.weight")
            .ok_or_else(|| invalid_data("missing required tensor output.weight"))?;
        if output.ty != T_Q6_K {
            return Err(invalid_data(format!(
                "output.weight has unsupported type {}",
                type_name(output.ty)
            )));
        }
        if output.ne.len() != 2 || output.ne[0] != e || output.ne[1] == 0 || output.ne[1] > u32::MAX as u64 {
            return Err(invalid_data(format!(
                "output.weight has invalid shape {:?}",
                output.ne
            )));
        }
        check_bounds_and_size(output)?;

        let n_vocab_pad = output.ne[1];
        if n_vocab > n_vocab_pad {
            return Err(invalid_data(format!(
                "vocab_size_unpadded ({n_vocab}) exceeds output.weight rows ({n_vocab_pad})"
            )));
        }
        if n_vocab > tok_embd.ne[1] {
            return Err(invalid_data(format!(
                "vocab_size_unpadded ({n_vocab}) exceeds token_embd.weight rows ({})",
                tok_embd.ne[1]
            )));
        }
        if let Some(toks_val) = self.kv.get("tokenizer.ggml.tokens") {
            match toks_val {
                Value::AS(toks) => {
                    if (toks.len() as u64) > tok_embd.ne[1] {
                        return Err(invalid_data(format!(
                            "tokenizer.ggml.tokens length ({}) exceeds token_embd.weight rows ({})",
                            toks.len(),
                            tok_embd.ne[1]
                        )));
                    }
                }
                _ => {
                    return Err(invalid_data(
                        "tokenizer.ggml.tokens must be a string array",
                    ));
                }
            }
        }

        Ok(())
    }
}
