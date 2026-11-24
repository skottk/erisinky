//! CPU sampler over the GPU logits: softcap -> top-k -> temperature -> top-p -> draw.

pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub softcap: f32,
    state: u64,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(temp: f32, top_k: usize, top_p: f32, softcap: f32, seed: u64) -> Sampler {
        Sampler {
            temp,
            top_k,
            top_p,
            softcap,
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            cand: Vec::with_capacity(256),
        }
    }

    pub fn greedy(&self) -> bool {
        self.temp <= 0.0
    }

    fn next_f32(&mut self) -> f32 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Sample a token ID from `logits`, returning `Err` if `logits` is empty or all `NaN`.
    /// `top_k == 0` disables top-k filtering (considers the full vocabulary).
    pub fn try_sample(&mut self, logits: &[f32]) -> Result<u32, &'static str> {
        if logits.is_empty() {
            return Err("empty logits");
        }
        // top-k on raw logits (softcap is monotonic, so the order is the same); top_k == 0 disables top-k
        let k = if self.top_k == 0 {
            logits.len()
        } else {
            self.top_k.min(logits.len())
        };
        self.cand.clear();
        // Phase 1: fill up to `k` non-NaN candidates (no NaN ever enters `self.cand`).
        let mut start = logits.len();
        for (i, &x) in logits.iter().enumerate() {
            if x.is_nan() {
                continue;
            }
            self.cand.push((x, i as u32));
            if self.cand.len() == k {
                start = i + 1;
                break;
            }
        }
        // Phase 2: 8-wide SIMD-friendly scan over the remaining vocabulary.
        // `x > floor` is `false` for both `x <= floor` and `x.is_nan()`, so NaN
        // filtering costs zero extra branches while LLVM vectorizes the 8-lane compare.
        if start < logits.len() {
            #[inline]
            fn find_min(cand: &[(f32, u32)]) -> (usize, f32) {
                let mut min_i = 0usize;
                let mut min_v = cand[0].0;
                for (j, &(v, _)) in cand.iter().enumerate().skip(1) {
                    if v < min_v {
                        min_v = v;
                        min_i = j;
                    }
                }
                (min_i, min_v)
            }
            let (mut floor_idx, mut floor) = find_min(&self.cand);
            let mut idx = start;
            let (chunks, rem) = logits[start..].as_chunks::<8>();
            for chunk in chunks {
                if (chunk[0] > floor)
                    | (chunk[1] > floor)
                    | (chunk[2] > floor)
                    | (chunk[3] > floor)
                    | (chunk[4] > floor)
                    | (chunk[5] > floor)
                    | (chunk[6] > floor)
                    | (chunk[7] > floor)
                {
                    for (j, &x) in chunk.iter().enumerate() {
                        if x > floor {
                            self.cand[floor_idx] = (x, (idx + j) as u32);
                            (floor_idx, floor) = find_min(&self.cand);
                        }
                    }
                }
                idx += 8;
            }
            for (j, &x) in rem.iter().enumerate() {
                if x > floor {
                    self.cand[floor_idx] = (x, (idx + j) as u32);
                    (floor_idx, floor) = find_min(&self.cand);
                }
            }
        }
        if self.cand.is_empty() {
            return Err("all logits are NaN");
        }
        self.cand
            .sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let t = self.temp;
        if t <= 0.0 || !t.is_finite() || self.cand.len() == 1 {
            return Ok(self.cand[0].1);
        }

        let c = self.softcap;
        let use_softcap = c.is_finite() && c > 0.0;
        let sc: Vec<f32> = self
            .cand
            .iter()
            .map(|&(x, _)| if use_softcap { c * (x / c).tanh() } else { x })
            .collect();
        let max_sc = sc[0];

        let mut p: Vec<f32> = if max_sc == f32::INFINITY {
            sc.iter()
                .map(|&x| if x == f32::INFINITY { 1.0 } else { 0.0 })
                .collect()
        } else if max_sc == f32::NEG_INFINITY {
            vec![1.0; sc.len()]
        } else {
            // Subtract max_sc before dividing by t so 0.0 / t == 0.0 even at tiny positive t.
            sc.iter().map(|&x| ((x - max_sc) / t).exp()).collect()
        };

        let sum: f32 = p.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Ok(self.cand[0].1);
        }
        for v in p.iter_mut() {
            *v /= sum;
        }

        // top-p cut
        let top_p = if self.top_p.is_nan() { 1.0 } else { self.top_p };
        let mut cum = 0.0f32;
        let mut n = p.len();
        for (i, &v) in p.iter().enumerate() {
            cum += v;
            if cum >= top_p {
                n = i + 1;
                break;
            }
        }
        while n > 1 && p[n - 1] == 0.0 {
            n -= 1;
        }
        let total: f32 = p[..n].iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Ok(self.cand[0].1);
        }
        let mut r = self.next_f32() * total;
        for (i, &prob) in p[..n].iter().enumerate() {
            r -= prob;
            if r <= 0.0 {
                return Ok(self.cand[i].1);
            }
        }
        Ok(self.cand[n - 1].1)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.try_sample(logits).unwrap_or(0)
    }
}
