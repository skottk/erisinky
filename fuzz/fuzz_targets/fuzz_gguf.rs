#![no_main]

use inkling::gguf::Gguf;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(g) = Gguf::from_bytes(data) {
        // Verify every parsed tensor slice matches its declared byte length without wrapping/OOB.
        for t in g.tensors.values() {
            let slice = g.bytes(t);
            assert_eq!(slice.len() as u64, t.nbytes);
        }

        // Exercise fallible metadata accessors on every key.
        for k in g.kv.keys() {
            let _ = g.try_u(k);
            let _ = g.try_f(k);
        }

        // Exercise the parallel weight & quant-scale finiteness audit.
        let report = g.audit_weights_finiteness();
        assert_eq!(report.tensors_checked, g.tensors.len());

        // Exercise full inkling2_loop schema validation (must return Ok or Err, never panic).
        let _ = g.validate_inkling_model();
    }
});
