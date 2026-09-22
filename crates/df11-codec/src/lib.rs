//! df11-codec -- the DFloat11 encoder.
//!
//! Nothing is implemented yet. Phase 1 fills this in, graded against the
//! frozen Phase 0 fixtures (see `df11-fixtures` and `docs/PLAN.md`).
//!
//! Read `docs/COMPATIBILITY.md` before touching the LUT path: the default
//! mode reproduces a bug in the official encoder on purpose.

/// Split a little-endian BF16 buffer into the two streams DF11 stores.
///
/// A BF16 weight is `s eeeeeeee mmmmmmm`. The official encoder derives, from the
/// int16 pattern `W`:
///
/// ```text
/// exponent      = (W >> 7) & 0xFF
/// sign_mantissa = ((W >> 8) & 0x80) | (W & 0x7F)
/// ```
///
/// so `sign_mantissa` carries the sign in bit 7 and the mantissa in bits 0..=6.
/// The official code shifts a *signed* int16, but both results are masked, so an
/// unsigned shift is equivalent.
///
/// Returns `(exponents, sign_mantissa)`, one byte each per weight.
///
/// # Panics
/// If `bf16_le` has an odd length; a BF16 buffer is always an even number of bytes.
pub fn split_fields(bf16_le: &[u8]) -> (Vec<u8>, Vec<u8>) {
    assert!(
        bf16_le.len() % 2 == 0,
        "BF16 buffer must have an even length, got {}",
        bf16_le.len()
    );
    let n = bf16_le.len() / 2;
    let mut exponents = Vec::with_capacity(n);
    let mut sign_mantissa = Vec::with_capacity(n);
    for c in bf16_le.chunks_exact(2) {
        let w = u16::from_le_bytes([c[0], c[1]]);
        exponents.push(((w >> 7) & 0xFF) as u8);
        sign_mantissa.push((((w >> 8) & 0x80) | (w & 0x7F)) as u8);
    }
    (exponents, sign_mantissa)
}
