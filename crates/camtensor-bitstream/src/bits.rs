//! RBSP extraction (emulation-prevention removal) and an MSB-first bit reader
//! with Exp-Golomb support, as used by H.264 / H.265 syntax.

use std::borrow::Cow;

use crate::{Error, Result};

/// Removes `emulation_prevention_three_byte`s: every `00 00 03` becomes
/// `00 00`. Borrows the input when there is nothing to remove.
pub fn remove_emulation_prevention(nal: &[u8]) -> Cow<'_, [u8]> {
    let first = find_epb(nal, 0);
    let Some(mut epb) = first else {
        return Cow::Borrowed(nal);
    };
    let mut out = Vec::with_capacity(nal.len());
    let mut start = 0;
    loop {
        // Keep `00 00`, drop the `03`.
        out.extend_from_slice(&nal[start..epb]);
        start = epb + 1;
        match find_epb(nal, start) {
            Some(next) => epb = next,
            None => break,
        }
    }
    out.extend_from_slice(&nal[start..]);
    Cow::Owned(out)
}

/// Finds the index of the next `03` that follows `00 00`, with the zero run
/// counted from `from` (an emulation-prevention byte resets the run).
fn find_epb(data: &[u8], from: usize) -> Option<usize> {
    let mut zeros = 0usize;
    for (i, &b) in data.iter().enumerate().skip(from) {
        if zeros >= 2 && b == 3 {
            return Some(i);
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    None
}

/// MSB-first bit reader over an RBSP (emulation prevention already removed).
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Creates a reader positioned at the first bit of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits not read yet.
    pub fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    /// Current position in bits.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// `true` when positioned on a byte boundary.
    pub fn byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }

    /// Reads one bit as a flag, `u(1)`.
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.u(1)? == 1)
    }

    /// Reads an unsigned `n`-bit value, `u(n)`, with `n <= 32`.
    pub fn u(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        Ok(self.u64(n)? as u32)
    }

    /// Reads an unsigned `n`-bit value with `n <= 64`.
    pub fn u64(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        let n = n as usize;
        if n > self.bits_left() {
            return Err(Error::Truncated);
        }
        let mut v = 0u64;
        let mut remaining = n;
        while remaining > 0 {
            let byte = self.data[self.pos / 8];
            let bit_off = self.pos % 8;
            let avail = 8 - bit_off;
            let take = avail.min(remaining);
            let shift = avail - take;
            let bits = (byte >> shift) & (((1u16 << take) - 1) as u8);
            v = (v << take) | u64::from(bits);
            remaining -= take;
            self.pos += take;
        }
        Ok(v)
    }

    /// Skips `n` bits.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        if n > self.bits_left() {
            return Err(Error::Truncated);
        }
        self.pos += n;
        Ok(())
    }

    /// Reads an unsigned Exp-Golomb code, `ue(v)` (values up to `2^32 - 2`).
    pub fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        while !self.flag()? {
            zeros += 1;
            if zeros > 31 {
                return Err(Error::Invalid("exp-golomb code longer than 32 bits"));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let rest = self.u64(zeros)?;
        Ok(((1u64 << zeros) - 1 + rest) as u32)
    }

    /// Reads a signed Exp-Golomb code, `se(v)`.
    pub fn se(&mut self) -> Result<i32> {
        let k = i64::from(self.ue()?);
        let v = if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) };
        Ok(v as i32)
    }

    /// `ue(v)` with an upper bound check.
    pub fn ue_max(&mut self, max: u32, what: &'static str) -> Result<u32> {
        let v = self.ue()?;
        if v > max {
            return Err(Error::Invalid(what));
        }
        Ok(v)
    }

    /// `more_rbsp_data()`: `true` if there is data before the RBSP trailing
    /// bits (the last `1` bit followed by zero bits).
    pub fn more_rbsp_data(&self) -> bool {
        let Some(last) = self.data.iter().rposition(|&b| b != 0) else {
            return false;
        };
        let trailing_bit = last * 8 + 7 - self.data[last].trailing_zeros() as usize;
        self.pos < trailing_bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epb_removal() {
        assert!(matches!(remove_emulation_prevention(&[1, 2, 3]), Cow::Borrowed(_)));
        assert_eq!(&*remove_emulation_prevention(&[0, 0, 3, 1]), &[0, 0, 1]);
        assert_eq!(&*remove_emulation_prevention(&[0, 0, 3, 0, 0, 3]), &[0, 0, 0, 0]);
        // `00 00 03 03`: only the first 03 is emulation prevention.
        assert_eq!(&*remove_emulation_prevention(&[0, 0, 3, 3]), &[0, 0, 3]);
        // `00 03` is not an escape.
        assert_eq!(&*remove_emulation_prevention(&[0, 3, 0, 0, 3, 2]), &[0, 3, 0, 0, 2]);
        // Zero run restarts after the escape: 00 00 03 00 03 keeps the second 03.
        assert_eq!(&*remove_emulation_prevention(&[0, 0, 3, 0, 3]), &[0, 0, 0, 3]);
    }

    #[test]
    fn fixed_width() {
        let mut r = BitReader::new(&[0b1010_1100, 0xff, 0x01]);
        assert!(r.flag().unwrap());
        assert_eq!(r.u(3).unwrap(), 0b010);
        assert_eq!(r.u(8).unwrap(), 0b1100_1111);
        assert_eq!(r.bits_left(), 12);
        assert_eq!(r.u(12).unwrap(), 0xf01);
        assert_eq!(r.u(1), Err(Error::Truncated));
        let mut r = BitReader::new(&[0x12, 0x34, 0x56, 0x78, 0x9a]);
        assert_eq!(r.u(32).unwrap(), 0x1234_5678);
        assert_eq!(r.u64(8).unwrap(), 0x9a);
    }

    #[test]
    fn exp_golomb() {
        // ue: 1 -> 0, 010 -> 1, 011 -> 2, 00100 -> 3, 00111 -> 6
        // bits: 1 010 011 00100 00111 -> 1010 0110 0100 0011 1(000 0000)
        let data = [0b1010_0110, 0b0100_0011, 0b1000_0000];
        let mut r = BitReader::new(&data);
        let v: Vec<u32> = (0..5).map(|_| r.ue().unwrap()).collect();
        assert_eq!(v, [0, 1, 2, 3, 6]);
        // se mapping: k=1 -> 1, k=2 -> -1, k=3 -> 2, k=6 -> -3
        let mut r = BitReader::new(&data);
        let v: Vec<i32> = (0..5).map(|_| r.se().unwrap()).collect();
        assert_eq!(v, [0, 1, -1, 2, -3]);
    }

    #[test]
    fn exp_golomb_limits() {
        // 31 zeros, the marker bit, then 31 one bits: 2^31 - 1 + 2^31 - 1.
        let mut data = vec![0u8; 3];
        data.push(0b0000_0001);
        data.extend([0xff, 0xff, 0xff, 0xff]);
        let mut r = BitReader::new(&data);
        assert_eq!(r.ue().unwrap(), u32::MAX - 1);
        // 32 zeros: invalid.
        let mut r = BitReader::new(&[0, 0, 0, 0, 0x80]);
        assert!(matches!(r.ue(), Err(Error::Invalid(_))));
        // Truncated code.
        let mut r = BitReader::new(&[0b0000_0001]);
        assert_eq!(r.ue(), Err(Error::Truncated));
    }

    #[test]
    fn rbsp_trailing() {
        let r = BitReader::new(&[0b1000_0000]);
        assert!(!r.more_rbsp_data());
        let mut r = BitReader::new(&[0b0110_0000, 0x00]);
        assert!(r.more_rbsp_data());
        r.skip(2).unwrap();
        assert!(!r.more_rbsp_data());
        assert!(!BitReader::new(&[0, 0]).more_rbsp_data());
    }
}
