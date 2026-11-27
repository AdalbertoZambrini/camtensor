//! Annex B byte streams, AVCC length-prefixed framing and the `avcC` /
//! `hvcC` decoder configuration records.

use crate::{Error, Result};

/// The 4-byte start code used for every NAL unit this crate writes.
pub const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Iterator over the NAL units of an Annex B byte stream.
///
/// Accepts 3- and 4-byte start codes (any number of leading zeros). Bytes
/// before the first start code are ignored, as are trailing zero bytes of
/// each NAL unit (`trailing_zero_8bits`). Yielded slices start with the NAL
/// header and never include start codes.
#[derive(Debug, Clone)]
pub struct NalIter<'a> {
    data: &'a [u8],
    /// Index of the first byte of the next NAL unit, if any.
    next: Option<usize>,
}

/// Iterates over the NAL units of an Annex B byte stream.
pub fn nal_units(data: &[u8]) -> NalIter<'_> {
    NalIter { data, next: find_start_code(data, 0).map(|i| i + 3) }
}

impl<'a> Iterator for NalIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        loop {
            let start = self.next?;
            let (end, next) = match find_start_code(self.data, start) {
                Some(sc) => (sc, Some(sc + 3)),
                None => (self.data.len(), None),
            };
            self.next = next;
            let mut nal = &self.data[start..end];
            while let [rest @ .., 0] = nal {
                nal = rest;
            }
            if !nal.is_empty() {
                return Some(nal);
            }
        }
    }
}

/// Returns the index of the next `00 00 01` at or after `from`.
pub fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from + 2;
    while i < data.len() {
        match data[i] {
            // Cannot be part of a start code ending at i, i+1 or i+2.
            b if b > 1 => i += 3,
            1 if data[i - 1] == 0 && data[i - 2] == 0 => return Some(i - 2),
            _ => i += 1,
        }
    }
    None
}

/// `true` if `data` begins with a 3- or 4-byte Annex B start code.
pub fn starts_with_start_code(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

/// Appends `nal` to `out` preceded by a 4-byte start code.
pub fn push_nal(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&START_CODE);
    out.extend_from_slice(nal);
}

/// Converts length-prefixed NAL units (AVCC / HVCC sample format, with
/// `length_size` bytes of big-endian length) to Annex B, appending to `out`.
pub fn avcc_to_annexb(data: &[u8], length_size: usize, out: &mut Vec<u8>) -> Result<()> {
    if !(1..=4).contains(&length_size) || length_size == 3 {
        return Err(Error::Invalid("NAL length size must be 1, 2 or 4"));
    }
    out.reserve(data.len() + data.len() / 64);
    let mut rest = data;
    while !rest.is_empty() {
        if rest.len() < length_size {
            return Err(Error::Truncated);
        }
        let len = rest[..length_size].iter().fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
        rest = &rest[length_size..];
        if len > rest.len() {
            return Err(Error::Truncated);
        }
        if len > 0 {
            push_nal(out, &rest[..len]);
        }
        rest = &rest[len..];
    }
    Ok(())
}

/// Parsed `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 5.3.3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvcConfig<'a> {
    /// `AVCProfileIndication`.
    pub profile: u8,
    /// `profile_compatibility`.
    pub compatibility: u8,
    /// `AVCLevelIndication`.
    pub level: u8,
    /// Size of the NAL length prefix in samples (1, 2 or 4).
    pub length_size: usize,
    /// Sequence parameter sets (NAL units, header included).
    pub sps: Vec<&'a [u8]>,
    /// Picture parameter sets.
    pub pps: Vec<&'a [u8]>,
}

impl<'a> AvcConfig<'a> {
    /// Parses an `avcC` record.
    pub fn parse(rec: &'a [u8]) -> Result<Self> {
        let mut c = Cursor(rec);
        if c.u8()? != 1 {
            return Err(Error::Invalid("avcC configurationVersion != 1"));
        }
        let profile = c.u8()?;
        let compatibility = c.u8()?;
        let level = c.u8()?;
        let length_size = usize::from(c.u8()? & 0x03) + 1;
        let num_sps = c.u8()? & 0x1f;
        let mut sps = Vec::with_capacity(usize::from(num_sps));
        for _ in 0..num_sps {
            sps.push(c.sized16()?);
        }
        let num_pps = c.u8()?;
        let mut pps = Vec::with_capacity(usize::from(num_pps));
        for _ in 0..num_pps {
            pps.push(c.sized16()?);
        }
        // Optional High-profile extension (chroma format, bit depth, SPS
        // extensions) is not needed: the SPS carries the same information.
        Ok(Self { profile, compatibility, level, length_size, sps, pps })
    }

    /// Parameter sets as Annex B (SPS first, then PPS).
    pub fn to_annexb(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in self.sps.iter().chain(&self.pps) {
            push_nal(&mut out, nal);
        }
        out
    }
}

/// Parsed `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15 8.3.3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcConfig<'a> {
    /// `general_profile_idc`.
    pub profile: u8,
    /// `general_tier_flag`.
    pub tier: bool,
    /// `general_level_idc`.
    pub level: u8,
    /// Size of the NAL length prefix in samples (1, 2 or 4).
    pub length_size: usize,
    /// NAL unit arrays as `(nal_unit_type, nal units)` in record order.
    pub arrays: Vec<(u8, Vec<&'a [u8]>)>,
}

impl<'a> HevcConfig<'a> {
    /// Parses an `hvcC` record.
    pub fn parse(rec: &'a [u8]) -> Result<Self> {
        let mut c = Cursor(rec);
        if c.u8()? != 1 {
            return Err(Error::Invalid("hvcC configurationVersion != 1"));
        }
        let ptl = c.u8()?;
        c.skip(4 + 6)?; // compatibility flags, constraint flags
        let level = c.u8()?;
        c.skip(2 + 1 + 1 + 1 + 1 + 2)?; // segmentation, parallelism, chroma, depths, avg fps
        let length_size = usize::from(c.u8()? & 0x03) + 1;
        let num_arrays = c.u8()?;
        let mut arrays = Vec::with_capacity(usize::from(num_arrays));
        for _ in 0..num_arrays {
            let nal_type = c.u8()? & 0x3f;
            let n = c.u16()?;
            let mut nals = Vec::with_capacity(usize::from(n));
            for _ in 0..n {
                nals.push(c.sized16()?);
            }
            arrays.push((nal_type, nals));
        }
        Ok(Self { profile: ptl & 0x1f, tier: ptl & 0x20 != 0, level, length_size, arrays })
    }

    /// Parameter sets as Annex B, ordered VPS, SPS, PPS, then any other
    /// arrays (SEI) in record order.
    pub fn to_annexb(&self) -> Vec<u8> {
        let rank = |t: u8| match t {
            32 => 0,
            33 => 1,
            34 => 2,
            _ => 3,
        };
        let mut arrays: Vec<_> = self.arrays.iter().collect();
        arrays.sort_by_key(|(t, _)| rank(*t)); // stable: keeps record order otherwise
        let mut out = Vec::new();
        for (_, nals) in arrays {
            for nal in nals {
                push_nal(&mut out, nal);
            }
        }
        out
    }
}

/// Converts an `avcC` record to Annex B parameter sets; returns them with the
/// sample NAL length size.
pub fn avcc_config_to_annexb(rec: &[u8]) -> Result<(Vec<u8>, usize)> {
    let c = AvcConfig::parse(rec)?;
    Ok((c.to_annexb(), c.length_size))
}

/// Converts an `hvcC` record to Annex B parameter sets; returns them with the
/// sample NAL length size.
pub fn hvcc_config_to_annexb(rec: &[u8]) -> Result<(Vec<u8>, usize)> {
    let c = HevcConfig::parse(rec)?;
    Ok((c.to_annexb(), c.length_size))
}

struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            return Err(Error::Truncated);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn sized16(&mut self) -> Result<&'a [u8]> {
        let n = self.u16()?;
        self.take(usize::from(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::*;

    #[test]
    fn iterates_mixed_start_codes() {
        let data = [
            0xaa, // garbage before the first start code
            0, 0, 0, 1, 0x67, 1, 2, // 4-byte
            0, 0, 1, 0x68, 3, // 3-byte
            0, 0, 0, 0, 1, 0x65, 0, 0, 3, 1, 0, 0, // extra zeros, EPB inside, trailing zeros
            0, 0, 1, // empty NAL at the end
        ];
        let nals: Vec<&[u8]> = nal_units(&data).collect();
        assert_eq!(nals, vec![&[0x67, 1, 2][..], &[0x68, 3], &[0x65, 0, 0, 3, 1]]);
        assert_eq!(nal_units(&[1, 2, 3]).count(), 0);
        assert_eq!(nal_units(&[]).count(), 0);
        assert_eq!(nal_units(&[0, 0, 1]).count(), 0);
        assert_eq!(nal_units(&[0, 0, 1, 9]).collect::<Vec<_>>(), vec![&[9][..]]);
    }

    #[test]
    fn find_start_code_skips() {
        assert_eq!(find_start_code(&[5, 5, 5, 0, 0, 1], 0), Some(3));
        assert_eq!(find_start_code(&[0, 0, 1, 0, 0, 1], 1), Some(3));
        assert_eq!(find_start_code(&[0, 1, 0, 2, 0, 0], 0), None);
        assert_eq!(find_start_code(&[0, 0], 0), None);
        assert!(starts_with_start_code(&[0, 0, 1, 5]));
        assert!(starts_with_start_code(&[0, 0, 0, 1, 5]));
        assert!(!starts_with_start_code(&[0, 1, 5]));
    }

    #[test]
    fn avcc_conversion() {
        let data = [0, 0, 0, 2, 0x65, 0xaa, 0, 0, 0, 1, 0x41];
        let mut out = vec![];
        avcc_to_annexb(&data, 4, &mut out).unwrap();
        assert_eq!(out, [0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x41]);
        let mut out = vec![];
        avcc_to_annexb(&[0, 2, 0x65, 0xaa, 0, 1, 0x41], 2, &mut out).unwrap();
        assert_eq!(out, [0, 0, 0, 1, 0x65, 0xaa, 0, 0, 0, 1, 0x41]);
        let mut out = vec![];
        avcc_to_annexb(&[1, 0x41], 1, &mut out).unwrap();
        assert_eq!(out, [0, 0, 0, 1, 0x41]);
        assert_eq!(avcc_to_annexb(&[0, 0, 0, 5, 1], 4, &mut vec![]), Err(Error::Truncated));
        assert_eq!(avcc_to_annexb(&[0, 0], 4, &mut vec![]), Err(Error::Truncated));
        assert!(avcc_to_annexb(&[], 3, &mut vec![]).is_err());
    }

    #[test]
    fn avcc_record() {
        let sps = hex(H264_720P_SPS);
        let pps = hex(H264_720P_PPS);
        let mut rec = vec![1, 0x64, 0x00, 0x1f, 0xff, 0xe1];
        rec.extend((sps.len() as u16).to_be_bytes());
        rec.extend(&sps);
        rec.push(1);
        rec.extend((pps.len() as u16).to_be_bytes());
        rec.extend(&pps);
        let c = AvcConfig::parse(&rec).unwrap();
        assert_eq!((c.profile, c.level, c.length_size), (100, 31, 4));
        let (annexb, ls) = avcc_config_to_annexb(&rec).unwrap();
        assert_eq!(ls, 4);
        let nals: Vec<&[u8]> = nal_units(&annexb).collect();
        assert_eq!(nals, vec![&sps[..], &pps[..]]);
        assert_eq!(AvcConfig::parse(&rec[..rec.len() - 1]), Err(Error::Truncated));
        assert!(AvcConfig::parse(&[0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn hvcc_record() {
        let vps = hex(H265_720P_VPS);
        let sps = hex(H265_720P_SPS);
        let pps = hex(H265_720P_PPS);
        let mut rec = vec![1, 0x01, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 0x5d];
        rec.extend([0xf0, 0x00, 0xfc, 0xfd, 0xf8, 0xf8, 0x00, 0x00, 0x0f]);
        // Arrays deliberately out of order: PPS, VPS, SPS.
        rec.push(3);
        for (t, nal) in [(34u8, &pps), (32, &vps), (33, &sps)] {
            rec.push(0x80 | t);
            rec.extend(1u16.to_be_bytes());
            rec.extend((nal.len() as u16).to_be_bytes());
            rec.extend(nal.iter());
        }
        let c = HevcConfig::parse(&rec).unwrap();
        assert_eq!((c.profile, c.tier, c.level, c.length_size), (1, false, 93, 4));
        let (annexb, _) = hvcc_config_to_annexb(&rec).unwrap();
        let nals: Vec<&[u8]> = nal_units(&annexb).collect();
        assert_eq!(nals, vec![&vps[..], &sps[..], &pps[..]]);
        assert_eq!(HevcConfig::parse(&rec[..30]), Err(Error::Truncated));
    }
}
