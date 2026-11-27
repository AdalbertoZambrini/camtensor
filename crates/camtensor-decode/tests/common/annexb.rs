//! Minimal test-only splitters: Annex B byte stream → access units, and
//! concatenated JPEGs → images. (The real bitstream crate lives elsewhere;
//! this is just enough to drive the decoders in tests and benchmarks.)

#![allow(dead_code)]

/// Video coding standard of an Annex B stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standard {
    H264,
    H265,
}

/// One access unit (one picture) with its NAL units, start codes included.
#[derive(Debug, Clone)]
pub struct AccessUnit {
    pub data: Vec<u8>,
    pub keyframe: bool,
}

/// Offsets of the start-code beginnings and NAL payload starts.
fn nal_positions(b: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= b.len() {
        if b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1 {
            let begin = if i > 0 && b[i - 1] == 0 { i - 1 } else { i };
            out.push((begin, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// What a NAL unit means for access-unit delimitation.
enum Nal {
    /// AUD / parameter set / prefix SEI: starts a new AU if the current one
    /// already has a picture.
    Prefix,
    /// Slice; `first` = first slice of a picture.
    Slice { first: bool, key: bool },
    /// Anything else (suffix SEI, end of sequence, filler...).
    Other,
}

fn classify(std: Standard, nal: &[u8]) -> Nal {
    match std {
        Standard::H264 => {
            let Some(&h) = nal.first() else { return Nal::Other };
            match h & 0x1f {
                // first_mb_in_slice is ue(v): first bit 1 <=> value 0.
                t @ (1 | 5) => {
                    Nal::Slice { first: nal.get(1).is_some_and(|b| b & 0x80 != 0), key: t == 5 }
                }
                6..=9 | 14 | 15 => Nal::Prefix,
                _ => Nal::Other,
            }
        }
        Standard::H265 => {
            if nal.len() < 3 {
                return Nal::Other;
            }
            match (nal[0] >> 1) & 0x3f {
                t @ 0..=31 => Nal::Slice { first: nal[2] & 0x80 != 0, key: (16..=23).contains(&t) },
                32..=35 | 39 => Nal::Prefix,
                _ => Nal::Other,
            }
        }
    }
}

/// Splits an Annex B stream into access units.
pub fn split_access_units(b: &[u8], std: Standard) -> Vec<AccessUnit> {
    let pos = nal_positions(b);
    let mut aus = Vec::new();
    let mut cur_start: Option<usize> = None;
    let mut cur_has_slice = false;
    let mut cur_key = false;
    for (k, &(begin, payload)) in pos.iter().enumerate() {
        let end = pos.get(k + 1).map_or(b.len(), |p| p.0);
        let starts_new = match classify(std, &b[payload..end]) {
            Nal::Prefix => cur_has_slice,
            Nal::Slice { first, .. } => first && cur_has_slice,
            Nal::Other => false,
        };
        if starts_new {
            if let Some(s) = cur_start {
                aus.push(AccessUnit { data: b[s..begin].to_vec(), keyframe: cur_key });
            }
            cur_start = Some(begin);
            cur_has_slice = false;
            cur_key = false;
        }
        cur_start.get_or_insert(begin);
        if let Nal::Slice { key, .. } = classify(std, &b[payload..end]) {
            cur_has_slice = true;
            cur_key |= key;
        }
    }
    if let (Some(s), true) = (cur_start, cur_has_slice) {
        aus.push(AccessUnit { data: b[s..].to_vec(), keyframe: cur_key });
    }
    aus
}

/// Splits concatenated JPEG images (as written by `ffmpeg -f mjpeg`).
pub fn split_jpegs(b: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= b.len() {
        if !(b[i] == 0xff && b[i + 1] == 0xd8) {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 2;
        let mut end = None;
        // Marker segments until SOS, then entropy-coded data until EOI.
        while j + 2 <= b.len() {
            if b[j] != 0xff {
                j += 1;
                continue;
            }
            let m = b[j + 1];
            match m {
                0xd9 => {
                    end = Some(j + 2);
                    break;
                }
                0x00 | 0xff | 0xd0..=0xd7 | 0x01 => j += if m == 0xff { 1 } else { 2 },
                _ if j + 4 > b.len() => break,
                _ => {
                    let len = u16::from_be_bytes([b[j + 2], b[j + 3]]) as usize;
                    j += 2 + len;
                }
            }
        }
        match end {
            Some(e) => {
                out.push(b[start..e].to_vec());
                i = e;
            }
            None => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_h264_on_aud_sps_and_first_slice() {
        let sps = [0, 0, 0, 1, 0x67, 1, 2];
        let pps = [0, 0, 0, 1, 0x68, 3];
        let idr = [0, 0, 1, 0x65, 0x88, 9];
        let p1 = [0, 0, 1, 0x41, 0x9a, 1];
        let p1b = [0, 0, 1, 0x41, 0x40, 1]; // second slice of the same picture
        let p2 = [0, 0, 1, 0x41, 0x9a, 2];
        let s: Vec<u8> = [&sps[..], &pps, &idr, &p1, &p1b, &p2].concat();
        let aus = split_access_units(&s, Standard::H264);
        assert_eq!(aus.len(), 3);
        assert!(aus[0].keyframe && !aus[1].keyframe);
        assert_eq!(aus[1].data.len(), p1.len() + p1b.len());
    }
}
