//! JPEG frame header (SOFn) parsing (ITU-T T.81 B.2.2).

use camtensor_core::{ColorMatrix, ColorRange, ColorSpec};

use crate::{Error, Result, VideoInfo};

/// Start of image.
pub const SOI: u8 = 0xd8;
/// End of image.
pub const EOI: u8 = 0xd9;
/// Start of scan.
pub const SOS: u8 = 0xda;
/// Baseline DCT frame.
pub const SOF0: u8 = 0xc0;
/// Extended sequential DCT frame.
pub const SOF1: u8 = 0xc1;
/// Progressive DCT frame.
pub const SOF2: u8 = 0xc2;

/// One image component of a frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Component {
    /// Component identifier.
    pub id: u8,
    /// Horizontal sampling factor (1..=4).
    pub h: u8,
    /// Vertical sampling factor (1..=4).
    pub v: u8,
    /// Quantisation table selector.
    pub tq: u8,
}

/// Parsed frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// SOF marker (`0xc0` baseline, `0xc1` extended, `0xc2` progressive, ...).
    pub marker: u8,
    /// Sample precision in bits (8 or 12).
    pub precision: u8,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of components (1..=4).
    pub num_components: u8,
    /// Components; only the first `num_components` entries are valid.
    pub components: [Component; 4],
}

impl FrameHeader {
    /// Scans the markers of a JPEG image up to the first SOFn and parses it.
    ///
    /// Only SOF0, SOF1 and SOF2 (Huffman-coded DCT) are accepted; other
    /// frame types (lossless, arithmetic, hierarchical) are reported as
    /// [`Error::Unsupported`].
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::Truncated);
        }
        if data[0] != 0xff || data[1] != SOI {
            return Err(Error::Invalid("missing JPEG SOI marker"));
        }
        let mut i = 2;
        loop {
            // Markers may be preceded by any number of 0xff fill bytes.
            while i < data.len() && data[i] == 0xff && data.get(i + 1) == Some(&0xff) {
                i += 1;
            }
            if i + 2 > data.len() {
                return Err(Error::Truncated);
            }
            if data[i] != 0xff {
                return Err(Error::Invalid("expected JPEG marker"));
            }
            let marker = data[i + 1];
            i += 2;
            match marker {
                // Standalone markers without a length.
                0x01 | 0xd0..=0xd7 => continue,
                SOI | EOI => return Err(Error::Invalid("unexpected SOI/EOI before frame header")),
                SOS => return Err(Error::Invalid("SOS before frame header")),
                _ => {}
            }
            if i + 2 > data.len() {
                return Err(Error::Truncated);
            }
            let len = usize::from(u16::from_be_bytes([data[i], data[i + 1]]));
            if len < 2 {
                return Err(Error::Invalid("JPEG segment length < 2"));
            }
            let is_sof = matches!(marker, 0xc0..=0xcf) && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
            if is_sof {
                if !matches!(marker, SOF0 | SOF1 | SOF2) {
                    return Err(Error::Unsupported("JPEG frame type other than SOF0/1/2"));
                }
                let seg = data.get(i + 2..i + len).ok_or(Error::Truncated)?;
                return Self::parse_sof(marker, seg);
            }
            i += len;
        }
    }

    fn parse_sof(marker: u8, seg: &[u8]) -> Result<Self> {
        if seg.len() < 6 {
            return Err(Error::Truncated);
        }
        let precision = seg[0];
        let height = u32::from(u16::from_be_bytes([seg[1], seg[2]]));
        let width = u32::from(u16::from_be_bytes([seg[3], seg[4]]));
        let n = seg[5];
        if width == 0 {
            return Err(Error::Invalid("JPEG width is zero"));
        }
        if height == 0 {
            return Err(Error::Unsupported("JPEG height defined by DNL"));
        }
        if n == 0 || n > 4 {
            return Err(Error::Unsupported("JPEG component count not in 1..=4"));
        }
        let comps = seg.get(6..6 + 3 * usize::from(n)).ok_or(Error::Truncated)?;
        let mut components = [Component::default(); 4];
        for (c, b) in components.iter_mut().zip(comps.as_chunks::<3>().0) {
            *c = Component { id: b[0], h: b[1] >> 4, v: b[1] & 0x0f, tq: b[2] };
            if !(1..=4).contains(&c.h) || !(1..=4).contains(&c.v) {
                return Err(Error::Invalid("JPEG sampling factor not in 1..=4"));
            }
        }
        Ok(Self { marker, precision, width, height, num_components: n, components })
    }

    /// Valid components.
    pub fn components(&self) -> &[Component] {
        &self.components[..usize::from(self.num_components)]
    }

    /// `true` for progressive (SOF2) images.
    pub fn is_progressive(&self) -> bool {
        self.marker == SOF2
    }

    /// Chroma format code as in H.26x: 0 = grey, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4; `None` for other layouts (4:1:1, 4:4:0, CMYK, ...).
    pub fn chroma_format(&self) -> Option<u8> {
        match self.components() {
            [_] => Some(0),
            [y, cb, cr] if cb.h == cr.h && cb.v == cr.v => {
                match (y.h / cb.h, y.v / cb.v, y.h % cb.h, y.v % cb.v) {
                    (2, 2, 0, 0) => Some(1),
                    (2, 1, 0, 0) => Some(2),
                    (1, 1, 0, 0) => Some(3),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Codec-independent summary. JPEG/JFIF is full-range BT.601; layouts
    /// without an H.26x chroma format code report `chroma_format = 255`.
    pub fn info(&self) -> VideoInfo {
        VideoInfo {
            width: self.width,
            height: self.height,
            bit_depth: self.precision,
            chroma_format: self.chroma_format().unwrap_or(u8::MAX),
            color: ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 },
            fps: None,
            profile: 0,
            level: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::*;

    #[test]
    fn ffmpeg_420() {
        let h = FrameHeader::parse(&hex(JPEG_420_HDR)).unwrap();
        assert_eq!((h.marker, h.precision, h.width, h.height), (SOF0, 8, 640, 480));
        assert_eq!(h.num_components, 3);
        assert_eq!(h.components()[0], Component { id: 1, h: 2, v: 2, tq: 0 });
        assert_eq!(h.components()[1], Component { id: 2, h: 1, v: 1, tq: 0 });
        assert_eq!(h.chroma_format(), Some(1));
        assert!(!h.is_progressive());
        let info = h.info();
        assert_eq!(info.color, ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 });
    }

    #[test]
    fn ffmpeg_422_odd_size() {
        let h = FrameHeader::parse(&hex(JPEG_422_HDR)).unwrap();
        assert_eq!((h.width, h.height), (322, 242));
        assert_eq!((h.components()[0].h, h.components()[0].v), (2, 2));
        assert_eq!((h.components()[1].h, h.components()[1].v), (1, 2));
        assert_eq!(h.chroma_format(), Some(2));
    }

    fn jpeg(sof: u8, comps: &[[u8; 3]]) -> Vec<u8> {
        let mut d = vec![0xff, SOI, 0xff, 0xe0, 0x00, 0x04, 0xaa, 0xbb, 0xff, 0xff, 0xff, sof];
        let len = 8 + 3 * comps.len() as u16;
        d.extend(len.to_be_bytes());
        d.extend([8, 0x00, 0x10, 0x00, 0x20, comps.len() as u8]);
        for c in comps {
            d.extend(c);
        }
        d.extend([0xff, SOS]);
        d
    }

    #[test]
    fn synthetic_variants() {
        let grey = FrameHeader::parse(&jpeg(SOF0, &[[1, 0x11, 0]])).unwrap();
        assert_eq!((grey.width, grey.height, grey.chroma_format()), (32, 16, Some(0)));
        let prog =
            FrameHeader::parse(&jpeg(SOF2, &[[1, 0x11, 0], [2, 0x11, 1], [3, 0x11, 1]])).unwrap();
        assert!(prog.is_progressive());
        assert_eq!(prog.chroma_format(), Some(3));
        let y411 =
            FrameHeader::parse(&jpeg(SOF1, &[[1, 0x41, 0], [2, 0x11, 1], [3, 0x11, 1]])).unwrap();
        assert_eq!(y411.chroma_format(), None);
        assert_eq!(y411.info().chroma_format, 255);
    }

    #[test]
    fn errors() {
        assert_eq!(FrameHeader::parse(&[0xff]), Err(Error::Truncated));
        assert!(FrameHeader::parse(&[0x00, 0xd8]).is_err());
        assert!(matches!(
            FrameHeader::parse(&jpeg(0xc3, &[[1, 0x11, 0]])),
            Err(Error::Unsupported(_))
        ));
        assert!(FrameHeader::parse(&jpeg(SOF0, &[[1, 0x00, 0]])).is_err());
        let full = hex(JPEG_420_HDR);
        assert_eq!(FrameHeader::parse(&full[..full.len() - 4]), Err(Error::Truncated));
        // SOS without SOF.
        assert!(FrameHeader::parse(&[0xff, SOI, 0xff, SOS, 0, 2]).is_err());
    }
}
