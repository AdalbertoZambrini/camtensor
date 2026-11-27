//! H.264 / AVC: NAL unit types, picture types and SPS parsing
//! (ITU-T H.264 7.3.2.1.1 and Annex E).

use camtensor_core::{ColorMatrix, ColorRange, ColorSpec};

use crate::bits::{BitReader, remove_emulation_prevention};
use crate::{Error, Result, VideoInfo, annexb};

/// Coded slice of a non-IDR picture.
pub const NAL_SLICE: u8 = 1;
/// Coded slice data partition A.
pub const NAL_SLICE_DPA: u8 = 2;
/// Coded slice of an IDR picture.
pub const NAL_IDR: u8 = 5;
/// Supplemental enhancement information.
pub const NAL_SEI: u8 = 6;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 7;
/// Picture parameter set.
pub const NAL_PPS: u8 = 8;
/// Access unit delimiter.
pub const NAL_AUD: u8 = 9;
/// End of sequence.
pub const NAL_END_OF_SEQ: u8 = 10;
/// End of stream.
pub const NAL_END_OF_STREAM: u8 = 11;
/// Filler data.
pub const NAL_FILLER: u8 = 12;

/// `nal_unit_type` of a NAL unit (header byte first), `None` if empty.
pub fn nal_unit_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| b & 0x1f)
}

/// `true` if the NAL unit type carries slice data.
pub fn is_slice(nal_type: u8) -> bool {
    matches!(nal_type, NAL_SLICE..=NAL_IDR)
}

/// `true` if the Annex B access unit contains an IDR slice.
pub fn is_keyframe(annexb: &[u8]) -> bool {
    annexb::nal_units(annexb).any(|n| nal_unit_type(n) == Some(NAL_IDR))
}

/// Slice type (`slice_type % 5`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceType {
    /// Predicted.
    P,
    /// Bi-predicted.
    B,
    /// Intra.
    I,
    /// Switching P.
    Sp,
    /// Switching I.
    Si,
}

impl SliceType {
    /// Maps `slice_type` (0..=9) to a slice type.
    pub fn from_raw(v: u32) -> Option<Self> {
        if v > 9 {
            return None;
        }
        Some(match v % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            _ => SliceType::Si,
        })
    }

    /// `true` for I and SI slices.
    pub fn is_intra(self) -> bool {
        matches!(self, SliceType::I | SliceType::Si)
    }
}

/// Slice type of the first slice of an Annex B access unit.
pub fn first_slice_type(annexb: &[u8]) -> Option<SliceType> {
    let nal = annexb::nal_units(annexb).find(|n| nal_unit_type(n).is_some_and(is_slice))?;
    // first_mb_in_slice and slice_type are within the first few bytes; an
    // emulation-prevention byte that early is possible but harmless to
    // remove on a short prefix.
    let head = &nal[1..nal.len().min(16)];
    let rbsp = remove_emulation_prevention(head);
    let mut r = BitReader::new(&rbsp);
    r.ue().ok()?; // first_mb_in_slice
    SliceType::from_raw(r.ue().ok()?)
}

/// Video usability information subset (Annex E) relevant to decoding and
/// colour conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vui {
    /// Sample aspect ratio `(w, h)` if signalled.
    pub sar: Option<(u16, u16)>,
    /// `video_full_range_flag` if `video_signal_type_present_flag`.
    pub full_range: Option<bool>,
    /// `colour_primaries` if signalled.
    pub colour_primaries: Option<u8>,
    /// `transfer_characteristics` if signalled.
    pub transfer_characteristics: Option<u8>,
    /// `matrix_coefficients` if signalled.
    pub matrix_coefficients: Option<u8>,
    /// `(num_units_in_tick, time_scale)` if `timing_info_present_flag`.
    pub timing: Option<(u32, u32)>,
    /// `fixed_frame_rate_flag`.
    pub fixed_frame_rate: bool,
}

/// Parsed sequence parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sps {
    /// `profile_idc`.
    pub profile_idc: u8,
    /// `constraint_set0_flag`..`constraint_set5_flag` + reserved bits.
    pub constraint_flags: u8,
    /// `level_idc`.
    pub level_idc: u8,
    /// `seq_parameter_set_id`.
    pub sps_id: u32,
    /// `chroma_format_idc` (0..=3).
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`.
    pub separate_colour_plane: bool,
    /// Luma bit depth.
    pub bit_depth_luma: u8,
    /// Chroma bit depth.
    pub bit_depth_chroma: u8,
    /// `log2_max_frame_num_minus4 + 4`.
    pub log2_max_frame_num: u8,
    /// `pic_order_cnt_type`.
    pub pic_order_cnt_type: u8,
    /// `max_num_ref_frames`.
    pub max_num_ref_frames: u32,
    /// `frame_mbs_only_flag` (`false` = interlaced coding possible).
    pub frame_mbs_only: bool,
    /// Coded width in luma samples (macroblock aligned).
    pub coded_width: u32,
    /// Coded frame height in luma samples (macroblock aligned).
    pub coded_height: u32,
    /// Cropping in luma samples: left, right, top, bottom.
    pub crop: [u32; 4],
    /// VUI, if present and parseable.
    pub vui: Option<Vui>,
}

impl Sps {
    /// Parses an SPS NAL unit (header byte included, emulation prevention
    /// still present).
    pub fn parse(nal: &[u8]) -> Result<Self> {
        match nal_unit_type(nal) {
            Some(NAL_SPS) => {}
            Some(_) => return Err(Error::Invalid("not an H.264 SPS NAL unit")),
            None => return Err(Error::Truncated),
        }
        let rbsp = remove_emulation_prevention(&nal[1..]);
        let mut r = BitReader::new(&rbsp);
        let profile_idc = r.u(8)? as u8;
        let constraint_flags = r.u(8)? as u8;
        let level_idc = r.u(8)? as u8;
        let sps_id = r.ue_max(31, "seq_parameter_set_id")?;

        let mut chroma_format_idc = 1u8;
        let mut separate_colour_plane = false;
        let mut bit_depth_luma = 8u8;
        let mut bit_depth_chroma = 8u8;
        if matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            chroma_format_idc = r.ue_max(3, "chroma_format_idc")? as u8;
            if chroma_format_idc == 3 {
                separate_colour_plane = r.flag()?;
            }
            bit_depth_luma = 8 + r.ue_max(6, "bit_depth_luma_minus8")? as u8;
            bit_depth_chroma = 8 + r.ue_max(6, "bit_depth_chroma_minus8")? as u8;
            r.skip(1)?; // qpprime_y_zero_transform_bypass_flag
            if r.flag()? {
                // seq_scaling_matrix_present_flag
                let lists = if chroma_format_idc == 3 { 12 } else { 8 };
                for i in 0..lists {
                    if r.flag()? {
                        skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?;
                    }
                }
            }
        }
        let log2_max_frame_num = 4 + r.ue_max(12, "log2_max_frame_num_minus4")? as u8;
        let pic_order_cnt_type = r.ue_max(2, "pic_order_cnt_type")? as u8;
        match pic_order_cnt_type {
            0 => {
                r.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")?;
            }
            1 => {
                r.skip(1)?; // delta_pic_order_always_zero_flag
                r.se()?; // offset_for_non_ref_pic
                r.se()?; // offset_for_top_to_bottom_field
                let n = r.ue_max(255, "num_ref_frames_in_pic_order_cnt_cycle")?;
                for _ in 0..n {
                    r.se()?;
                }
            }
            _ => {}
        }
        let max_num_ref_frames = r.ue()?;
        r.skip(1)?; // gaps_in_frame_num_value_allowed_flag
        let width_mbs = r.ue_max(1023, "pic_width_in_mbs_minus1")? + 1;
        let height_map_units = r.ue_max(1023, "pic_height_in_map_units_minus1")? + 1;
        let frame_mbs_only = r.flag()?;
        if !frame_mbs_only {
            r.skip(1)?; // mb_adaptive_frame_field_flag
        }
        r.skip(1)?; // direct_8x8_inference_flag
        let coded_width = width_mbs * 16;
        let coded_height = height_map_units * 16 * if frame_mbs_only { 1 } else { 2 };

        let mut crop = [0u32; 4];
        if r.flag()? {
            // frame_cropping_flag
            let chroma_array_type = if separate_colour_plane { 0 } else { chroma_format_idc };
            let field_factor = if frame_mbs_only { 1 } else { 2 };
            let (unit_x, unit_y) = match chroma_array_type {
                0 => (1, field_factor),
                1 => (2, 2 * field_factor),
                2 => (2, field_factor),
                _ => (1, field_factor),
            };
            for (i, c) in crop.iter_mut().enumerate() {
                *c = r.ue()?.saturating_mul(if i < 2 { unit_x } else { unit_y });
            }
            if crop[0] + crop[1] >= coded_width || crop[2] + crop[3] >= coded_height {
                return Err(Error::Invalid("frame cropping exceeds picture size"));
            }
        }

        let vui = if r.flag()? { parse_vui(&mut r) } else { None };

        Ok(Self {
            profile_idc,
            constraint_flags,
            level_idc,
            sps_id,
            chroma_format_idc,
            separate_colour_plane,
            bit_depth_luma,
            bit_depth_chroma,
            log2_max_frame_num,
            pic_order_cnt_type,
            max_num_ref_frames,
            frame_mbs_only,
            coded_width,
            coded_height,
            crop,
            vui,
        })
    }

    /// Display width (after cropping).
    pub fn width(&self) -> u32 {
        self.coded_width - self.crop[0] - self.crop[1]
    }

    /// Display height (after cropping).
    pub fn height(&self) -> u32 {
        self.coded_height - self.crop[2] - self.crop[3]
    }

    /// Frame rate from VUI timing info (`time_scale / (2 * num_units_in_tick)`).
    pub fn fps(&self) -> Option<f64> {
        let (units, scale) = self.vui?.timing?;
        (units > 0 && scale > 0).then(|| f64::from(scale) / (2.0 * f64::from(units)))
    }

    /// Colour description, defaulting to limited-range BT.709.
    pub fn color(&self) -> ColorSpec {
        let vui = self.vui.unwrap_or_default();
        color_spec(vui.full_range, vui.matrix_coefficients)
    }

    /// Codec-independent summary.
    pub fn info(&self) -> VideoInfo {
        VideoInfo {
            width: self.width(),
            height: self.height(),
            bit_depth: self.bit_depth_luma,
            chroma_format: self.chroma_format_idc,
            color: self.color(),
            fps: self.fps(),
            profile: self.profile_idc,
            level: self.level_idc,
        }
    }
}

/// Maps VUI `video_full_range_flag` / `matrix_coefficients` (H.273 code
/// points, shared by H.264 and H.265) to a [`ColorSpec`]. Unsignalled or
/// unsupported matrices default to BT.709.
pub(crate) fn color_spec(full_range: Option<bool>, matrix: Option<u8>) -> ColorSpec {
    let range = if full_range == Some(true) { ColorRange::Full } else { ColorRange::Limited };
    let matrix = match matrix {
        // 4 = FCC, 5 = BT.470BG, 6 = SMPTE 170M: all BT.601 coefficients
        // (FCC is close enough).
        Some(4..=6) => ColorMatrix::Bt601,
        // 1 = BT.709, 2 = unspecified; others (BT.2020, SMPTE 240M, ...)
        // are not representable yet and fall back to BT.709.
        _ => ColorMatrix::Bt709,
    };
    ColorSpec { range, matrix }
}

fn skip_scaling_list(r: &mut BitReader<'_>, size: usize) -> Result<()> {
    let mut last = 8i32;
    let mut next = 8i32;
    for _ in 0..size {
        if next != 0 {
            let delta = r.se()?;
            if !(-128..=127).contains(&delta) {
                return Err(Error::Invalid("delta_scale"));
            }
            next = (last + delta + 256) % 256;
        }
        if next != 0 {
            last = next;
        }
    }
    Ok(())
}

/// Parses as much of the VUI as possible; a truncated or odd VUI keeps the
/// fields read so far (cameras are known to write broken VUIs).
fn parse_vui(r: &mut BitReader<'_>) -> Option<Vui> {
    let mut vui = Vui::default();
    let _ = parse_vui_into(r, &mut vui);
    Some(vui)
}

fn parse_vui_into(r: &mut BitReader<'_>, vui: &mut Vui) -> Result<()> {
    if r.flag()? {
        // aspect_ratio_info_present_flag
        let idc = r.u(8)? as u8;
        vui.sar =
            if idc == 255 { Some((r.u(16)? as u16, r.u(16)? as u16)) } else { sar_from_idc(idc) };
    }
    if r.flag()? {
        // overscan_info_present_flag
        r.skip(1)?;
    }
    if r.flag()? {
        // video_signal_type_present_flag
        r.skip(3)?; // video_format
        vui.full_range = Some(r.flag()?);
        if r.flag()? {
            vui.colour_primaries = Some(r.u(8)? as u8);
            vui.transfer_characteristics = Some(r.u(8)? as u8);
            vui.matrix_coefficients = Some(r.u(8)? as u8);
        }
    }
    if r.flag()? {
        // chroma_loc_info_present_flag
        r.ue()?;
        r.ue()?;
    }
    if r.flag()? {
        // timing_info_present_flag
        let units = r.u(32)?;
        let scale = r.u(32)?;
        vui.timing = Some((units, scale));
        vui.fixed_frame_rate = r.flag()?;
    }
    Ok(())
}

/// Table E-1 sample aspect ratios (shared with H.265).
pub(crate) fn sar_from_idc(idc: u8) -> Option<(u16, u16)> {
    const SAR: [(u16, u16); 17] = [
        (0, 0),
        (1, 1),
        (12, 11),
        (10, 11),
        (16, 11),
        (40, 33),
        (24, 11),
        (20, 11),
        (32, 11),
        (80, 33),
        (18, 11),
        (15, 11),
        (64, 33),
        (160, 99),
        (4, 3),
        (3, 2),
        (2, 1),
    ];
    SAR.get(usize::from(idc)).copied().filter(|&(w, _)| w != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::*;

    #[test]
    fn nal_types() {
        assert_eq!(nal_unit_type(&[0x67]), Some(NAL_SPS));
        assert_eq!(nal_unit_type(&[0x65]), Some(NAL_IDR));
        assert_eq!(nal_unit_type(&[0x41]), Some(NAL_SLICE));
        assert_eq!(nal_unit_type(&[]), None);
        assert!(is_slice(NAL_IDR) && is_slice(NAL_SLICE) && !is_slice(NAL_SEI));
        let idr = [0, 0, 0, 1, 0x67, 1, 0, 0, 1, 0x68, 2, 0, 0, 1, 0x65, 0x88];
        let p = [0, 0, 0, 1, 0x09, 0x10, 0, 0, 1, 0x41, 0x9a];
        assert!(is_keyframe(&idr));
        assert!(!is_keyframe(&p));
    }

    #[test]
    fn slice_types() {
        // IDR: first_mb_in_slice = 0 ('1'), slice_type = 7 ('0001000') -> I
        assert_eq!(first_slice_type(&[0, 0, 1, 0x65, 0b1000_1000, 0]), Some(SliceType::I));
        // P: first_mb = 0, slice_type = 5 ('00110') -> P
        assert_eq!(first_slice_type(&[0, 0, 1, 0x41, 0b1001_1000]), Some(SliceType::P));
        // B: slice_type = 1 ('010')
        assert_eq!(first_slice_type(&[0, 0, 1, 0x01, 0b1010_0000]), Some(SliceType::B));
        // SPS before the slice is skipped.
        assert_eq!(
            first_slice_type(&[0, 0, 1, 0x67, 0xff, 0, 0, 1, 0x41, 0b1001_1000]),
            Some(SliceType::P)
        );
        assert_eq!(first_slice_type(&[0, 0, 1, 0x67, 0xff]), None);
        assert!(SliceType::I.is_intra() && !SliceType::P.is_intra());
        assert_eq!(SliceType::from_raw(10), None);
    }

    #[test]
    fn sps_720p_high_full_range() {
        let s = Sps::parse(&hex(H264_720P_SPS)).unwrap();
        assert_eq!((s.profile_idc, s.level_idc, s.chroma_format_idc), (100, 31, 1));
        assert_eq!((s.coded_width, s.coded_height), (1280, 720));
        assert_eq!((s.width(), s.height()), (1280, 720));
        assert_eq!((s.bit_depth_luma, s.bit_depth_chroma), (8, 8));
        assert!(s.frame_mbs_only);
        let vui = s.vui.unwrap();
        assert_eq!(vui.full_range, Some(true));
        assert_eq!(vui.matrix_coefficients, Some(1));
        assert_eq!(vui.colour_primaries, Some(1));
        assert_eq!(s.fps(), Some(25.0));
        assert_eq!(s.color(), ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt709 });
    }

    #[test]
    fn sps_cropped_422_10bit() {
        let s = Sps::parse(&hex(H264_422_10_SPS)).unwrap();
        assert_eq!(s.profile_idc, 122);
        assert_eq!(s.chroma_format_idc, 2);
        assert_eq!((s.bit_depth_luma, s.bit_depth_chroma), (10, 10));
        assert_eq!((s.coded_width, s.coded_height), (1920, 1088));
        assert_eq!((s.width(), s.height()), (1918, 1078));
        assert_eq!(s.fps(), Some(30.0));
        assert_eq!(s.color(), ColorSpec { range: ColorRange::Limited, matrix: ColorMatrix::Bt601 });
    }

    #[test]
    fn sps_interlaced() {
        let s = Sps::parse(&hex(H264_1080I_SPS)).unwrap();
        assert!(!s.frame_mbs_only);
        assert_eq!((s.coded_width, s.coded_height), (1920, 1088));
        assert_eq!((s.width(), s.height()), (1920, 1080));
        assert_eq!(s.color(), ColorSpec::default());
    }

    #[test]
    fn sps_baseline() {
        let s = Sps::parse(&hex(H264_BASELINE_SPS)).unwrap();
        assert_eq!(s.profile_idc, 66);
        assert_eq!(s.chroma_format_idc, 1);
        assert_eq!((s.width(), s.height()), (640, 360));
        assert_eq!(s.fps(), Some(15.0));
        let info = s.info();
        assert_eq!((info.width, info.height, info.bit_depth), (640, 360, 8));
    }

    #[test]
    fn sps_errors() {
        let sps = hex(H264_720P_SPS);
        assert_eq!(Sps::parse(&sps[..6]), Err(Error::Truncated));
        assert!(Sps::parse(&hex(H264_720P_PPS)).is_err());
        assert_eq!(Sps::parse(&[]), Err(Error::Truncated));
        // VUI truncated: base fields survive.
        let s = Sps::parse(&sps[..14]).unwrap();
        assert_eq!((s.width(), s.height()), (1280, 720));
        assert_eq!(s.fps(), None);
    }

    #[test]
    fn sar_table() {
        assert_eq!(sar_from_idc(1), Some((1, 1)));
        assert_eq!(sar_from_idc(16), Some((2, 1)));
        assert_eq!(sar_from_idc(0), None);
        assert_eq!(sar_from_idc(200), None);
    }
}
