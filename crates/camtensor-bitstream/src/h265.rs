//! H.265 / HEVC: NAL unit types, IRAP detection and SPS parsing
//! (ITU-T H.265 7.3.2.2 and Annex E).

use camtensor_core::ColorSpec;

use crate::bits::{BitReader, remove_emulation_prevention};
use crate::h264::{color_spec, sar_from_idc};
use crate::{Error, Result, VideoInfo, annexb};

/// Trailing picture, non-reference.
pub const NAL_TRAIL_N: u8 = 0;
/// Trailing picture, reference.
pub const NAL_TRAIL_R: u8 = 1;
/// Random access skipped leading picture, reference.
pub const NAL_RASL_R: u8 = 9;
/// Broken link access (first IRAP type).
pub const NAL_BLA_W_LP: u8 = 16;
/// IDR with possible RADL pictures.
pub const NAL_IDR_W_RADL: u8 = 19;
/// IDR without leading pictures.
pub const NAL_IDR_N_LP: u8 = 20;
/// Clean random access.
pub const NAL_CRA: u8 = 21;
/// Last (reserved) IRAP type.
pub const NAL_IRAP_RESERVED_23: u8 = 23;
/// Video parameter set.
pub const NAL_VPS: u8 = 32;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 33;
/// Picture parameter set.
pub const NAL_PPS: u8 = 34;
/// Access unit delimiter.
pub const NAL_AUD: u8 = 35;
/// Prefix SEI.
pub const NAL_SEI_PREFIX: u8 = 39;
/// Suffix SEI.
pub const NAL_SEI_SUFFIX: u8 = 40;

/// `nal_unit_type` of a NAL unit (2-byte header first), `None` if too short.
pub fn nal_unit_type(nal: &[u8]) -> Option<u8> {
    (nal.len() >= 2).then(|| (nal[0] >> 1) & 0x3f)
}

/// `true` for VCL NAL unit types (coded slice segments, 0..=31).
pub fn is_vcl(nal_type: u8) -> bool {
    nal_type < 32
}

/// `true` for intra random access point types (16..=23).
pub fn is_irap(nal_type: u8) -> bool {
    (NAL_BLA_W_LP..=NAL_IRAP_RESERVED_23).contains(&nal_type)
}

/// `true` if the Annex B access unit contains an IRAP slice segment.
pub fn is_keyframe(annexb: &[u8]) -> bool {
    annexb::nal_units(annexb).any(|n| nal_unit_type(n).is_some_and(is_irap))
}

/// General profile / tier / level (from `profile_tier_level`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileTierLevel {
    /// `general_profile_space`.
    pub profile_space: u8,
    /// `general_tier_flag` (`true` = High tier).
    pub tier: bool,
    /// `general_profile_idc` (1 = Main, 2 = Main 10, 3 = Main Still, 4 = RExt).
    pub profile_idc: u8,
    /// `general_profile_compatibility_flag[0..32]`.
    pub compatibility_flags: u32,
    /// `general_level_idc` (30 x level number).
    pub level_idc: u8,
}

/// Video usability information subset (Annex E).
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
    /// `field_seq_flag`.
    pub field_seq: bool,
    /// `(num_units_in_tick, time_scale)` if `vui_timing_info_present_flag`.
    pub timing: Option<(u32, u32)>,
}

/// Parsed sequence parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sps {
    /// `sps_video_parameter_set_id`.
    pub vps_id: u8,
    /// `sps_max_sub_layers_minus1 + 1`.
    pub max_sub_layers: u8,
    /// General profile, tier and level.
    pub ptl: ProfileTierLevel,
    /// `sps_seq_parameter_set_id`.
    pub sps_id: u32,
    /// `chroma_format_idc` (0..=3).
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`.
    pub separate_colour_plane: bool,
    /// `pic_width_in_luma_samples`.
    pub coded_width: u32,
    /// `pic_height_in_luma_samples`.
    pub coded_height: u32,
    /// Conformance window in luma samples: left, right, top, bottom.
    pub crop: [u32; 4],
    /// Luma bit depth.
    pub bit_depth_luma: u8,
    /// Chroma bit depth.
    pub bit_depth_chroma: u8,
    /// VUI, if present (fields read before any parse problem are kept).
    pub vui: Option<Vui>,
}

impl Sps {
    /// Parses an SPS NAL unit (2-byte header included, emulation prevention
    /// still present).
    pub fn parse(nal: &[u8]) -> Result<Self> {
        match nal_unit_type(nal) {
            Some(NAL_SPS) => {}
            Some(_) => return Err(Error::Invalid("not an H.265 SPS NAL unit")),
            None => return Err(Error::Truncated),
        }
        let rbsp = remove_emulation_prevention(&nal[2..]);
        let mut r = BitReader::new(&rbsp);
        let vps_id = r.u(4)? as u8;
        let max_sub_layers_minus1 = r.u(3)? as u8;
        if max_sub_layers_minus1 > 6 {
            return Err(Error::Invalid("sps_max_sub_layers_minus1"));
        }
        r.skip(1)?; // sps_temporal_id_nesting_flag
        let ptl = parse_profile_tier_level(&mut r, max_sub_layers_minus1)?;
        let sps_id = r.ue_max(15, "sps_seq_parameter_set_id")?;
        let chroma_format_idc = r.ue_max(3, "chroma_format_idc")? as u8;
        let separate_colour_plane = chroma_format_idc == 3 && r.flag()?;
        let coded_width = r.ue_max(16888, "pic_width_in_luma_samples")?;
        let coded_height = r.ue_max(16888, "pic_height_in_luma_samples")?;
        if coded_width == 0 || coded_height == 0 {
            return Err(Error::Invalid("zero picture size"));
        }
        let mut crop = [0u32; 4];
        if r.flag()? {
            // conformance_window_flag
            let chroma_array_type = if separate_colour_plane { 0 } else { chroma_format_idc };
            let sub_w = if matches!(chroma_array_type, 1 | 2) { 2 } else { 1 };
            let sub_h = if chroma_array_type == 1 { 2 } else { 1 };
            for (i, c) in crop.iter_mut().enumerate() {
                *c = r.ue()?.saturating_mul(if i < 2 { sub_w } else { sub_h });
            }
            if crop[0] + crop[1] >= coded_width || crop[2] + crop[3] >= coded_height {
                return Err(Error::Invalid("conformance window exceeds picture size"));
            }
        }
        let bit_depth_luma = 8 + r.ue_max(8, "bit_depth_luma_minus8")? as u8;
        let bit_depth_chroma = 8 + r.ue_max(8, "bit_depth_chroma_minus8")? as u8;

        let mut sps = Self {
            vps_id,
            max_sub_layers: max_sub_layers_minus1 + 1,
            ptl,
            sps_id,
            chroma_format_idc,
            separate_colour_plane,
            coded_width,
            coded_height,
            crop,
            bit_depth_luma,
            bit_depth_chroma,
            vui: None,
        };
        // The rest only matters for the VUI. Failures past this point keep
        // the essential fields above.
        if let Ok(true) = skip_to_vui(&mut r, max_sub_layers_minus1) {
            let mut vui = Vui::default();
            let _ = parse_vui_into(&mut r, &mut vui);
            sps.vui = Some(vui);
        }
        Ok(sps)
    }

    /// Display width (after the conformance window).
    pub fn width(&self) -> u32 {
        self.coded_width - self.crop[0] - self.crop[1]
    }

    /// Display height (after the conformance window).
    pub fn height(&self) -> u32 {
        self.coded_height - self.crop[2] - self.crop[3]
    }

    /// Frame rate from VUI timing info (`time_scale / num_units_in_tick`;
    /// halved for field-coded streams).
    pub fn fps(&self) -> Option<f64> {
        let vui = self.vui?;
        let (units, scale) = vui.timing?;
        let fps = f64::from(scale) / f64::from(units);
        (units > 0 && scale > 0).then_some(if vui.field_seq { fps / 2.0 } else { fps })
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
            profile: self.ptl.profile_idc,
            level: self.ptl.level_idc,
        }
    }
}

fn parse_profile_tier_level(
    r: &mut BitReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<ProfileTierLevel> {
    let profile_space = r.u(2)? as u8;
    let tier = r.flag()?;
    let profile_idc = r.u(5)? as u8;
    let compatibility_flags = r.u(32)?;
    r.skip(4 + 43 + 1)?; // source flags, constraint flags, inbld/reserved
    let level_idc = r.u(8)? as u8;
    let n = usize::from(max_sub_layers_minus1);
    let mut present = [(false, false); 8];
    for p in present.iter_mut().take(n) {
        *p = (r.flag()?, r.flag()?);
    }
    if n > 0 {
        r.skip(2 * (8 - n))?; // reserved_zero_2bits
    }
    for &(profile, level) in present.iter().take(n) {
        if profile {
            r.skip(88)?;
        }
        if level {
            r.skip(8)?;
        }
    }
    Ok(ProfileTierLevel { profile_space, tier, profile_idc, compatibility_flags, level_idc })
}

/// Skips the SPS fields between the bit depths and `vui_parameters_present_flag`;
/// returns that flag.
fn skip_to_vui(r: &mut BitReader<'_>, max_sub_layers_minus1: u8) -> Result<bool> {
    let log2_max_poc_lsb = 4 + r.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")?;
    let ordering_info_present = r.flag()?;
    let first = if ordering_info_present { 0 } else { max_sub_layers_minus1 };
    for _ in first..=max_sub_layers_minus1 {
        r.ue()?; // sps_max_dec_pic_buffering_minus1
        r.ue()?; // sps_max_num_reorder_pics
        r.ue()?; // sps_max_latency_increase_plus1
    }
    for _ in 0..6 {
        // log2_min_luma_coding_block_size_minus3 .. max_transform_hierarchy_depth_intra
        r.ue()?;
    }
    if r.flag()? && r.flag()? {
        // scaling_list_enabled_flag && sps_scaling_list_data_present_flag
        skip_scaling_list_data(r)?;
    }
    r.skip(2)?; // amp_enabled_flag, sample_adaptive_offset_enabled_flag
    if r.flag()? {
        // pcm_enabled_flag
        r.skip(8)?;
        r.ue()?;
        r.ue()?;
        r.skip(1)?;
    }
    let num_st_rps = r.ue_max(64, "num_short_term_ref_pic_sets")? as usize;
    let mut num_delta_pocs = [0u32; 64];
    for idx in 0..num_st_rps {
        num_delta_pocs[idx] = skip_st_ref_pic_set(r, idx, &num_delta_pocs)?;
    }
    if r.flag()? {
        // long_term_ref_pics_present_flag
        let n = r.ue_max(32, "num_long_term_ref_pics_sps")?;
        for _ in 0..n {
            r.skip(log2_max_poc_lsb as usize + 1)?;
        }
    }
    r.skip(2)?; // sps_temporal_mvp_enabled_flag, strong_intra_smoothing_enabled_flag
    r.flag()
}

fn skip_scaling_list_data(r: &mut BitReader<'_>) -> Result<()> {
    for size_id in 0..4u32 {
        let step = if size_id == 3 { 3 } else { 1 };
        for _ in (0..6).step_by(step) {
            if !r.flag()? {
                r.ue()?; // scaling_list_pred_matrix_id_delta
            } else {
                let coef_num = 64.min(1 << (4 + (size_id << 1)));
                if size_id > 1 {
                    r.se()?; // scaling_list_dc_coef_minus8
                }
                for _ in 0..coef_num {
                    r.se()?;
                }
            }
        }
    }
    Ok(())
}

/// Skips `st_ref_pic_set(idx)` (as found in an SPS) and returns its
/// `NumDeltaPocs`.
fn skip_st_ref_pic_set(r: &mut BitReader<'_>, idx: usize, num_delta_pocs: &[u32]) -> Result<u32> {
    let inter_rps_pred = idx != 0 && r.flag()?;
    if inter_rps_pred {
        // In an SPS delta_idx_minus1 is not coded (inferred 0).
        r.skip(1)?; // delta_rps_sign
        r.ue()?; // abs_delta_rps_minus1
        let ref_idx = idx - 1;
        let mut count = 0u32;
        for _ in 0..=num_delta_pocs[ref_idx] {
            let used_by_curr = r.flag()?;
            let use_delta = used_by_curr || r.flag()?;
            if use_delta {
                count += 1;
            }
        }
        Ok(count)
    } else {
        let neg = r.ue_max(16, "num_negative_pics")?;
        let pos = r.ue_max(16, "num_positive_pics")?;
        for _ in 0..neg + pos {
            r.ue()?; // delta_poc_sX_minus1
            r.skip(1)?; // used_by_curr_pic_sX_flag
        }
        Ok(neg + pos)
    }
}

fn parse_vui_into(r: &mut BitReader<'_>, vui: &mut Vui) -> Result<()> {
    if r.flag()? {
        let idc = r.u(8)? as u8;
        vui.sar =
            if idc == 255 { Some((r.u(16)? as u16, r.u(16)? as u16)) } else { sar_from_idc(idc) };
    }
    if r.flag()? {
        r.skip(1)?; // overscan_appropriate_flag
    }
    if r.flag()? {
        // video_signal_type_present_flag
        r.skip(3)?;
        vui.full_range = Some(r.flag()?);
        if r.flag()? {
            vui.colour_primaries = Some(r.u(8)? as u8);
            vui.transfer_characteristics = Some(r.u(8)? as u8);
            vui.matrix_coefficients = Some(r.u(8)? as u8);
        }
    }
    if r.flag()? {
        r.ue()?;
        r.ue()?;
    }
    r.skip(1)?; // neutral_chroma_indication_flag
    vui.field_seq = r.flag()?;
    r.skip(1)?; // frame_field_info_present_flag
    if r.flag()? {
        // default_display_window_flag
        for _ in 0..4 {
            r.ue()?;
        }
    }
    if r.flag()? {
        // vui_timing_info_present_flag
        let units = r.u(32)?;
        let scale = r.u(32)?;
        vui.timing = Some((units, scale));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::*;
    use camtensor_core::{ColorMatrix, ColorRange};

    #[test]
    fn nal_types() {
        assert_eq!(nal_unit_type(&hex(H265_720P_VPS)), Some(NAL_VPS));
        assert_eq!(nal_unit_type(&hex(H265_720P_SPS)), Some(NAL_SPS));
        assert_eq!(nal_unit_type(&hex(H265_720P_PPS)), Some(NAL_PPS));
        assert_eq!(nal_unit_type(&[0x26, 0x01]), Some(NAL_IDR_W_RADL));
        assert_eq!(nal_unit_type(&[0x02, 0x01]), Some(NAL_TRAIL_R));
        assert_eq!(nal_unit_type(&[0x26]), None);
        for t in 0..64u8 {
            assert_eq!(is_irap(t), (16..=23).contains(&t), "type {t}");
        }
        assert!(is_vcl(NAL_CRA) && is_vcl(NAL_RASL_R) && !is_vcl(NAL_SEI_PREFIX));
        let idr = [0, 0, 0, 1, 0x40, 0x01, 0xaa, 0, 0, 1, 0x26, 0x01, 0xaf];
        let cra = [0, 0, 1, 0x2a, 0x01, 0xaf];
        let trail = [0, 0, 0, 1, 0x46, 0x01, 0x10, 0, 0, 1, 0x02, 0x01, 0xd0];
        assert!(is_keyframe(&idr));
        assert!(is_keyframe(&cra));
        assert!(!is_keyframe(&trail));
        assert_eq!(NAL_IDR_N_LP, 20);
        assert_eq!(NAL_TRAIL_N, 0);
        assert_eq!((NAL_AUD, NAL_SEI_SUFFIX), (35, 40));
    }

    #[test]
    fn sps_720p_main() {
        let s = Sps::parse(&hex(H265_720P_SPS)).unwrap();
        assert_eq!(s.ptl.profile_idc, 1);
        assert!(!s.ptl.tier);
        assert_eq!(s.ptl.level_idc, 93);
        assert_eq!(s.max_sub_layers, 1);
        assert_eq!(s.chroma_format_idc, 1);
        assert_eq!((s.coded_width, s.coded_height), (1280, 720));
        assert_eq!((s.width(), s.height()), (1280, 720));
        assert_eq!((s.bit_depth_luma, s.bit_depth_chroma), (8, 8));
        assert_eq!(s.fps(), Some(25.0));
        assert_eq!(s.color(), ColorSpec::default());
    }

    #[test]
    fn sps_main10_cropped_full_range() {
        let s = Sps::parse(&hex(H265_MAIN10_SPS)).unwrap();
        assert_eq!(s.ptl.profile_idc, 2);
        assert_eq!((s.bit_depth_luma, s.bit_depth_chroma), (10, 10));
        assert_eq!((s.width(), s.height()), (1918, 1078));
        assert!(s.coded_width >= 1918 && s.coded_height >= 1078);
        assert_eq!(s.fps(), Some(30.0));
        let vui = s.vui.unwrap();
        assert_eq!(vui.full_range, Some(true));
        assert_eq!(vui.matrix_coefficients, Some(5));
        assert_eq!(s.color(), ColorSpec { range: ColorRange::Full, matrix: ColorMatrix::Bt601 });
        let info = s.info();
        assert_eq!((info.profile, info.bit_depth, info.chroma_format), (2, 10, 1));
    }

    #[test]
    fn sub_layers_ptl() {
        // Hand-built PTL with 2 sub-layers: sub-layer 0 has profile and level.
        // general: 2+1+5 + 32 + 48 + 8 = 96 bits; flags: 2 bits; reserved 2*7
        // bits; sub-layer 0: 88 + 8 bits.
        let mut bits = String::new();
        bits.push_str("00"); // profile_space
        bits.push('1'); // tier
        bits.push_str("00010"); // profile 2
        bits.push_str(&"0".repeat(32 + 48));
        bits.push_str(&format!("{:08b}", 120)); // level
        bits.push_str("11"); // sub-layer 0 profile + level present
        bits.push_str(&"0".repeat(14));
        bits.push_str(&"1".repeat(88 + 8));
        bits.push_str("10110"); // trailing marker bits
        while !bits.len().is_multiple_of(8) {
            bits.push('0');
        }
        let bytes: Vec<u8> = (0..bits.len() / 8)
            .map(|i| u8::from_str_radix(&bits[i * 8..i * 8 + 8], 2).unwrap())
            .collect();
        let mut r = BitReader::new(&bytes);
        let ptl = parse_profile_tier_level(&mut r, 1).unwrap();
        assert_eq!((ptl.tier, ptl.profile_idc, ptl.level_idc), (true, 2, 120));
        assert_eq!(r.u(5).unwrap(), 0b10110);
    }

    #[test]
    fn sps_errors() {
        let sps = hex(H265_720P_SPS);
        assert_eq!(Sps::parse(&sps[..10]), Err(Error::Truncated));
        assert!(Sps::parse(&hex(H265_720P_PPS)).is_err());
        assert_eq!(Sps::parse(&[0x42]), Err(Error::Truncated));
        // Truncated after the essential fields: dimensions survive, no VUI.
        let s = Sps::parse(&sps[..24]).unwrap();
        assert_eq!((s.width(), s.height()), (1280, 720));
        assert_eq!(s.fps(), None);
    }
}
