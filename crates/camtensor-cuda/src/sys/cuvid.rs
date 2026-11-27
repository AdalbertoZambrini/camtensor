//! NVDEC API (`nvcuvid.dll` / `libnvcuvid.so.1`), subset.
//!
//! Note: the headers use `unsigned long` (`tcu_ulong`) in several structs,
//! which is 32-bit on Windows and 64-bit on Linux; [`c_ulong`] mirrors that.

use std::ffi::{c_int, c_short, c_uint, c_ulong, c_void};
use std::sync::OnceLock;

use super::driver::{CUresult, CUstream};
use super::{dynamic_api, open_library};

pub type CUvideodecoder = *mut c_void;
pub type CUvideoparser = *mut c_void;
pub type CUvideoctxlock = *mut c_void;
pub type CUvideotimestamp = i64;

// cudaVideoCodec
pub type cudaVideoCodec = c_int;
pub const cudaVideoCodec_H264: cudaVideoCodec = 4;
pub const cudaVideoCodec_JPEG: cudaVideoCodec = 5;
pub const cudaVideoCodec_HEVC: cudaVideoCodec = 8;

// cudaVideoSurfaceFormat
pub type cudaVideoSurfaceFormat = c_int;
pub const cudaVideoSurfaceFormat_NV12: cudaVideoSurfaceFormat = 0;
pub const cudaVideoSurfaceFormat_P016: cudaVideoSurfaceFormat = 1;
pub const cudaVideoSurfaceFormat_YUV444: cudaVideoSurfaceFormat = 2;

// cudaVideoDeinterlaceMode
pub type cudaVideoDeinterlaceMode = c_int;
pub const cudaVideoDeinterlaceMode_Weave: cudaVideoDeinterlaceMode = 0;
pub const cudaVideoDeinterlaceMode_Adaptive: cudaVideoDeinterlaceMode = 2;

// cudaVideoChromaFormat
pub type cudaVideoChromaFormat = c_int;
pub const cudaVideoChromaFormat_Monochrome: cudaVideoChromaFormat = 0;
pub const cudaVideoChromaFormat_420: cudaVideoChromaFormat = 1;
pub const cudaVideoChromaFormat_422: cudaVideoChromaFormat = 2;
pub const cudaVideoChromaFormat_444: cudaVideoChromaFormat = 3;

// cudaVideoCreateFlags
pub const cudaVideoCreate_PreferCUVID: c_ulong = 0x04;

// cuvidDecodeStatus
pub const cuvidDecodeStatus_Error: c_int = 8;
pub const cuvidDecodeStatus_Error_Concealed: c_int = 9;

// CUvideopacketflags
pub const CUVID_PKT_ENDOFSTREAM: c_ulong = 0x01;
pub const CUVID_PKT_TIMESTAMP: c_ulong = 0x02;
pub const CUVID_PKT_DISCONTINUITY: c_ulong = 0x04;
pub const CUVID_PKT_ENDOFPICTURE: c_ulong = 0x08;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CUVIDDECODECAPS {
    pub eCodecType: cudaVideoCodec,
    pub eChromaFormat: cudaVideoChromaFormat,
    pub nBitDepthMinus8: c_uint,
    pub reserved1: [c_uint; 3],
    pub bIsSupported: u8,
    pub nNumNVDECs: u8,
    pub nOutputFormatMask: u16,
    pub nMaxWidth: c_uint,
    pub nMaxHeight: c_uint,
    pub nMaxMBCount: c_uint,
    pub nMinWidth: u16,
    pub nMinHeight: u16,
    pub bIsHistogramSupported: u8,
    pub nCounterBitDepth: u8,
    pub nMaxHistogramBins: u16,
    pub bIsDecodeStatsSupported: u8,
    pub reserved4: [u8; 3],
    pub reserved3: [c_uint; 9],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDRECT16 {
    pub left: c_short,
    pub top: c_short,
    pub right: c_short,
    pub bottom: c_short,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDDECODECREATEINFO {
    pub ulWidth: c_ulong,
    pub ulHeight: c_ulong,
    pub ulNumDecodeSurfaces: c_ulong,
    pub CodecType: cudaVideoCodec,
    pub ChromaFormat: cudaVideoChromaFormat,
    pub ulCreationFlags: c_ulong,
    pub bitDepthMinus8: c_ulong,
    pub ulIntraDecodeOnly: c_ulong,
    pub ulMaxWidth: c_ulong,
    pub ulMaxHeight: c_ulong,
    pub Reserved1: c_ulong,
    pub display_area: CUVIDRECT16,
    pub OutputFormat: cudaVideoSurfaceFormat,
    pub DeinterlaceMode: cudaVideoDeinterlaceMode,
    pub ulTargetWidth: c_ulong,
    pub ulTargetHeight: c_ulong,
    pub ulNumOutputSurfaces: c_ulong,
    pub vidLock: CUvideoctxlock,
    pub target_rect: CUVIDRECT16,
    pub enableHistogram: c_ulong,
    pub enableDecodeFeatures: c_ulong,
    pub Reserved2: [c_ulong; 3],
}

impl Default for CUVIDDECODECREATEINFO {
    fn default() -> Self {
        Self {
            ulWidth: 0,
            ulHeight: 0,
            ulNumDecodeSurfaces: 0,
            CodecType: 0,
            ChromaFormat: 0,
            ulCreationFlags: 0,
            bitDepthMinus8: 0,
            ulIntraDecodeOnly: 0,
            ulMaxWidth: 0,
            ulMaxHeight: 0,
            Reserved1: 0,
            display_area: CUVIDRECT16::default(),
            OutputFormat: 0,
            DeinterlaceMode: 0,
            ulTargetWidth: 0,
            ulTargetHeight: 0,
            ulNumOutputSurfaces: 0,
            vidLock: std::ptr::null_mut(),
            target_rect: CUVIDRECT16::default(),
            enableHistogram: 0,
            enableDecodeFeatures: 0,
            Reserved2: [0; 3],
        }
    }
}

/// `CUVIDPICPARAMS`: only the common prefix is declared. Values of this type
/// are only ever handled through pointers produced by the parser, never
/// created or copied by value, so the codec-specific union is omitted.
#[repr(C)]
#[derive(Debug)]
pub struct CUVIDPICPARAMS {
    pub PicWidthInMbs: c_int,
    pub FrameHeightInMbs: c_int,
    pub CurrPicIdx: c_int,
    pub field_pic_flag: c_int,
    pub bottom_field_flag: c_int,
    pub second_field: c_int,
    pub nBitstreamDataLen: c_uint,
    pub pBitstreamData: *const u8,
    pub nNumSlices: c_uint,
    pub pSliceDataOffsets: *const c_uint,
    pub ref_pic_flag: c_int,
    pub intra_pic_flag: c_int,
    pub timestamp: CUvideotimestamp,
    pub Reserved: [c_uint; 28],
    // union CodecSpecific follows (opaque).
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDPROCPARAMS {
    pub progressive_frame: c_int,
    pub second_field: c_int,
    pub top_field_first: c_int,
    pub unpaired_field: c_int,
    pub reserved_flags: c_uint,
    pub reserved_zero: c_uint,
    pub raw_input_dptr: u64,
    pub raw_input_pitch: c_uint,
    pub raw_input_format: c_uint,
    pub raw_output_dptr: u64,
    pub raw_output_pitch: c_uint,
    pub Reserved1: c_uint,
    pub output_stream: CUstream,
    pub Reserved: [c_uint; 46],
    pub histogram_dptr: *mut u64,
    pub pCuvidProcExt: *mut c_void,
}

impl Default for CUVIDPROCPARAMS {
    fn default() -> Self {
        Self {
            progressive_frame: 0,
            second_field: 0,
            top_field_first: 0,
            unpaired_field: 0,
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            Reserved1: 0,
            output_stream: std::ptr::null_mut(),
            Reserved: [0; 46],
            histogram_dptr: std::ptr::null_mut(),
            pCuvidProcExt: std::ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDGETDECODESTATUS {
    pub decodeStatus: c_int,
    pub reserved: [c_uint; 31],
    pub pReserved: [*mut c_void; 8],
}

impl Default for CUVIDGETDECODESTATUS {
    fn default() -> Self {
        Self { decodeStatus: 0, reserved: [0; 31], pReserved: [std::ptr::null_mut(); 8] }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDFRAMERATE {
    pub numerator: c_uint,
    pub denominator: c_uint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDRECT32 {
    pub left: c_int,
    pub top: c_int,
    pub right: c_int,
    pub bottom: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDASPECT {
    pub x: c_int,
    pub y: c_int,
}

/// `video_signal_description` of `CUVIDEOFORMAT`. The first byte holds the
/// bit fields `video_format:3`, `video_full_range_flag:1`, `reserved:4`
/// (allocated LSB first by both MSVC and GCC/Clang on x86_64).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDSIGNALDESC {
    pub bits: u8,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

impl CUVIDSIGNALDESC {
    /// `video_format` (0 = component, 5 = unspecified).
    pub fn video_format(&self) -> u8 {
        self.bits & 0x7
    }
    /// `video_full_range_flag`.
    pub fn full_range(&self) -> bool {
        self.bits & 0x8 != 0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDEOFORMAT {
    pub codec: cudaVideoCodec,
    pub frame_rate: CUVIDFRAMERATE,
    pub progressive_sequence: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub min_num_decode_surfaces: u8,
    pub coded_width: c_uint,
    pub coded_height: c_uint,
    pub display_area: CUVIDRECT32,
    pub chroma_format: cudaVideoChromaFormat,
    pub bitrate: c_uint,
    pub display_aspect_ratio: CUVIDASPECT,
    pub video_signal_description: CUVIDSIGNALDESC,
    pub seqhdr_data_length: c_uint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUVIDPARSERDISPINFO {
    pub picture_index: c_int,
    pub progressive_frame: c_int,
    pub top_field_first: c_int,
    pub repeat_first_field: c_int,
    pub timestamp: CUvideotimestamp,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDSOURCEDATAPACKET {
    pub flags: c_ulong,
    pub payload_size: c_ulong,
    pub payload: *const u8,
    pub timestamp: CUvideotimestamp,
}

pub type PFNVIDSEQUENCECALLBACK =
    Option<unsafe extern "system" fn(*mut c_void, *mut CUVIDEOFORMAT) -> c_int>;
pub type PFNVIDDECODECALLBACK =
    Option<unsafe extern "system" fn(*mut c_void, *mut CUVIDPICPARAMS) -> c_int>;
pub type PFNVIDDISPLAYCALLBACK =
    Option<unsafe extern "system" fn(*mut c_void, *mut CUVIDPARSERDISPINFO) -> c_int>;
pub type PFNVIDOPAQUECALLBACK =
    Option<unsafe extern "system" fn(*mut c_void, *mut c_void) -> c_int>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CUVIDPARSERPARAMS {
    pub CodecType: cudaVideoCodec,
    pub ulMaxNumDecodeSurfaces: c_uint,
    pub ulClockRate: c_uint,
    pub ulErrorThreshold: c_uint,
    pub ulMaxDisplayDelay: c_uint,
    /// Bit fields `bAnnexb:1`, `bMemoryOptimize:1`, `uReserved:30`.
    pub bits: c_uint,
    pub uReserved1: [c_uint; 4],
    pub pUserData: *mut c_void,
    pub pfnSequenceCallback: PFNVIDSEQUENCECALLBACK,
    pub pfnDecodePicture: PFNVIDDECODECALLBACK,
    pub pfnDisplayPicture: PFNVIDDISPLAYCALLBACK,
    pub pfnGetOperatingPoint: PFNVIDOPAQUECALLBACK,
    pub pfnGetSEIMsg: PFNVIDOPAQUECALLBACK,
    pub pvReserved2: [*mut c_void; 5],
    pub pExtVideoInfo: *mut c_void,
}

impl Default for CUVIDPARSERPARAMS {
    fn default() -> Self {
        Self {
            CodecType: 0,
            ulMaxNumDecodeSurfaces: 0,
            ulClockRate: 0,
            ulErrorThreshold: 0,
            ulMaxDisplayDelay: 0,
            bits: 0,
            uReserved1: [0; 4],
            pUserData: std::ptr::null_mut(),
            pfnSequenceCallback: None,
            pfnDecodePicture: None,
            pfnDisplayPicture: None,
            pfnGetOperatingPoint: None,
            pfnGetSEIMsg: None,
            pvReserved2: [std::ptr::null_mut(); 5],
            pExtVideoInfo: std::ptr::null_mut(),
        }
    }
}

dynamic_api! {
    /// Function table of the NVDEC API.
    pub struct CuvidApi {
        cuvidGetDecoderCaps = "cuvidGetDecoderCaps": fn(*mut CUVIDDECODECAPS) -> CUresult;
        cuvidCreateDecoder = "cuvidCreateDecoder": fn(*mut CUvideodecoder, *mut CUVIDDECODECREATEINFO) -> CUresult;
        cuvidDestroyDecoder = "cuvidDestroyDecoder": fn(CUvideodecoder) -> CUresult;
        cuvidDecodePicture = "cuvidDecodePicture": fn(CUvideodecoder, *mut CUVIDPICPARAMS) -> CUresult;
        cuvidGetDecodeStatus = "cuvidGetDecodeStatus": fn(CUvideodecoder, c_int, *mut CUVIDGETDECODESTATUS) -> CUresult;
        cuvidMapVideoFrame64 = "cuvidMapVideoFrame64": fn(CUvideodecoder, c_int, *mut u64, *mut c_uint, *mut CUVIDPROCPARAMS) -> CUresult;
        cuvidUnmapVideoFrame64 = "cuvidUnmapVideoFrame64": fn(CUvideodecoder, u64) -> CUresult;
        cuvidCreateVideoParser = "cuvidCreateVideoParser": fn(*mut CUvideoparser, *mut CUVIDPARSERPARAMS) -> CUresult;
        cuvidParseVideoData = "cuvidParseVideoData": fn(CUvideoparser, *mut CUVIDSOURCEDATAPACKET) -> CUresult;
        cuvidDestroyVideoParser = "cuvidDestroyVideoParser": fn(CUvideoparser) -> CUresult;
    }
}

#[cfg(windows)]
const CUVID_LIBS: &[&str] = &["nvcuvid.dll"];
#[cfg(not(windows))]
const CUVID_LIBS: &[&str] = &["libnvcuvid.so.1", "libnvcuvid.so"];

static CUVID: OnceLock<Result<CuvidApi, String>> = OnceLock::new();

/// Loads the NVDEC library once per process.
pub fn api() -> Result<&'static CuvidApi, &'static str> {
    CUVID
        .get_or_init(|| {
            let lib = open_library(CUVID_LIBS)?;
            // SAFETY: `lib` is nvcuvid; the declared signatures match
            // cuviddec.h / nvcuvid.h.
            unsafe { CuvidApi::load(lib) }
        })
        .as_ref()
        .map_err(|e| e.as_str())
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// `sizeof(unsigned long)`: 4 on Windows (LLP64), 8 on Linux (LP64).
    const L: usize = size_of::<c_ulong>();

    #[test]
    fn decode_caps_layout() {
        assert_eq!(size_of::<CUVIDDECODECAPS>(), 88);
        assert_eq!(offset_of!(CUVIDDECODECAPS, bIsSupported), 24);
        assert_eq!(offset_of!(CUVIDDECODECAPS, nOutputFormatMask), 26);
        assert_eq!(offset_of!(CUVIDDECODECAPS, nMaxWidth), 28);
        assert_eq!(offset_of!(CUVIDDECODECAPS, nMaxMBCount), 36);
        assert_eq!(offset_of!(CUVIDDECODECAPS, nMinWidth), 40);
        assert_eq!(offset_of!(CUVIDDECODECAPS, nMaxHistogramBins), 46);
        assert_eq!(offset_of!(CUVIDDECODECAPS, bIsDecodeStatsSupported), 48);
        assert_eq!(offset_of!(CUVIDDECODECAPS, reserved3), 52);
    }

    #[test]
    fn decode_create_info_layout() {
        type T = CUVIDDECODECREATEINFO;
        if L == 4 {
            assert_eq!(offset_of!(T, ulNumDecodeSurfaces), 8);
            assert_eq!(offset_of!(T, CodecType), 12);
            assert_eq!(offset_of!(T, ChromaFormat), 16);
            assert_eq!(offset_of!(T, ulCreationFlags), 20);
            assert_eq!(offset_of!(T, ulMaxWidth), 32);
            assert_eq!(offset_of!(T, Reserved1), 40);
            assert_eq!(offset_of!(T, display_area), 44);
            assert_eq!(offset_of!(T, OutputFormat), 52);
            assert_eq!(offset_of!(T, DeinterlaceMode), 56);
            assert_eq!(offset_of!(T, ulTargetWidth), 60);
            assert_eq!(offset_of!(T, ulNumOutputSurfaces), 68);
            assert_eq!(offset_of!(T, vidLock), 72);
            assert_eq!(offset_of!(T, target_rect), 80);
            assert_eq!(offset_of!(T, enableHistogram), 88);
            assert_eq!(offset_of!(T, Reserved2), 96);
            assert_eq!(size_of::<T>(), 112);
        } else {
            assert_eq!(offset_of!(T, ulNumDecodeSurfaces), 16);
            assert_eq!(offset_of!(T, CodecType), 24);
            assert_eq!(offset_of!(T, ChromaFormat), 28);
            assert_eq!(offset_of!(T, ulCreationFlags), 32);
            assert_eq!(offset_of!(T, ulMaxWidth), 56);
            assert_eq!(offset_of!(T, Reserved1), 72);
            assert_eq!(offset_of!(T, display_area), 80);
            assert_eq!(offset_of!(T, OutputFormat), 88);
            assert_eq!(offset_of!(T, DeinterlaceMode), 92);
            assert_eq!(offset_of!(T, ulTargetWidth), 96);
            assert_eq!(offset_of!(T, ulNumOutputSurfaces), 112);
            assert_eq!(offset_of!(T, vidLock), 120);
            assert_eq!(offset_of!(T, target_rect), 128);
            assert_eq!(offset_of!(T, enableHistogram), 136);
            assert_eq!(offset_of!(T, Reserved2), 152);
            assert_eq!(size_of::<T>(), 176);
        }
    }

    #[test]
    fn pic_params_prefix_layout() {
        type T = CUVIDPICPARAMS;
        assert_eq!(offset_of!(T, CurrPicIdx), 8);
        assert_eq!(offset_of!(T, nBitstreamDataLen), 24);
        assert_eq!(offset_of!(T, pBitstreamData), 32);
        assert_eq!(offset_of!(T, nNumSlices), 40);
        assert_eq!(offset_of!(T, pSliceDataOffsets), 48);
        assert_eq!(offset_of!(T, ref_pic_flag), 56);
        assert_eq!(offset_of!(T, intra_pic_flag), 60);
        assert_eq!(offset_of!(T, timestamp), 64);
        // The codec-specific union starts right after `Reserved`.
        assert_eq!(size_of::<T>(), 184);
    }

    #[test]
    fn proc_params_layout() {
        type T = CUVIDPROCPARAMS;
        assert_eq!(offset_of!(T, raw_input_dptr), 24);
        assert_eq!(offset_of!(T, raw_output_dptr), 40);
        assert_eq!(offset_of!(T, output_stream), 56);
        assert_eq!(offset_of!(T, Reserved), 64);
        assert_eq!(offset_of!(T, histogram_dptr), 248);
        assert_eq!(offset_of!(T, pCuvidProcExt), 256);
        assert_eq!(size_of::<T>(), 264);
    }

    #[test]
    fn decode_status_layout() {
        assert_eq!(offset_of!(CUVIDGETDECODESTATUS, pReserved), 128);
        assert_eq!(size_of::<CUVIDGETDECODESTATUS>(), 192);
    }

    #[test]
    fn video_format_layout() {
        type T = CUVIDEOFORMAT;
        assert_eq!(offset_of!(T, frame_rate), 4);
        assert_eq!(offset_of!(T, progressive_sequence), 12);
        assert_eq!(offset_of!(T, min_num_decode_surfaces), 15);
        assert_eq!(offset_of!(T, coded_width), 16);
        assert_eq!(offset_of!(T, display_area), 24);
        assert_eq!(offset_of!(T, chroma_format), 40);
        assert_eq!(offset_of!(T, bitrate), 44);
        assert_eq!(offset_of!(T, display_aspect_ratio), 48);
        assert_eq!(offset_of!(T, video_signal_description), 56);
        assert_eq!(offset_of!(T, seqhdr_data_length), 60);
        assert_eq!(size_of::<T>(), 64);
    }

    #[test]
    fn parser_disp_info_layout() {
        assert_eq!(offset_of!(CUVIDPARSERDISPINFO, timestamp), 16);
        assert_eq!(size_of::<CUVIDPARSERDISPINFO>(), 24);
    }

    #[test]
    fn source_data_packet_layout() {
        type T = CUVIDSOURCEDATAPACKET;
        assert_eq!(offset_of!(T, payload_size), L);
        assert_eq!(offset_of!(T, payload), 2 * L);
        assert_eq!(offset_of!(T, timestamp), 2 * L + 8);
        assert_eq!(size_of::<T>(), 2 * L + 16);
    }

    #[test]
    fn parser_params_layout() {
        type T = CUVIDPARSERPARAMS;
        assert_eq!(offset_of!(T, ulMaxDisplayDelay), 16);
        assert_eq!(offset_of!(T, bits), 20);
        assert_eq!(offset_of!(T, uReserved1), 24);
        assert_eq!(offset_of!(T, pUserData), 40);
        assert_eq!(offset_of!(T, pfnSequenceCallback), 48);
        assert_eq!(offset_of!(T, pfnDecodePicture), 56);
        assert_eq!(offset_of!(T, pfnDisplayPicture), 64);
        assert_eq!(offset_of!(T, pfnGetOperatingPoint), 72);
        assert_eq!(offset_of!(T, pfnGetSEIMsg), 80);
        assert_eq!(offset_of!(T, pvReserved2), 88);
        assert_eq!(offset_of!(T, pExtVideoInfo), 128);
        assert_eq!(size_of::<T>(), 136);
    }

    #[test]
    fn signal_desc_bits() {
        let d = CUVIDSIGNALDESC { bits: 0b0000_1101, ..Default::default() };
        assert_eq!(d.video_format(), 5);
        assert!(d.full_range());
    }
}
