//! Integration tests of the libavcodec decoder. Media is generated with the
//! FFmpeg CLI (`CAMTENSOR_FFMPEG`); the shared libraries are located through
//! `CAMTENSOR_FFMPEG_DIR`. Tests print "skipped" and pass when either is
//! missing.

#![cfg(feature = "ffmpeg")]

mod common;

use std::collections::HashSet;

use camtensor_core::{
    Codec, ColorMatrix, ColorRange, DecoderPref, Device, Error, PixelFormat, StreamParams,
};
use camtensor_decode::ffmpeg::FfmpegDecoder;
use camtensor_decode::{Decoder, DecoderKind, DecoderOptions};
use common::{Media, Spec, decode_all, media, packet, plane_means};

const SRC_720: &str = "testsrc2=size=1280x720:rate=25";
const SRC_ODD: &str = "testsrc2=size=642x362:rate=25";

fn h264(name: &'static str, src: &str, extra: &str) -> Spec {
    Spec::new(
        name,
        Codec::H264,
        src,
        50,
        &format!("-c:v libx264 -profile:v main -pix_fmt yuv420p -g 25 -preset veryfast {extra}"),
    )
}

fn h265(name: &'static str, src: &str) -> Spec {
    Spec::new(
        name,
        Codec::H265,
        src,
        50,
        "-c:v libx265 -pix_fmt yuv420p -preset veryfast \
         -x265-params log-level=error:keyint=25:bframes=3",
    )
}

fn mjpeg(name: &'static str, src: &str, pix: &str) -> Spec {
    Spec::new(name, Codec::Mjpeg, src, 20, &format!("-c:v mjpeg -q:v 3 -pix_fmt {pix}"))
}

fn params(m: &Media) -> StreamParams {
    StreamParams {
        codec: m.codec,
        width: m.width,
        height: m.height,
        extradata: Default::default(),
        fps: Some(25.0),
    }
}

/// Common preamble: libraries + media, or skip.
fn setup(test: &str, spec: &Spec) -> Option<std::sync::Arc<Media>> {
    if !common::ffmpeg_libs(test) {
        return None;
    }
    media(test, spec)
}

/// Decodes everything and checks count, size, format and pts order.
fn check_full_decode(test: &str, spec: Spec, format: PixelFormat, threads: usize) {
    let Some(m) = setup(test, &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), threads).unwrap();
    assert_eq!(dec.kind(), DecoderKind::Ffmpeg);
    assert_eq!(dec.device(), Device::Cpu);
    let frames = decode_all(&mut dec, &m.packets, |_| true).unwrap();
    assert_eq!(frames.len(), m.packets.len(), "{test}: frame count");
    for f in &frames {
        assert_eq!((f.desc.width, f.desc.height), (m.width, m.height), "{test}: size");
        assert_eq!(f.desc.format, format, "{test}: format");
        assert_eq!(f.host_bytes().unwrap().len(), f.desc.byte_len(), "{test}: packed buffer");
    }
    // Output is in presentation order with the input timestamps.
    let pts: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    assert_eq!(pts, m.sorted_pts(), "{test}: pts order");
}

#[test]
fn h264_main_no_bframes_720p() {
    let spec = h264("h264_nob_720", SRC_720, "-bf 0");
    if let Some(m) = setup("h264_main_no_bframes_720p", &spec) {
        assert!(!m.has_reordering());
    }
    check_full_decode("h264_main_no_bframes_720p", spec, PixelFormat::Yuv420p, 0);
}

#[test]
fn h264_main_bframes_720p_reorders() {
    let spec = h264("h264_b_720", SRC_720, "-bf 3");
    let Some(m) = setup("h264_main_bframes_720p_reorders", &spec) else { return };
    assert!(m.has_reordering(), "clip should contain B-frames");
    check_full_decode("h264_main_bframes_720p_reorders", spec.clone(), PixelFormat::Yuv420p, 0);
    check_full_decode("h264_main_bframes_720p_reorders/1t", spec, PixelFormat::Yuv420p, 1);
}

#[test]
fn h264_odd_size() {
    check_full_decode("h264_odd_size", h264("h264_odd", SRC_ODD, "-bf 2"), PixelFormat::Yuv420p, 0);
}

#[test]
fn h265_720p() {
    let spec = h265("h265_720", SRC_720);
    if let Some(m) = setup("h265_720p", &spec) {
        assert!(m.has_reordering(), "clip should contain B-frames");
    }
    check_full_decode("h265_720p", spec, PixelFormat::Yuv420p, 0);
}

#[test]
fn h265_odd_size() {
    check_full_decode("h265_odd_size", h265("h265_odd", SRC_ODD), PixelFormat::Yuv420p, 0);
}

#[test]
fn mjpeg_720p_and_odd() {
    check_full_decode(
        "mjpeg_720p",
        mjpeg("mjpeg_720", SRC_720, "yuvj420p"),
        PixelFormat::Yuv420p,
        0,
    );
    check_full_decode(
        "mjpeg_odd_422",
        mjpeg("mjpeg_odd422", SRC_ODD, "yuvj422p"),
        PixelFormat::Yuv422p,
        0,
    );
}

#[test]
fn other_chroma_formats() {
    // libavcodec's H.264 decoder outputs 4:0:0 as yuv420p with neutral
    // chroma; HEVC outputs real grey frames.
    let cases = [
        (
            "h264_422",
            Codec::H264,
            "-c:v libx264 -profile:v high422 -pix_fmt yuv422p",
            PixelFormat::Yuv422p,
        ),
        (
            "h264_444",
            Codec::H264,
            "-c:v libx264 -profile:v high444 -pix_fmt yuv444p",
            PixelFormat::Yuv444p,
        ),
        (
            "h265_gray",
            Codec::H265,
            "-c:v libx265 -pix_fmt gray -x265-params log-level=error",
            PixelFormat::Gray8,
        ),
    ];
    for (name, codec, enc, fmt) in cases {
        let spec = Spec::new(name, codec, SRC_ODD, 10, enc);
        check_full_decode(name, spec, fmt, 0);
    }
}

#[test]
fn ten_bit_is_rejected_with_format_name() {
    let spec = Spec::new(
        "h265_10bit",
        Codec::H265,
        SRC_ODD,
        5,
        "-c:v libx265 -pix_fmt yuv420p10le -x265-params log-level=error",
    );
    let Some(m) = setup("ten_bit_is_rejected_with_format_name", &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), 0).unwrap();
    match decode_all(&mut dec, &m.packets, |_| true) {
        Err(Error::Decode(msg)) => assert!(msg.contains("yuv420p10"), "{msg}"),
        other => panic!("expected a decode error, got {other:?}"),
    }
}

#[test]
fn emit_false_suppresses_exactly_those_frames() {
    let spec = h264("h264_b_720", SRC_720, "-bf 3");
    let Some(m) = setup("emit_false_suppresses_exactly_those_frames", &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), 0).unwrap();
    let emit = |i: usize| i % 3 != 1;
    let frames = decode_all(&mut dec, &m.packets, emit).unwrap();
    let mut expected: Vec<i64> =
        m.packets.iter().enumerate().filter(|(i, _)| emit(*i)).map(|(_, p)| p.pts).collect();
    expected.sort_unstable();
    let got: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    assert_eq!(got, expected);
}

#[test]
fn flush_then_reuse() {
    let spec = h264("h264_b_720", SRC_720, "-bf 3");
    let Some(m) = setup("flush_then_reuse", &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), 0).unwrap();
    // First half, flush.
    let half = &m.packets[..25];
    let a = decode_all(&mut dec, half, |_| true).unwrap();
    assert_eq!(a.len(), half.len());
    // Whole clip again, with shifted timestamps.
    let shifted: Vec<_> = m
        .packets
        .iter()
        .map(|p| {
            let mut q = p.clone();
            q.pts += 10_000_000;
            q
        })
        .collect();
    let b = decode_all(&mut dec, &shifted, |_| true).unwrap();
    assert_eq!(b.len(), m.packets.len());
    assert!(b.iter().all(|f| f.pts >= 10_000_000));
    // Flushing an idle decoder is a no-op.
    dec.flush(&mut |_| panic!("no frame expected")).unwrap();
}

#[test]
fn corrupt_packets_do_not_kill_the_decoder() {
    let spec = h264("h264_nob_720", SRC_720, "-bf 0");
    let Some(m) = setup("corrupt_packets_do_not_kill_the_decoder", &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), 0).unwrap();
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as u8
    };
    let mut frames = Vec::new();
    let mut sink = |f| frames.push(f);
    for (i, p) in m.packets.iter().enumerate() {
        match i {
            // Bit flips in the slice data of a P-frame.
            10 => {
                let mut d = p.data.to_vec();
                for k in (d.len() / 3..d.len()).step_by(7) {
                    d[k] ^= rnd();
                }
                let q = packet(p.codec, d, p.pts, p.keyframe, p.seq);
                dec.decode(&q, true, &mut sink).unwrap();
            }
            // Truncated P-frame.
            15 => {
                let q = packet(p.codec, p.data[..p.data.len() / 2].to_vec(), p.pts, false, p.seq);
                dec.decode(&q, true, &mut sink).unwrap();
            }
            // Pure garbage, with and without a start code, before frame 20.
            20 => {
                let g1: Vec<u8> =
                    [0, 0, 0, 1].into_iter().chain((0..3000).map(|_| rnd())).collect();
                let g2: Vec<u8> = (0..500).map(|_| rnd()).collect();
                for (k, g) in [g1, g2].into_iter().enumerate() {
                    let q = packet(p.codec, g, 9_000_000 + k as i64, false, 0);
                    dec.decode(&q, true, &mut sink).unwrap();
                }
                dec.decode(p, true, &mut sink).unwrap();
            }
            _ => dec.decode(p, true, &mut sink).unwrap(),
        }
    }
    dec.flush(&mut sink).unwrap();
    eprintln!("{} of {} frames survived", frames.len(), m.packets.len());
    let pts: HashSet<i64> = frames.iter().map(|f| f.pts).collect();
    // Everything from the next keyframe (25) on must be intact.
    for p in &m.packets[25..] {
        assert!(pts.contains(&p.pts), "frame {} missing after recovery", p.pts);
    }
    assert!(frames.len() >= 40, "only {} frames survived", frames.len());
}

#[test]
fn corrupt_jpeg_does_not_kill_the_decoder() {
    let spec = mjpeg("mjpeg_720", SRC_720, "yuvj420p");
    let Some(m) = setup("corrupt_jpeg_does_not_kill_the_decoder", &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), 0).unwrap();
    let mut frames = Vec::new();
    let mut sink = |f| frames.push(f);
    let p = &m.packets[0];
    let trunc = packet(p.codec, p.data[..200].to_vec(), 1, true, 0);
    dec.decode(&trunc, true, &mut sink).unwrap();
    let junk = packet(p.codec, vec![0xab; 1000], 2, true, 0);
    dec.decode(&junk, true, &mut sink).unwrap();
    for p in &m.packets {
        dec.decode(p, true, &mut sink).unwrap();
    }
    dec.flush(&mut sink).unwrap();
    let pts: HashSet<i64> = frames.iter().map(|f| f.pts).collect();
    assert!(m.packets.iter().all(|p| pts.contains(&p.pts)));
}

/// Solid 0x3366CC (R=51, G=102, B=204) converted by swscale with the given
/// matrix / range.
fn solid(name: &'static str, codec: Codec, size: &str, vf: &str, enc: &str) -> Spec {
    Spec::new(
        name,
        codec,
        &format!("color=c=0x3366CC:size={size}:rate=25,format=rgb24,{vf}"),
        5,
        enc,
    )
}

fn check_colour(
    test: &str,
    spec: Spec,
    range: ColorRange,
    matrix: ColorMatrix,
    expect_yuv: [f64; 3],
) {
    let Some(m) = setup(test, &spec) else { return };
    let mut dec = FfmpegDecoder::open(&params(&m), 0).unwrap();
    let frames = decode_all(&mut dec, &m.packets, |_| true).unwrap();
    assert_eq!(frames.len(), m.packets.len());
    let f = frames.last().unwrap();
    assert_eq!(f.desc.color.range, range, "{test}: range");
    assert_eq!(f.desc.color.matrix, matrix, "{test}: matrix");
    let means = plane_means(f);
    eprintln!("{test}: plane means {means:.2?}, expected {expect_yuv:.2?}");
    for (c, (got, want)) in means.iter().zip(expect_yuv).enumerate() {
        assert!((got - want).abs() < 1.5, "{test}: plane {c}: got {got:.2}, want {want:.2}");
    }
}

// The sizes are chosen so that the signalled matrix differs from the
// size-based default (BT.709 for >= 720 lines), proving it is read from the
// stream.
//
// Expected values: Y = 16 + 219 Y', C = 128 + 224 P (limited) or
// Y = 255 Y', C = 128 + 255 P (full), with Y'/Pb/Pr from the BT.709 or
// BT.601 matrix applied to (0.2, 0.4, 0.8).

#[test]
fn colour_h264_bt709_limited() {
    let spec = solid(
        "solid_709_tv",
        Codec::H264,
        "640x360",
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v libx264 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv",
    );
    check_colour(
        "colour_h264_bt709_limited",
        spec,
        ColorRange::Limited,
        ColorMatrix::Bt709,
        [100.61, 177.93, 101.49],
    );
}

#[test]
fn colour_h264_bt709_full() {
    let spec = solid(
        "solid_709_pc",
        Codec::H264,
        "1280x720",
        "scale=out_color_matrix=bt709:out_range=pc,format=yuv420p",
        "-c:v libx264 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range pc",
    );
    check_colour(
        "colour_h264_bt709_full",
        spec,
        ColorRange::Full,
        ColorMatrix::Bt709,
        [98.52, 184.84, 97.82],
    );
}

#[test]
fn colour_h265_bt601_limited() {
    let spec = solid(
        "solid_601_tv_hevc",
        Codec::H265,
        "1280x720",
        "scale=out_color_matrix=bt601:out_range=tv,format=yuv420p",
        "-c:v libx265 -colorspace smpte170m -color_range tv -x265-params log-level=error",
    );
    check_colour(
        "colour_h265_bt601_limited",
        spec,
        ColorRange::Limited,
        ColorMatrix::Bt601,
        [100.49, 180.36, 98.31],
    );
}

#[test]
fn colour_h265_bt709_full() {
    // HEVC never uses the yuvj* formats: the range comes from the VUI only.
    let spec = solid(
        "solid_709_pc_hevc",
        Codec::H265,
        "640x360",
        "scale=out_color_matrix=bt709:out_range=pc,format=yuv420p",
        "-c:v libx265 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range pc          -x265-params log-level=error",
    );
    check_colour(
        "colour_h265_bt709_full",
        spec,
        ColorRange::Full,
        ColorMatrix::Bt709,
        [98.52, 184.84, 97.82],
    );
}

#[test]
fn colour_mjpeg_full_bt601() {
    let spec = solid(
        "solid_mjpeg",
        Codec::Mjpeg,
        "642x362",
        "scale=out_color_matrix=bt601:out_range=pc,format=yuvj420p",
        "-c:v mjpeg -q:v 2",
    );
    check_colour(
        "colour_mjpeg_full_bt601",
        spec,
        ColorRange::Full,
        ColorMatrix::Bt601,
        [98.38, 187.59, 94.22],
    );
}

#[test]
fn open_dispatch_prefers_ffmpeg_on_cpu() {
    if !common::ffmpeg_libs("open_dispatch_prefers_ffmpeg_on_cpu") {
        return;
    }
    let opts = DecoderOptions { pref: DecoderPref::Cpu, device: Device::Cpu, ..Default::default() };
    for codec in [Codec::H264, Codec::H265, Codec::Mjpeg] {
        let p =
            StreamParams { codec, width: 0, height: 0, extradata: Default::default(), fps: None };
        let d = camtensor_decode::open(&p, &opts).unwrap();
        assert_eq!(d.kind(), DecoderKind::Ffmpeg);
    }
}

#[test]
fn decoder_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<FfmpegDecoder>();
}
