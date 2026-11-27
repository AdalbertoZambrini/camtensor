//! Integration tests of the pure-Rust MJPEG decoder (zune-jpeg). Media is
//! generated with the FFmpeg CLI; tests print "skipped" when it is missing.

#![cfg(feature = "jpeg")]

mod common;

use camtensor_core::{Codec, ColorRange, Device, PixelFormat};
use camtensor_decode::jpeg::JpegDecoder;
use camtensor_decode::{Decoder, DecoderKind};
use common::{Spec, decode_all, media, packet, rgb_means};

fn mjpeg(name: &'static str, size: &str, pix: &str) -> Spec {
    Spec::new(
        name,
        Codec::Mjpeg,
        &format!("testsrc2=size={size}:rate=25"),
        20,
        &format!("-c:v mjpeg -q:v 3 -pix_fmt {pix}"),
    )
}

fn check(test: &str, spec: Spec) {
    let Some(m) = media(test, &spec) else { return };
    let mut dec = JpegDecoder::new();
    assert_eq!(dec.kind(), DecoderKind::Jpeg);
    assert_eq!(dec.device(), Device::Cpu);
    let frames = decode_all(&mut dec, &m.packets, |_| true).unwrap();
    assert_eq!(frames.len(), m.packets.len());
    for (f, p) in frames.iter().zip(&m.packets) {
        assert_eq!(f.pts, p.pts);
        assert_eq!((f.desc.width, f.desc.height), (m.width, m.height));
        assert_eq!(f.desc.format, PixelFormat::Rgb24);
        assert_eq!(f.desc.color.range, ColorRange::Full);
        assert_eq!(f.host_bytes().unwrap().len(), f.desc.byte_len());
    }
}

#[test]
fn mjpeg_720p_420() {
    check("mjpeg_720p_420", mjpeg("z_720_420", "1280x720", "yuvj420p"));
}

#[test]
fn mjpeg_odd_size_422() {
    check("mjpeg_odd_size_422", mjpeg("z_odd_422", "642x362", "yuvj422p"));
}

#[test]
fn emit_false_skips_and_corrupt_is_dropped() {
    let spec = mjpeg("z_720_420", "1280x720", "yuvj420p");
    let Some(m) = media("emit_false_skips_and_corrupt_is_dropped", &spec) else { return };
    let mut dec = JpegDecoder::new();
    let frames = decode_all(&mut dec, &m.packets, |i| i % 2 == 0).unwrap();
    let want: Vec<i64> = m.packets.iter().step_by(2).map(|p| p.pts).collect();
    assert_eq!(frames.iter().map(|f| f.pts).collect::<Vec<_>>(), want);

    let mut out = Vec::new();
    let p = &m.packets[0];
    for bad in [p.data[..300].to_vec(), vec![0u8; 64], vec![0xff, 0xd8, 0xff, 0xc0, 0, 2]] {
        dec.decode(&packet(Codec::Mjpeg, bad, 1, true, 0), true, &mut |f| out.push(f)).unwrap();
    }
    // A truncated image may still decode partially; garbage must not.
    assert!(out.len() <= 1);
    dec.decode(p, true, &mut |f| out.push(f)).unwrap();
    assert_eq!(out.last().unwrap().pts, p.pts);
}

#[test]
fn solid_colour_rgb() {
    let spec = Spec::new(
        "z_solid",
        Codec::Mjpeg,
        "color=c=0x3366CC:size=642x362:rate=25,format=rgb24,\
         scale=out_color_matrix=bt601:out_range=pc,format=yuvj444p",
        3,
        "-c:v mjpeg -q:v 2",
    );
    let Some(m) = media("solid_colour_rgb", &spec) else { return };
    let mut dec = JpegDecoder::new();
    let frames = decode_all(&mut dec, &m.packets, |_| true).unwrap();
    let rgb = rgb_means(&frames[0]);
    eprintln!("solid_colour_rgb: {rgb:.2?}");
    for (got, want) in rgb.iter().zip([51.0, 102.0, 204.0]) {
        assert!((got - want).abs() < 3.0, "got {rgb:?}");
    }
}

#[test]
fn decoder_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<JpegDecoder>();
}
