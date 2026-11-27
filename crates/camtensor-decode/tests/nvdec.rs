//! NVDEC decoder tests on bitstreams generated with ffmpeg.
//!
//! Skipped (with a message) when NVDEC or ffmpeg is unavailable. ffmpeg is
//! taken from `CAMTENSOR_FFMPEG` or the default developer path below.
#![cfg(feature = "nvdec")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use bytes::Bytes;
use camtensor_core::{
    Affine2, Codec, ColorMatrix, ColorRange, DType, EncodedPacket, Error, Frame, FrameData,
    FrameDesc, Normalize, PixelFormat, StreamParams, TensorSpec,
};
use camtensor_cuda::{CudaDevice, CudaStream, DeviceBuffer, Preprocessor, SourceImage};
use camtensor_decode::Decoder;
use camtensor_decode::nvdec::NvdecDecoder;

const DEFAULT_FFMPEG: &str = r"C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe";

fn ffmpeg() -> Option<PathBuf> {
    let p = std::env::var_os("CAMTENSOR_FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_FFMPEG));
    p.exists().then_some(p)
}

/// Returns ffmpeg when both it and NVDEC are usable.
fn setup() -> Option<PathBuf> {
    if !camtensor_cuda::nvdec::is_available() {
        eprintln!("skipping: NVDEC not available");
        return None;
    }
    let f = ffmpeg();
    if f.is_none() {
        eprintln!("skipping: ffmpeg not found (set CAMTENSOR_FFMPEG)");
    }
    f
}

fn tmp(name: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("nvdec-{name}"))
}

/// Runs ffmpeg with `args` (output path last) unless the output exists.
fn generate(ffmpeg: &Path, name: &str, args: &[&str]) -> PathBuf {
    // Tests run in parallel and share media files: generate under a lock
    // into a temporary name, then rename.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = tmp(name);
    if !out.exists() {
        let ext = out.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        let part = out.with_extension(format!("part.{ext}"));
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-y", "-loglevel", "error"]).args(args);
        let st = cmd.arg(&part).status().expect("run ffmpeg");
        assert!(st.success(), "ffmpeg failed for {name}");
        std::fs::rename(&part, &out).unwrap();
    }
    out
}

fn testsrc(size: &str, frames: u32) -> String {
    format!("testsrc2=size={size}:rate=25:duration={}", frames as f64 / 25.0)
}

fn h264(ff: &Path, name: &str, size: &str, frames: u32, extra: &[&str]) -> Vec<u8> {
    let src = testsrc(size, frames);
    let mut args =
        vec!["-f", "lavfi", "-i", &src, "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p"];
    args.extend_from_slice(extra);
    args.extend_from_slice(&["-bsf:v", "h264_mp4toannexb", "-f", "h264"]);
    std::fs::read(generate(ff, name, &args)).unwrap()
}

fn h265(ff: &Path, name: &str, size: &str, frames: u32, pix_fmt: &str) -> Vec<u8> {
    let src = testsrc(size, frames);
    let args = [
        "-f",
        "lavfi",
        "-i",
        &src,
        "-c:v",
        "libx265",
        "-pix_fmt",
        pix_fmt,
        "-x265-params",
        "keyint=25:bframes=0:log-level=error",
        "-f",
        "hevc",
    ];
    std::fs::read(generate(ff, name, &args)).unwrap()
}

/// Decodes `input` with ffmpeg to raw NV12 frames.
fn ffmpeg_nv12(ff: &Path, input: &Path, name: &str) -> Vec<u8> {
    let i = input.to_string_lossy().into_owned();
    std::fs::read(generate(ff, name, &["-i", &i, "-f", "rawvideo", "-pix_fmt", "nv12"])).unwrap()
}

/// Splits an Annex B stream into access units.
fn split_annexb(data: &[u8], codec: Codec) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut cur_has_vcl = false;
    for (k, &s) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map(|&n| n - 3).unwrap_or(data.len());
        while end > s && data[end - 1] == 0 {
            end -= 1;
        }
        let nal = &data[s..end];
        if nal.is_empty() {
            continue;
        }
        let (vcl, first_slice, au_start) = match codec {
            Codec::H264 => {
                let t = nal[0] & 0x1f;
                let vcl = (1..=5).contains(&t);
                (vcl, vcl && nal.len() > 1 && nal[1] & 0x80 != 0, matches!(t, 6..=9 | 14..=18))
            }
            _ => {
                let t = (nal[0] >> 1) & 0x3f;
                let vcl = t < 32;
                (
                    vcl,
                    vcl && nal.len() > 2 && nal[2] & 0x80 != 0,
                    matches!(t, 32..=35 | 39 | 41..=44 | 48..=55),
                )
            }
        };
        if cur_has_vcl && ((vcl && first_slice) || (!vcl && au_start)) {
            aus.push(std::mem::take(&mut cur));
            cur_has_vcl = false;
        }
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        cur_has_vcl |= vcl;
    }
    if !cur.is_empty() {
        aus.push(cur);
    }
    aus
}

/// Splits concatenated JPEG images (as written by `-f mjpeg`).
fn split_jpeg(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts: Vec<usize> = (0..data.len().saturating_sub(2))
        .filter(|&i| data[i..i + 3] == [0xff, 0xd8, 0xff])
        .collect();
    starts.push(data.len());
    starts.windows(2).map(|w| data[w[0]..w[1]].to_vec()).collect()
}

fn packet(codec: Codec, data: Vec<u8>, i: usize) -> EncodedPacket {
    EncodedPacket {
        stream_id: Arc::from("test"),
        codec,
        data: Bytes::from(data),
        pts: i as i64 * 3600,
        keyframe: false,
        seq: i as u64,
        recv_time_ns: 0,
        capture_time_ns: None,
        loss: false,
    }
}

fn params(codec: Codec) -> StreamParams {
    StreamParams { codec, width: 0, height: 0, extradata: Bytes::new(), fps: Some(25.0) }
}

fn decode_all(codec: Codec, aus: Vec<Vec<u8>>) -> Vec<Frame> {
    let mut dec = NvdecDecoder::open(&params(codec), 0, 8).expect("open nvdec");
    let mut frames = Vec::new();
    for (i, au) in aus.into_iter().enumerate() {
        dec.decode(&packet(codec, au, i), true, &mut |f| frames.push(f)).expect("decode");
    }
    dec.flush(&mut |f| frames.push(f)).expect("flush");
    frames
}

fn download(frame: &Frame) -> Vec<u8> {
    let FrameData::Device(mem) = &frame.data else { panic!("expected a device frame") };
    let buf = mem.as_any().downcast_ref::<DeviceBuffer>().expect("DeviceBuffer");
    let mut v = buf.to_vec().unwrap();
    v.truncate(frame.desc.byte_len());
    v
}

/// CPU NV12 â†’ RGB letterbox reference (bilinear), `[c][y*W+x]` pixel units.
fn cpu_rgb(nv12: &[u8], desc: &FrameDesc, m: &Affine2, w: usize, h: usize) -> Vec<Vec<f32>> {
    let (sw, sh) = (desc.width as usize, desc.height as usize);
    let (cw, ch) = (sw.div_ceil(2), sh.div_ceil(2));
    let (yp, cp) = (desc.planes[0], desc.planes[1]);
    let bil =
        |off: usize, pitch: usize, pw: usize, ph: usize, step: usize, c: usize, x: f32, y: f32| {
            let at = |xi: i64, yi: i64| {
                let xi = xi.clamp(0, pw as i64 - 1) as usize;
                let yi = yi.clamp(0, ph as i64 - 1) as usize;
                nv12[off + yi * pitch + xi * step + c] as f32
            };
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            (1.0 - fx) * (1.0 - fy) * at(x0, y0)
                + fx * (1.0 - fy) * at(x0 + 1, y0)
                + (1.0 - fx) * fy * at(x0, y0 + 1)
                + fx * fy * at(x0 + 1, y0 + 1)
        };
    let (kr, kgb, kgr, kb) = match desc.color.matrix {
        ColorMatrix::Bt601 => (1.402f32, 0.344136f32, 0.714136f32, 1.772f32),
        ColorMatrix::Bt709 => (1.5748, 0.187324, 0.468124, 1.8556),
    };
    let mut out = vec![vec![0f32; w * h]; 3];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = m.apply(x as f32, y as f32);
            let v = if sx < -0.5 || sx > sw as f32 - 0.5 || sy < -0.5 || sy > sh as f32 - 0.5 {
                [114.0; 3]
            } else {
                let yy = bil(yp.offset, yp.pitch, sw, sh, 1, 0, sx, sy);
                let (cx, cy) = ((sx + 0.5) / 2.0 - 0.5, (sy + 0.5) / 2.0 - 0.5);
                let u = bil(cp.offset, cp.pitch, cw, ch, 2, 0, cx, cy);
                let vv = bil(cp.offset, cp.pitch, cw, ch, 2, 1, cx, cy);
                let (y1, cb, cr) = match desc.color.range {
                    ColorRange::Limited => (
                        (yy - 16.0) * 255.0 / 219.0,
                        (u - 128.0) * 255.0 / 224.0,
                        (vv - 128.0) * 255.0 / 224.0,
                    ),
                    ColorRange::Full => (yy, u - 128.0, vv - 128.0),
                };
                [
                    (y1 + kr * cr).clamp(0.0, 255.0),
                    (y1 - kgb * cb - kgr * cr).clamp(0.0, 255.0),
                    (y1 + kb * cb).clamp(0.0, 255.0),
                ]
            };
            for c in 0..3 {
                out[c][y * w + x] = v[c];
            }
        }
    }
    out
}

/// Converts a decoded frame on the GPU and checks it against the CPU
/// reference computed from the downloaded NV12 (tolerance 1.0 pixel units).
fn check_rgb(frame: &Frame) {
    let device = CudaDevice::get(0).unwrap();
    let stream = CudaStream::new(&device).unwrap();
    let pre = Preprocessor::get(&device).unwrap();
    let spec = TensorSpec {
        width: 640,
        height: 640,
        normalize: Normalize::NONE,
        dtype: DType::F32,
        ..Default::default()
    };
    let m =
        Affine2::fit(frame.desc.width as f32, frame.desc.height as f32, 640.0, 640.0, spec.resize);
    let s = Affine2::sampling_matrix(&m).unwrap();
    let out = DeviceBuffer::alloc(&device, spec.image_bytes()).unwrap();
    pre.preprocess_into(&stream, &SourceImage::from_frame(frame).unwrap(), &[s], &out, 0, &spec)
        .unwrap();
    stream.synchronize().unwrap();
    let gpu: Vec<f32> =
        out.to_vec().unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let cpu = cpu_rgb(&download(frame), &frame.desc, &s, 640, 640);
    let mut max = 0f32;
    for c in 0..3 {
        for p in 0..640 * 640 {
            max = max.max((gpu[c * 640 * 640 + p] - cpu[c][p]).abs());
        }
    }
    assert!(max <= 1.0, "GPU vs CPU RGB max diff {max}");
    // testsrc2 is colourful: the image area must not be flat.
    let row = &cpu[0][320 * 640..321 * 640];
    let (lo, hi) = row.iter().fold((255f32, 0f32), |(l, h), &v| (l.min(v), h.max(v)));
    assert!(hi - lo > 50.0, "decoded image looks flat ({lo}..{hi})");
}

/// Compares decoded frames with ffmpeg's NV12 output.
fn compare_with_ffmpeg(frames: &[Frame], reference: &[u8], max_mean_diff: f64) {
    let n = frames[0].desc.byte_len();
    assert_eq!(reference.len(), n * frames.len(), "ffmpeg frame count differs");
    for (i, f) in frames.iter().enumerate().step_by(7) {
        let ours = download(f);
        let theirs = &reference[i * n..(i + 1) * n];
        let sum: u64 = ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
            .sum();
        let mean = sum as f64 / n as f64;
        assert!(mean <= max_mean_diff, "frame {i}: mean abs diff {mean} vs ffmpeg");
    }
}

#[test]
fn h264_decode_matches_ffmpeg() {
    let Some(ff) = setup() else { return };
    let data = h264(&ff, "h264-720p.h264", "1280x720", 50, &["-bf", "0"]);
    let aus = split_annexb(&data, Codec::H264);
    assert_eq!(aus.len(), 50);
    let frames = decode_all(Codec::H264, aus);
    assert_eq!(frames.len(), 50);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.desc.width, f.desc.height, f.desc.format), (1280, 720, PixelFormat::Nv12));
        assert_eq!(f.pts, i as i64 * 3600);
        assert!(f.is_device());
    }
    assert_eq!(frames[0].desc.color.matrix, ColorMatrix::Bt709);
    assert_eq!(frames[0].desc.color.range, ColorRange::Limited);
    let reference = ffmpeg_nv12(&ff, &tmp("h264-720p.h264"), "h264-720p.nv12");
    compare_with_ffmpeg(&frames, &reference, 0.0);
    check_rgb(&frames[10]);
}

#[test]
fn h264_with_b_frames_outputs_every_picture() {
    let Some(ff) = setup() else { return };
    let data = h264(&ff, "h264-bframes.h264", "640x360", 40, &["-bf", "3"]);
    let aus = split_annexb(&data, Codec::H264);
    assert_eq!(aus.len(), 40);
    let frames = decode_all(Codec::H264, aus);
    assert_eq!(frames.len(), 40);
    // Packets carry decode-order timestamps here; the set must be complete.
    let mut pts: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    pts.sort_unstable();
    assert_eq!(pts, (0..40).map(|i| i * 3600).collect::<Vec<_>>());
}

#[test]
fn h265_decode_matches_ffmpeg() {
    let Some(ff) = setup() else { return };
    let data = h265(&ff, "h265-720p.hevc", "1280x720", 50, "yuv420p");
    let aus = split_annexb(&data, Codec::H265);
    assert_eq!(aus.len(), 50);
    let frames = decode_all(Codec::H265, aus);
    assert_eq!(frames.len(), 50);
    assert!(frames.iter().all(|f| (f.desc.width, f.desc.height) == (1280, 720)));
    let reference = ffmpeg_nv12(&ff, &tmp("h265-720p.hevc"), "h265-720p.nv12");
    compare_with_ffmpeg(&frames, &reference, 0.0);
    check_rgb(&frames[3]);
}

#[test]
fn mjpeg_decode() {
    let Some(ff) = setup() else { return };
    for pix in ["yuvj420p", "yuvj422p"] {
        let name = format!("mjpeg-{pix}.mjpeg");
        let src = testsrc("640x480", 10);
        let data = std::fs::read(generate(
            &ff,
            &name,
            &[
                "-f", "lavfi", "-i", &src, "-c:v", "mjpeg", "-pix_fmt", pix, "-q:v", "3", "-f",
                "mjpeg",
            ],
        ))
        .unwrap();
        let jpegs = split_jpeg(&data);
        assert_eq!(jpegs.len(), 10);
        let frames = match NvdecDecoder::open(&params(Codec::Mjpeg), 0, 4) {
            Ok(_) => decode_all(Codec::Mjpeg, jpegs),
            Err(Error::Unavailable(e)) => {
                eprintln!("skipping MJPEG: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        assert_eq!(frames.len(), 10, "{pix}");
        let f = &frames[0];
        assert_eq!((f.desc.width, f.desc.height), (640, 480));
        assert_eq!(f.desc.color.range, ColorRange::Full);
        assert_eq!(f.desc.color.matrix, ColorMatrix::Bt601);
        // JPEG IDCTs are not bit-exact across decoders; compare luma loosely.
        let reference = ffmpeg_nv12(&ff, &tmp(&name), &format!("{name}.nv12"));
        let ours = download(f);
        let luma = 640 * 480;
        let mean = ours[..luma]
            .iter()
            .zip(&reference[..luma])
            .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
            .sum::<f64>()
            / luma as f64;
        assert!(mean < 2.0, "{pix}: luma mean abs diff {mean}");
        check_rgb(f);
    }
}

#[test]
fn emit_false_suppresses_frames() {
    let Some(ff) = setup() else { return };
    let data = h264(&ff, "h264-720p.h264", "1280x720", 50, &["-bf", "0"]);
    let aus = split_annexb(&data, Codec::H264);
    let mut dec = NvdecDecoder::open(&params(Codec::H264), 0, 4).unwrap();
    let mut frames = Vec::new();
    for (i, au) in aus.into_iter().enumerate() {
        dec.decode(&packet(Codec::H264, au, i), i % 2 == 0, &mut |f| frames.push(f)).unwrap();
    }
    dec.flush(&mut |f| frames.push(f)).unwrap();
    let pts: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    assert_eq!(pts, (0..50).step_by(2).map(|i| i * 3600).collect::<Vec<i64>>());
}

#[test]
fn flush_then_continue_and_resolution_change() {
    let Some(ff) = setup() else { return };
    let small =
        split_annexb(&h264(&ff, "h264-360p.h264", "640x360", 25, &["-bf", "0"]), Codec::H264);
    let big =
        split_annexb(&h264(&ff, "h264-720p.h264", "1280x720", 50, &["-bf", "0"]), Codec::H264);
    let mut dec = NvdecDecoder::open(&params(Codec::H264), 0, 4).unwrap();
    let mut frames = Vec::new();
    for (i, au) in small.iter().take(25).enumerate() {
        dec.decode(&packet(Codec::H264, au.clone(), i), true, &mut |f| frames.push(f)).unwrap();
    }
    dec.flush(&mut |f| frames.push(f)).unwrap();
    assert_eq!(frames.len(), 25);
    // Continue after flush with a different resolution (new keyframe).
    for (i, au) in big.iter().take(25).enumerate() {
        dec.decode(&packet(Codec::H264, au.clone(), 100 + i), true, &mut |f| frames.push(f))
            .unwrap();
    }
    // And switch back without flushing.
    for (i, au) in small.iter().take(25).enumerate() {
        dec.decode(&packet(Codec::H264, au.clone(), 200 + i), true, &mut |f| frames.push(f))
            .unwrap();
    }
    dec.flush(&mut |f| frames.push(f)).unwrap();
    assert_eq!(frames.len(), 75);
    assert!(frames[..25].iter().all(|f| f.desc.width == 640 && f.desc.height == 360));
    assert!(frames[25..50].iter().all(|f| f.desc.width == 1280 && f.desc.height == 720));
    assert!(frames[50..].iter().all(|f| f.desc.width == 640 && f.desc.height == 360));
}

#[test]
fn ten_bit_is_rejected_clearly() {
    let Some(ff) = setup() else { return };
    let data = h265(&ff, "h265-10bit.hevc", "640x360", 5, "yuv420p10le");
    let aus = split_annexb(&data, Codec::H265);
    let mut dec = NvdecDecoder::open(&params(Codec::H265), 0, 4).unwrap();
    let mut err = None;
    for (i, au) in aus.into_iter().enumerate() {
        if let Err(e) = dec.decode(&packet(Codec::H265, au, i), true, &mut |_| {}) {
            err = Some(e);
            break;
        }
    }
    match err {
        Some(Error::Decode(msg)) => assert!(msg.contains("10-bit"), "{msg}"),
        other => panic!("expected a decode error, got {other:?}"),
    }
}

#[test]
fn frame_buffers_are_pooled() {
    let Some(ff) = setup() else { return };
    let aus = split_annexb(&h264(&ff, "h264-360p.h264", "640x360", 25, &["-bf", "0"]), Codec::H264);
    let mut dec = NvdecDecoder::open(&params(Codec::H264), 0, 4).unwrap();
    let mut ptrs = std::collections::HashSet::new();
    for (i, au) in aus.into_iter().enumerate() {
        dec.decode(&packet(Codec::H264, au, i), true, &mut |f| {
            if let FrameData::Device(m) = &f.data {
                ptrs.insert(m.device_ptr());
            }
        })
        .unwrap();
    }
    // Frames are dropped right away, so very few distinct buffers are used.
    assert!(ptrs.len() <= 2, "{} distinct buffers", ptrs.len());
}

/// NVDEC throughput on a generated 1080p H.264 clip:
/// `cargo test --release -p camtensor-decode --test nvdec -- --ignored --nocapture`
#[test]
#[ignore = "benchmark"]
fn bench_nvdec_1080p_h264() {
    let Some(ff) = setup() else { return };
    let data = h264(&ff, "h264-1080p.h264", "1920x1080", 250, &["-bf", "0", "-preset", "veryfast"]);
    let aus = split_annexb(&data, Codec::H264);
    let mut dec = NvdecDecoder::open(&params(Codec::H264), 0, 8).unwrap();
    let mut n = 0usize;
    // Warm-up pass, then timed passes (flush restarts the parser).
    for (i, au) in aus.iter().take(25).enumerate() {
        dec.decode(&packet(Codec::H264, au.clone(), i), true, &mut |_| {}).unwrap();
    }
    dec.flush(&mut |_| {}).unwrap();
    let t = std::time::Instant::now();
    for pass in 0..4 {
        for (i, au) in aus.iter().enumerate() {
            dec.decode(&packet(Codec::H264, au.clone(), pass * 1000 + i), true, &mut |_| n += 1)
                .unwrap();
        }
        dec.flush(&mut |_| n += 1).unwrap();
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "NVDEC 1080p H.264: {n} frames in {s:.2} s = {:.0} fps (decode + copy to pooled buffer)",
        n as f64 / s
    );
    assert_eq!(n, 4 * aus.len());
}
