//! Decode-throughput benchmark for the CPU decoders.
//!
//! ```text
//! cargo run --release -p camtensor-decode --example decode_bench -- \
//!     clip.h264 [--threads N] [--backend ffmpeg|jpeg] [--loops N]
//! ```
//!
//! The codec is taken from the extension: `.h264`/`.264` (Annex B H.264),
//! `.h265`/`.hevc`/`.265` (Annex B H.265), `.mjpeg`/`.mjpg` (concatenated
//! JPEGs). The file is split into packets up front; only decoding is timed.
//! Set `CAMTENSOR_FFMPEG_DIR` if libavcodec is not on the library path.

#[path = "../tests/common/annexb.rs"]
mod annexb;

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use camtensor_core::{Codec, EncodedPacket, StreamParams};
use camtensor_decode::Decoder;

fn usage() -> ! {
    eprintln!(
        "usage: decode_bench <file.h264|.h265|.hevc|.mjpeg> [--threads N] \
         [--backend ffmpeg|jpeg] [--loops N]"
    );
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else { usage() };
    let (mut threads, mut backend, mut loops) = (0usize, None::<String>, 3usize);
    while let Some(a) = args.next() {
        let v = args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--threads" => threads = v.parse().unwrap_or_else(|_| usage()),
            "--backend" => backend = Some(v),
            "--loops" => loops = v.parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let codec = match ext.as_str() {
        "h264" | "264" => Codec::H264,
        "h265" | "hevc" | "265" => Codec::H265,
        "mjpeg" | "mjpg" => Codec::Mjpeg,
        _ => usage(),
    };
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });

    // Split into packets; timestamps are synthetic (decode order, 25 fps).
    let units: Vec<(Vec<u8>, bool)> = match codec {
        Codec::Mjpeg => annexb::split_jpegs(&data).into_iter().map(|j| (j, true)).collect(),
        Codec::H264 | Codec::H265 => {
            let std =
                if codec == Codec::H264 { annexb::Standard::H264 } else { annexb::Standard::H265 };
            annexb::split_access_units(&data, std)
                .into_iter()
                .map(|a| (a.data, a.keyframe))
                .collect()
        }
    };
    let stream_id: Arc<str> = Arc::from("bench");
    let packets: Vec<EncodedPacket> = units
        .into_iter()
        .enumerate()
        .map(|(i, (d, key))| EncodedPacket {
            stream_id: stream_id.clone(),
            codec,
            data: Bytes::from(d),
            pts: i as i64 * 3600,
            keyframe: key,
            seq: i as u64,
            recv_time_ns: 0,
            capture_time_ns: None,
            loss: false,
        })
        .collect();
    if packets.is_empty() {
        eprintln!("{path}: no packets found");
        std::process::exit(1);
    }

    let params = StreamParams { codec, width: 0, height: 0, extradata: Bytes::new(), fps: None };
    let backend = backend.unwrap_or_else(|| "ffmpeg".into());
    let mut dec: Box<dyn Decoder> = match backend.as_str() {
        #[cfg(feature = "ffmpeg")]
        "ffmpeg" => {
            match camtensor_decode::ffmpeg::library_info() {
                Ok(info) => println!("{info}"),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1)
                }
            }
            Box::new(camtensor_decode::ffmpeg::FfmpegDecoder::open(&params, threads).unwrap())
        }
        #[cfg(feature = "jpeg")]
        "jpeg" if codec == Codec::Mjpeg => Box::new(camtensor_decode::jpeg::JpegDecoder::new()),
        other => {
            eprintln!("backend '{other}' unavailable for {codec} (check enabled features)");
            std::process::exit(2)
        }
    };

    let mut size = None;
    for run in 0..=loops {
        let mut frames = 0usize;
        let t = Instant::now();
        let mut sink = |f: camtensor_core::Frame| {
            frames += 1;
            size.get_or_insert((f.desc.width, f.desc.height, f.desc.format));
        };
        for p in &packets {
            dec.decode(p, true, &mut sink).unwrap();
        }
        dec.flush(&mut sink).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let label = if run == 0 { "warm-up".to_string() } else { format!("run {run}") };
        println!(
            "{label}: {frames} frames in {:.3} s = {:.1} fps ({backend}, threads={threads}, {:?})",
            secs,
            frames as f64 / secs,
            size.unwrap_or((0, 0, camtensor_core::PixelFormat::Gray8)),
        );
    }
}
