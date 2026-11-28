//! End-to-end tests against MediaMTX with FFmpeg publishers. Skipped (with a
//! message) when the executables are not available.

mod common;

use std::time::Duration;

use camtensor_bitstream::{annexb, h264, h265, jpeg};
use camtensor_core::{Codec, EncodedPacket, StreamParams, Transport};
use camtensor_ingest::{RtspSource, SourceConfig, SourceEvent, SourceHandle};
use tokio::sync::mpsc;

use common::*;

const WAIT: Duration = Duration::from_secs(30);

fn spawn(url: &str, transport: Transport) -> (SourceHandle, mpsc::Receiver<SourceEvent>) {
    let (tx, rx) = mpsc::channel(256);
    let mut cfg = SourceConfig::new("test", url);
    cfg.transport = transport;
    cfg.max_reconnect_ms = 1000;
    (RtspSource::spawn(&tokio::runtime::Handle::current(), cfg, tx), rx)
}

#[derive(Default)]
struct Collected {
    params: Option<StreamParams>,
    connects: usize,
    disconnects: Vec<String>,
    packets: Vec<EncodedPacket>,
}

/// Receives events until `n` packets arrived.
async fn collect(rx: &mut mpsc::Receiver<SourceEvent>, n: usize, c: &mut Collected) {
    let target = c.packets.len() + n;
    let res = tokio::time::timeout(WAIT, async {
        while c.packets.len() < target {
            match rx.recv().await.expect("source ended") {
                SourceEvent::Connected(p) => {
                    c.connects += 1;
                    c.params = Some(p);
                }
                SourceEvent::ParamsChanged(p) => c.params = Some(p),
                SourceEvent::Packet(p) => c.packets.push(p),
                SourceEvent::Disconnected { reason } => {
                    eprintln!("disconnected: {reason}");
                    c.disconnects.push(reason)
                }
                SourceEvent::Frame(_) => {}
            }
        }
    })
    .await;
    assert!(res.is_ok(), "timed out: {} packets, disconnects {:?}", c.packets.len(), c.disconnects);
}

fn check_common(c: &Collected, codec: Codec, w: u32, h: u32) {
    let p = c.params.as_ref().expect("no params");
    assert_eq!(p.codec, codec);
    assert_eq!((p.width, p.height), (w, h), "{p:?}");
    let fps = p.fps.unwrap_or(0.0);
    println!("{codec}: {}x{} fps={:?} extradata={}B", p.width, p.height, p.fps, p.extradata.len());
    if codec != Codec::Mjpeg {
        assert!((fps - 25.0).abs() < 0.01, "fps {fps}");
        assert!(annexb::starts_with_start_code(&p.extradata));
    }
    for w in c.packets.windows(2) {
        assert_eq!(w[1].seq, w[0].seq + 1, "seq gap without drops");
        assert!(w[1].pts > w[0].pts, "pts not increasing: {} -> {}", w[0].pts, w[1].pts);
    }
    // 25 fps -> 3600 ticks per frame on average.
    let first = &c.packets[0];
    let last = c.packets.last().unwrap();
    let avg = (last.pts - first.pts) as f64 / (c.packets.len() - 1) as f64;
    assert!((avg - 3600.0).abs() < 400.0, "avg pts delta {avg}");
    assert!(c.packets.iter().all(|p| p.codec == codec && &*p.stream_id == "test"));
    let with_capture = c.packets.iter().filter(|p| p.capture_time_ns.is_some()).count();
    println!("{codec}: {with_capture}/{} packets with RTCP capture time", c.packets.len());
    for p in c.packets.iter().filter(|p| p.capture_time_ns.is_some()) {
        let cap = p.capture_time_ns.unwrap() as i128;
        let skew = (p.recv_time_ns as i128 - cap).abs();
        assert!(skew < 5_000_000_000, "capture time off by {} ms", skew / 1_000_000);
    }
}

fn nal_types(codec: Codec, data: &[u8]) -> Vec<u8> {
    annexb::nal_units(data)
        .filter_map(|n| match codec {
            Codec::H264 => h264::nal_unit_type(n),
            _ => h265::nal_unit_type(n),
        })
        .collect()
}

fn check_h26x(c: &Collected, codec: Codec) {
    let (ps, is_irap): (&[u8], fn(u8) -> bool) = match codec {
        Codec::H264 => (&[h264::NAL_SPS, h264::NAL_PPS], |t| t == h264::NAL_IDR),
        _ => (&[h265::NAL_VPS, h265::NAL_SPS, h265::NAL_PPS], h265::is_irap),
    };
    let mut keys = 0;
    for p in &c.packets {
        assert!(annexb::starts_with_start_code(&p.data), "packet without start code");
        let types = nal_types(codec, &p.data);
        assert_eq!(
            p.keyframe,
            types.iter().any(|&t| is_irap(t)),
            "keyframe flag vs NAL types {types:?}"
        );
        if p.keyframe {
            keys += 1;
            for t in ps {
                assert!(types.contains(t), "keyframe lacks parameter set {t}: {types:?}");
            }
        } else {
            assert!(!types.iter().any(|t| ps.contains(t)), "parameter sets in a non-key packet");
        }
    }
    // GOP 25 at 25 fps.
    assert!(keys >= 2, "only {keys} keyframes");
    let key_seqs: Vec<u64> = c.packets.iter().filter(|p| p.keyframe).map(|p| p.seq).collect();
    assert!(key_seqs.windows(2).all(|w| w[1] - w[0] == 25), "keyframe interval {key_seqs:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn h264_tcp() {
    let Some((mtx_exe, ff)) = tools("h264_tcp") else { return };
    let mtx = MediaMtx::start(&mtx_exe, 18600);
    let _publisher = publish(&ff, &mtx.url("cam"), "1280x720", H264);
    mtx.wait_ready("cam", WAIT);
    let (handle, mut rx) = spawn(&mtx.url("cam"), Transport::Tcp);
    let mut c = Collected::default();
    collect(&mut rx, 75, &mut c).await;
    check_common(&c, Codec::H264, 1280, 720);
    check_h26x(&c, Codec::H264);
    let st = handle.stats();
    assert!(st.connected && st.packets >= 75 && st.dropped == 0, "{st:?}");
    assert_eq!(c.connects, 1);
    handle.join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn h264_udp() {
    let Some((mtx_exe, ff)) = tools("h264_udp") else { return };
    let mtx = MediaMtx::start(&mtx_exe, 18610);
    let _publisher = publish(&ff, &mtx.url("cam"), "640x360", H264);
    mtx.wait_ready("cam", WAIT);
    let (handle, mut rx) = spawn(&mtx.url("cam"), Transport::Udp);
    let mut c = Collected::default();
    collect(&mut rx, 50, &mut c).await;
    // Loopback UDP should be lossless; keep only checks robust to rare loss.
    assert_eq!(c.params.as_ref().map(|p| (p.width, p.height)), Some((640, 360)));
    assert!(c.packets.iter().all(|p| annexb::starts_with_start_code(&p.data)));
    assert!(c.packets.iter().any(|p| p.keyframe));
    handle.join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn h265_tcp() {
    let Some((mtx_exe, ff)) = tools("h265_tcp") else { return };
    let mtx = MediaMtx::start(&mtx_exe, 18620);
    let _publisher = publish(&ff, &mtx.url("cam"), "1280x720", H265);
    mtx.wait_ready("cam", WAIT);
    let (handle, mut rx) = spawn(&mtx.url("cam"), Transport::Tcp);
    let mut c = Collected::default();
    collect(&mut rx, 75, &mut c).await;
    check_common(&c, Codec::H265, 1280, 720);
    check_h26x(&c, Codec::H265);
    handle.join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mjpeg_tcp() {
    let Some((mtx_exe, ff)) = tools("mjpeg_tcp") else { return };
    let mtx = MediaMtx::start(&mtx_exe, 18630);
    let _publisher = publish(&ff, &mtx.url("cam"), "640x480", MJPEG);
    mtx.wait_ready("cam", WAIT);
    let (handle, mut rx) = spawn(&mtx.url("cam"), Transport::Tcp);
    let mut c = Collected::default();
    collect(&mut rx, 50, &mut c).await;
    check_common(&c, Codec::Mjpeg, 640, 480);
    for p in &c.packets {
        assert!(p.keyframe);
        assert!(
            p.data.starts_with(&[0xff, 0xd8]) && p.data.ends_with(&[0xff, 0xd9]),
            "not a full JPEG"
        );
        let hdr = jpeg::FrameHeader::parse(&p.data).unwrap();
        assert_eq!((hdr.width, hdr.height, hdr.chroma_format()), (640, 480, Some(1)));
    }
    handle.join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reconnects_after_publisher_restart() {
    let Some((mtx_exe, ff)) = tools("reconnects_after_publisher_restart") else { return };
    let mtx = MediaMtx::start(&mtx_exe, 18640);
    let mut publisher = publish(&ff, &mtx.url("cam"), "640x360", H264);
    mtx.wait_ready("cam", WAIT);
    let (handle, mut rx) = spawn(&mtx.url("cam"), Transport::Tcp);
    let mut c = Collected::default();
    collect(&mut rx, 30, &mut c).await;
    let before = c.packets.len();
    let last_before = c.packets.last().unwrap().clone();

    publisher.kill();
    // Wait for the failure to be reported.
    let reason = tokio::time::timeout(WAIT, async {
        loop {
            match rx.recv().await.expect("source ended") {
                SourceEvent::Disconnected { reason } => break reason,
                SourceEvent::Packet(p) => c.packets.push(p),
                _ => {}
            }
        }
    })
    .await
    .expect("no Disconnected event");
    println!("disconnected: {reason}");
    assert!(!handle.stats().connected || handle.stats().reconnects > 0);

    let _publisher = publish(&ff, &mtx.url("cam"), "640x360", H264);
    let mut c2 = Collected::default();
    collect(&mut rx, 30, &mut c2).await;
    assert!(c2.connects >= 1, "no Connected after restart");
    let first = &c2.packets[0];
    assert!(first.loss, "first packet after reconnect must flag loss");
    assert!(first.seq > last_before.seq, "seq must continue across reconnects");
    assert!(first.pts > c.packets.last().unwrap().pts, "pts must stay monotonic");
    for w in c2.packets.windows(2) {
        assert!(w[1].pts > w[0].pts);
    }
    let st = handle.stats();
    assert!(st.reconnects >= 1 && st.last_error.is_some(), "{st:?}");
    println!("reconnect ok: {before} packets before, stats {st:?}");
    handle.join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backpressure_drops_to_next_keyframe() {
    let Some((mtx_exe, ff)) = tools("backpressure_drops_to_next_keyframe") else { return };
    let mtx = MediaMtx::start(&mtx_exe, 18650);
    let _publisher = publish(&ff, &mtx.url("cam"), "640x360", H264);
    mtx.wait_ready("cam", WAIT);
    let (tx, mut rx) = mpsc::channel(4);
    let h = RtspSource::spawn(
        &tokio::runtime::Handle::current(),
        SourceConfig::new("test", mtx.url("cam")),
        tx,
    );
    // Do not read for 3 s: the channel fills up and packets are dropped.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(h.stats().dropped > 0);
    let mut seen_after_full = Vec::new();
    let res = tokio::time::timeout(WAIT, async {
        while seen_after_full.len() < 30 {
            if let Some(SourceEvent::Packet(p)) = rx.recv().await {
                seen_after_full.push(p);
            }
        }
    })
    .await;
    assert!(res.is_ok());
    // After the queued packets, delivery resumes at a keyframe flagged as loss.
    let resume = seen_after_full.iter().position(|p| p.loss).expect("no loss-flagged packet");
    assert!(seen_after_full[resume].keyframe);
    assert!(seen_after_full[resume].seq > seen_after_full[resume - 1].seq + 1);
    h.join().await;
}
