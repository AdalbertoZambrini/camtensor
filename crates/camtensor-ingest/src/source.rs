//! The RTSP session loop: connect, demux, convert, deliver, reconnect.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt;
use retina::client::{
    Credentials, PlayOptions, Session, SessionOptions, SetupOptions, TcpTransportOptions,
    Transport as RetinaTransport, UdpTransportOptions,
};
use retina::codec::{CodecItem, FrameFormat, ParameterSetInsertion, ParametersRef, h26x::Framing};
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use url::Url;

use camtensor_core::time::now_ns;
use camtensor_core::{Codec, EncodedPacket, Transport};

use crate::convert::{ParamState, PtsMapper, SrClock, codec_from_encoding, to_pts};
use crate::{Shared, SourceConfig, SourceEvent};

/// Timeout of each RTSP request phase (connect + DESCRIBE, SETUP, PLAY).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// First reconnect delay.
const MIN_BACKOFF_MS: u64 = 500;

/// Why a session ended without error.
enum End {
    /// The receiver was dropped.
    ReceiverClosed,
}

/// State that survives reconnects.
#[derive(Debug, Default)]
struct Persistent {
    seq: u64,
    pts: PtsMapper,
    /// Backpressure: drop non-key packets until the next keyframe.
    drop_until_key: bool,
    /// Mark the next delivered packet with `loss = true`.
    pending_loss: bool,
}

pub(crate) async fn run(
    cfg: SourceConfig,
    tx: mpsc::Sender<SourceEvent>,
    shared: Arc<Shared>,
    mut stop: watch::Receiver<bool>,
) {
    let mut st = Persistent::default();
    let mut backoff = Backoff::new(cfg.max_reconnect_ms);
    loop {
        if *stop.borrow() {
            break;
        }
        let mut delivered_any = false;
        let result = tokio::select! {
            r = run_session(&cfg, &tx, &shared, &mut st, &mut delivered_any) => r,
            _ = stop.wait_for(|s| *s) => break,
        };
        shared.connected.store(false, Ordering::Relaxed);
        let reason = match result {
            Ok(End::ReceiverClosed) => break,
            Err(reason) => reason,
        };
        tracing::warn!(stream = %cfg.stream_id, "rtsp session failed: {reason}");
        *shared.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason.clone());
        if delivered_any {
            backoff.reset();
        }
        let sent = tokio::select! {
            r = tx.send(SourceEvent::Disconnected { reason }) => r.is_ok(),
            _ = stop.wait_for(|s| *s) => break,
        };
        if !sent {
            break;
        }
        let delay = backoff.next_delay();
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            _ = stop.wait_for(|s| *s) => break,
        }
        shared.reconnects.fetch_add(1, Ordering::Relaxed);
    }
    shared.connected.store(false, Ordering::Relaxed);
}

/// Splits credentials out of the URL; explicit config values win.
fn prepare_url(cfg: &SourceConfig) -> Result<(Url, Option<Credentials>), String> {
    let mut url = Url::parse(&cfg.url).map_err(|e| format!("invalid url: {e}"))?;
    let url_user = (!url.username().is_empty()).then(|| percent_decode(url.username()));
    let url_pass = url.password().map(percent_decode);
    let username = cfg.username.clone().or(url_user);
    let password = cfg.password.clone().or(url_pass);
    let _ = url.set_username("");
    let _ = url.set_password(None);
    let creds =
        username.map(|username| Credentials { username, password: password.unwrap_or_default() });
    Ok((url, creds))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok();
            if let Some(v) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

async fn timed<T, E: std::fmt::Display>(
    what: &str,
    fut: impl Future<Output = Result<T, E>>,
) -> Result<T, String> {
    match tokio::time::timeout(REQUEST_TIMEOUT, fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(format!("{what}: {e}")),
        Err(_) => Err(format!("{what}: timed out after {} s", REQUEST_TIMEOUT.as_secs())),
    }
}

async fn run_session(
    cfg: &SourceConfig,
    tx: &mpsc::Sender<SourceEvent>,
    shared: &Shared,
    st: &mut Persistent,
    delivered_any: &mut bool,
) -> Result<End, String> {
    let (url, creds) = prepare_url(cfg)?;
    let opts = SessionOptions::default()
        .creds(creds)
        .user_agent(concat!("camtensor/", env!("CARGO_PKG_VERSION")).to_owned());
    let mut session = timed("DESCRIBE", Session::describe(url, opts)).await?;

    let (idx, codec) = session
        .streams()
        .iter()
        .enumerate()
        .find_map(|(i, s)| {
            (s.media() == "video")
                .then(|| codec_from_encoding(s.encoding_name()))
                .flatten()
                .map(|c| (i, c))
        })
        .ok_or_else(|| {
            let found: Vec<String> = session
                .streams()
                .iter()
                .map(|s| format!("{}/{}", s.media(), s.encoding_name()))
                .collect();
            format!("no H.264/H.265/JPEG video stream (found: {})", found.join(", "))
        })?;
    let transport = match cfg.transport {
        Transport::Tcp => RetinaTransport::Tcp(TcpTransportOptions::default()),
        Transport::Udp => RetinaTransport::Udp(UdpTransportOptions::default()),
    };
    let format = FrameFormat {
        h26x_framing: Framing::FourByteLength,
        parameter_set_insertion: ParameterSetInsertion::Never,
        ..FrameFormat::default()
    };
    timed(
        "SETUP",
        session.setup(idx, SetupOptions::default().transport(transport).frame_format(format)),
    )
    .await?;
    let clock_rate = session.streams()[idx].clock_rate_hz();
    let sdp_fps = session.streams()[idx].framerate().map(f64::from);
    let playing = timed("PLAY", session.play(PlayOptions::default())).await?;
    let mut demuxed = playing.demuxed().map_err(|e| format!("demux: {e}"))?;

    let mut params = current_params(codec, demuxed.streams()[idx].parameters(), sdp_fps)
        .unwrap_or_else(|| ParamState::unknown(codec, sdp_fps));
    let mut sr = SrClock::default();
    st.pts.new_session();
    if st.seq > 0 {
        // Discontinuity: the decoder must restart from a keyframe.
        st.pending_loss = true;
    }
    shared.connected.store(true, Ordering::Relaxed);
    if tx.send(SourceEvent::Connected(params.params.clone())).await.is_err() {
        return Ok(End::ReceiverClosed);
    }

    let mut deadline = Instant::now() + cfg.data_timeout;
    loop {
        let item = match tokio::time::timeout_at(deadline, demuxed.next()).await {
            Err(_) => {
                return Err(format!("no data for {} s", cfg.data_timeout.as_secs_f32()));
            }
            Ok(None) => return Err("stream ended by server".into()),
            Ok(Some(Err(e))) => return Err(e.to_string()),
            Ok(Some(Ok(item))) => item,
        };
        match item {
            CodecItem::VideoFrame(frame) if frame.stream_id() == idx => {
                deadline = Instant::now() + cfg.data_timeout;
                if (frame.has_new_parameters() || !params.complete())
                    && let Some(new) =
                        current_params(codec, demuxed.streams()[idx].parameters(), sdp_fps)
                {
                    let changed = new.params != params.params;
                    params = new;
                    if changed
                        && tx.send(SourceEvent::ParamsChanged(params.params.clone())).await.is_err()
                    {
                        return Ok(End::ReceiverClosed);
                    }
                }
                let recv = now_ns();
                let keyframe = frame.is_random_access_point() || codec == Codec::Mjpeg;
                let loss = frame.loss() > 0;
                let ts = frame.timestamp();
                let pts = st.pts.map(to_pts(ts.elapsed(), clock_rate), recv);
                let capture_time_ns = sr.capture_time_ns(ts.timestamp());
                let data = match params.payload(keyframe, frame.into_data()) {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::debug!(stream = %cfg.stream_id, "{e}");
                        st.pending_loss = true;
                        continue;
                    }
                };
                let pkt = EncodedPacket {
                    stream_id: cfg.stream_id.clone(),
                    codec,
                    data,
                    pts,
                    keyframe,
                    seq: st.seq,
                    recv_time_ns: recv,
                    capture_time_ns,
                    loss,
                };
                st.seq += 1;
                match deliver(tx, shared, st, pkt) {
                    Delivery::Sent => *delivered_any = true,
                    Delivery::Dropped => {}
                    Delivery::Closed => return Ok(End::ReceiverClosed),
                }
            }
            CodecItem::Rtcp(rtcp) if rtcp.stream_id() == idx => {
                // `rtp_timestamp()` is `Some` iff the compound packet starts
                // with a sender report; it is already extended to 64 bits.
                if let Some(rtp) = rtcp.rtp_timestamp()
                    && let Some(Ok(Some(report))) = rtcp.pkts().next().map(|p| p.as_sender_report())
                {
                    sr.on_sender_report(report.ntp_timestamp().0, rtp.timestamp(), clock_rate);
                }
            }
            _ => {}
        }
    }
}

fn current_params(
    codec: Codec,
    p: Option<ParametersRef<'_>>,
    sdp_fps: Option<f64>,
) -> Option<ParamState> {
    match p? {
        ParametersRef::Video(vp) => Some(ParamState::from_video_parameters(codec, vp, sdp_fps)),
        _ => None,
    }
}

enum Delivery {
    Sent,
    Dropped,
    Closed,
}

/// Non-blocking delivery with keyframe-aligned dropping under backpressure.
fn deliver(
    tx: &mpsc::Sender<SourceEvent>,
    shared: &Shared,
    st: &mut Persistent,
    mut pkt: EncodedPacket,
) -> Delivery {
    shared.packets.fetch_add(1, Ordering::Relaxed);
    shared.bytes.fetch_add(pkt.data.len() as u64, Ordering::Relaxed);
    if st.drop_until_key && !pkt.keyframe {
        shared.dropped.fetch_add(1, Ordering::Relaxed);
        return Delivery::Dropped;
    }
    if st.pending_loss {
        pkt.loss = true;
    }
    match tx.try_send(SourceEvent::Packet(pkt)) {
        Ok(()) => {
            st.drop_until_key = false;
            st.pending_loss = false;
            Delivery::Sent
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            shared.dropped.fetch_add(1, Ordering::Relaxed);
            st.drop_until_key = true;
            st.pending_loss = true;
            Delivery::Dropped
        }
        Err(mpsc::error::TrySendError::Closed(_)) => Delivery::Closed,
    }
}

/// Exponential back-off with +-20 % jitter.
#[derive(Debug)]
struct Backoff {
    current_ms: u64,
    max_ms: u64,
    rng: u64,
}

impl Backoff {
    fn new(max_ms: u64) -> Self {
        let max_ms = max_ms.max(MIN_BACKOFF_MS);
        Self { current_ms: MIN_BACKOFF_MS, max_ms, rng: now_ns() | 1 }
    }

    fn reset(&mut self) {
        self.current_ms = MIN_BACKOFF_MS;
    }

    fn next_delay(&mut self) -> Duration {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let jitter = (self.rng % 401) as i64 - 200; // -200..=200 per mille
        let base = self.current_ms as i64;
        let ms = (base + base * jitter / 1000).max(1) as u64;
        self.current_ms = (self.current_ms * 2).min(self.max_ms);
        Duration::from_millis(ms.min(self.max_ms + self.max_ms / 5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn cfg(url: &str) -> SourceConfig {
        SourceConfig { url: url.into(), ..SourceConfig::new("t", "rtsp://x") }
    }

    #[test]
    fn url_credentials() {
        let (url, creds) = prepare_url(&cfg("rtsp://adm%40in:p%3Ass@cam:554/s1")).unwrap();
        assert_eq!(url.as_str(), "rtsp://cam:554/s1");
        let c = creds.unwrap();
        assert_eq!((c.username.as_str(), c.password.as_str()), ("adm@in", "p:ss"));
        let mut c2 = cfg("rtsp://u:p@cam/s");
        c2.password = Some("override".into());
        let (_, creds) = prepare_url(&c2).unwrap();
        assert_eq!(creds.unwrap().password, "override");
        let (_, creds) = prepare_url(&cfg("rtsp://cam/s")).unwrap();
        assert!(creds.is_none());
        assert!(prepare_url(&cfg("not a url")).is_err());
        assert_eq!(percent_decode("a%2"), "a%2");
        assert_eq!(percent_decode("%zz%41"), "%zzA");
    }

    #[test]
    fn backoff_grows_with_jitter_and_caps() {
        let mut b = Backoff::new(4000);
        let d: Vec<u64> = (0..6).map(|_| b.next_delay().as_millis() as u64).collect();
        let bases = [500u64, 1000, 2000, 4000, 4000, 4000];
        for (d, base) in d.iter().zip(bases) {
            assert!(*d >= base * 8 / 10 && *d <= base * 12 / 10, "{d} vs {base}");
        }
        b.reset();
        assert!(b.next_delay().as_millis() <= 600);
    }

    fn pkt(seq: u64, key: bool) -> EncodedPacket {
        EncodedPacket {
            stream_id: Arc::from("t"),
            codec: Codec::H264,
            data: Bytes::from_static(&[0, 0, 0, 1, 0x41]),
            pts: 0,
            keyframe: key,
            seq,
            recv_time_ns: 0,
            capture_time_ns: None,
            loss: false,
        }
    }

    #[test]
    fn backpressure_drops_until_keyframe() {
        let (tx, mut rx) = mpsc::channel(2);
        let shared = Shared::default();
        let mut st = Persistent::default();
        let keys = [true, false, false, false, false, true, false];
        let res: Vec<bool> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                if i == 4 {
                    // Consumer frees space before packet 4 (non-key: still dropped).
                    let _ = rx.try_recv();
                    let _ = rx.try_recv();
                }
                matches!(deliver(&tx, &shared, &mut st, pkt(i as u64, k)), Delivery::Sent)
            })
            .collect();
        assert_eq!(res, [true, true, false, false, false, true, true]);
        assert_eq!(shared.dropped.load(Ordering::Relaxed), 3);
        assert_eq!(shared.packets.load(Ordering::Relaxed), 7);
        let got: Vec<(u64, bool)> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|e| match e {
                SourceEvent::Packet(p) => (p.seq, p.loss),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(got, [(5, true), (6, false)]);
        drop(rx);
        assert!(matches!(deliver(&tx, &shared, &mut st, pkt(7, true)), Delivery::Closed));
    }
}
