//! Compressed-domain motion gating.
//!
//! A [`Gate`] sits between the RTSP source and the decoder and decides which
//! packets are worth decoding, using only cheap compressed-domain signals:
//! picture type (key / non-key) and packet size.
//!
//! For H.264 / H.265 in [`GatingMode::Auto`], a static scene produces small
//! predicted pictures; motion produces larger ones. While idle, only
//! keyframes are decoded and the non-key packets since the last keyframe are
//! kept in a buffer. When packet sizes rise above `threshold` times the
//! learnt baseline for `trigger_frames` consecutive packets, the buffered
//! packets are replayed into the decoder (only the newest one is emitted to
//! consumers) and every packet is decoded until activity stops for
//! `hold_ms`.
//!
//! # Camera configuration
//!
//! Size-based detection needs **VBR** (variable bitrate) encoding: a CBR
//! encoder pads or re-quantises pictures to keep sizes constant and hides
//! motion. Keep the GOP (keyframe interval) at **2 s or less**: it bounds the
//! idle decode rate, the replay cost on activation and the buffer size.
//!
//! # Determinism
//!
//! The gate has no clock of its own: every decision depends only on the
//! packet sequence and the `now_ns` values passed to [`Gate::push`].

use std::collections::VecDeque;

use camtensor_core::{Codec, EncodedPacket, GatingConfig, GatingMode};

/// Default cap on buffered non-key packets while idle.
pub const MAX_GOP_PACKETS: usize = 300;
/// Default cap on buffered bytes while idle.
pub const MAX_GOP_BYTES: usize = 32 << 20;

/// EMA weight used to learn the baseline during warm-up.
const WARMUP_ALPHA: f64 = 0.05;
/// EMA weight used to track the baseline afterwards (quiet packets only).
const TRACK_ALPHA: f64 = 0.01;
/// 90 kHz media clock.
const PTS_HZ: f64 = 90_000.0;

/// Gate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateState {
    /// Learning the baseline; everything is decoded.
    Warmup,
    /// Static scene: keyframes only (or `idle_fps` for MJPEG).
    Idle,
    /// Activity: everything is decoded.
    Active,
}

/// Gate counters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GateStats {
    /// Packets handed to the decoder (`emit` true or false).
    pub decoded: u64,
    /// Packets that were never handed to the decoder.
    pub skipped: u64,
    /// Idle to Active transitions.
    pub activations: u64,
    /// Current baseline packet size in bytes (0 until learnt).
    pub baseline: f64,
    /// Score of the last scored packet (size relative to the baseline).
    pub last_score: f32,
}

/// Motion gate, see the [module documentation](self).
#[derive(Debug)]
pub struct Gate {
    cfg: GatingConfig,
    codec: Codec,
    state: GateState,
    stats: GateStats,
    /// Drop non-key packets until the next keyframe (start, loss).
    need_key: bool,
    /// Packets seen during warm-up.
    warm_count: u32,
    /// EMA of packet sizes.
    baseline: Option<f64>,
    /// Consecutive packets above the threshold.
    consecutive: u32,
    /// `now_ns` of the last packet above the threshold (or forced).
    last_active_ns: u64,
    /// `force_active` deadline.
    forced_until: u64,
    /// Non-key packets since the last decoded packet (H.26x idle).
    gop: VecDeque<EncodedPacket>,
    gop_bytes: usize,
    /// The buffer overflowed: no replay possible until the next keyframe.
    gop_broken: bool,
    max_gop_packets: usize,
    max_gop_bytes: usize,
    /// MJPEG idle rate limiter: next pts due, last pts seen.
    next_emit_pts: Option<i64>,
    last_pts: Option<i64>,
}

impl Gate {
    /// Creates a gate with the default buffer limits.
    pub fn new(cfg: GatingConfig, codec: Codec) -> Self {
        Self::with_gop_limits(cfg, codec, MAX_GOP_PACKETS, MAX_GOP_BYTES)
    }

    /// Creates a gate with custom idle buffer limits.
    pub fn with_gop_limits(
        cfg: GatingConfig,
        codec: Codec,
        max_packets: usize,
        max_bytes: usize,
    ) -> Self {
        let state = match cfg.mode {
            GatingMode::Always => GateState::Active,
            GatingMode::Keyframes => GateState::Idle,
            GatingMode::Auto => GateState::Warmup,
        };
        Self {
            cfg,
            codec,
            state,
            stats: GateStats::default(),
            need_key: true,
            warm_count: 0,
            baseline: None,
            consecutive: 0,
            last_active_ns: 0,
            forced_until: 0,
            gop: VecDeque::new(),
            gop_bytes: 0,
            gop_broken: false,
            max_gop_packets: max_packets.max(1),
            max_gop_bytes: max_bytes.max(1),
            next_emit_pts: None,
            last_pts: None,
        }
    }

    /// Current state.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Counters.
    pub fn stats(&self) -> GateStats {
        GateStats { baseline: self.baseline.unwrap_or(0.0), ..self.stats }
    }

    /// Number of packets currently buffered while idle.
    pub fn buffered(&self) -> usize {
        self.gop.len()
    }

    /// Forces full decoding until `until_ns` (for example on an external
    /// trigger). Takes effect on the next [`push`](Self::push); in
    /// [`GatingMode::Keyframes`] it has no effect.
    pub fn force_active(&mut self, until_ns: u64) {
        self.forced_until = self.forced_until.max(until_ns);
    }

    /// Pushes a packet; appends to `out` the packets to feed to the decoder,
    /// each with an `emit` flag (`false` = decode only to rebuild reference
    /// pictures, do not deliver the decoded frame).
    pub fn push(&mut self, pkt: EncodedPacket, now_ns: u64, out: &mut Vec<(EncodedPacket, bool)>) {
        if pkt.loss {
            self.need_key = true;
            self.drop_gop();
        }
        if self.need_key && !pkt.keyframe {
            self.stats.skipped += 1;
            return;
        }
        if pkt.keyframe {
            self.need_key = false;
        }
        match self.cfg.mode {
            GatingMode::Always => self.emit(pkt, true, out),
            GatingMode::Keyframes => {
                if self.codec == Codec::Mjpeg {
                    if self.rate_limit_due(pkt.pts) {
                        self.emit(pkt, true, out);
                    } else {
                        self.stats.skipped += 1;
                    }
                } else if pkt.keyframe {
                    self.emit(pkt, true, out);
                } else {
                    self.stats.skipped += 1;
                }
            }
            GatingMode::Auto if self.codec == Codec::Mjpeg => {
                self.push_auto_mjpeg(pkt, now_ns, out)
            }
            GatingMode::Auto => self.push_auto_h26x(pkt, now_ns, out),
        }
    }

    fn push_auto_h26x(
        &mut self,
        pkt: EncodedPacket,
        now_ns: u64,
        out: &mut Vec<(EncodedPacket, bool)>,
    ) {
        match self.state {
            GateState::Warmup => {
                if !pkt.keyframe {
                    self.learn(pkt.data.len(), WARMUP_ALPHA);
                }
                self.emit(pkt, true, out);
                self.warm_count += 1;
                if self.warm_count >= self.cfg.warmup_frames {
                    self.state = GateState::Idle;
                }
            }
            GateState::Idle => {
                if pkt.keyframe {
                    // The keyframe resets the reference chain: older buffered
                    // packets are no longer needed.
                    self.drop_gop();
                    self.gop_broken = false;
                    self.emit(pkt, true, out);
                    if self.is_forced(now_ns) {
                        self.activate(now_ns, out);
                    }
                    return;
                }
                let active = self.score(pkt.data.len());
                if !self.gop_broken {
                    self.gop_bytes += pkt.data.len();
                    self.gop.push_back(pkt);
                    if self.gop.len() > self.max_gop_packets || self.gop_bytes > self.max_gop_bytes
                    {
                        self.drop_gop();
                        self.gop_broken = true;
                    }
                } else {
                    self.stats.skipped += 1;
                }
                if active {
                    self.consecutive += 1;
                }
                if self.consecutive >= self.cfg.trigger_frames.max(1) || self.is_forced(now_ns) {
                    self.activate(now_ns, out);
                }
            }
            GateState::Active => {
                if !pkt.keyframe && self.score(pkt.data.len()) {
                    self.last_active_ns = now_ns;
                }
                self.emit(pkt, true, out);
                self.maybe_deactivate(now_ns);
            }
        }
    }

    fn push_auto_mjpeg(
        &mut self,
        pkt: EncodedPacket,
        now_ns: u64,
        out: &mut Vec<(EncodedPacket, bool)>,
    ) {
        match self.state {
            GateState::Warmup => {
                self.learn(pkt.data.len(), WARMUP_ALPHA);
                self.track_pts(pkt.pts);
                self.emit(pkt, true, out);
                self.warm_count += 1;
                if self.warm_count >= self.cfg.warmup_frames {
                    self.state = GateState::Idle;
                }
            }
            GateState::Idle => {
                if self.score(pkt.data.len()) {
                    self.consecutive += 1;
                }
                if self.consecutive >= self.cfg.trigger_frames.max(1) || self.is_forced(now_ns) {
                    self.track_pts(pkt.pts);
                    self.emit(pkt, true, out);
                    self.activate(now_ns, out);
                } else if self.rate_limit_due(pkt.pts) {
                    self.emit(pkt, true, out);
                } else {
                    self.stats.skipped += 1;
                }
            }
            GateState::Active => {
                if self.score(pkt.data.len()) {
                    self.last_active_ns = now_ns;
                }
                self.track_pts(pkt.pts);
                self.emit(pkt, true, out);
                self.maybe_deactivate(now_ns);
            }
        }
    }

    /// Scores a packet against the baseline and updates the baseline with
    /// quiet packets. Returns `true` if the packet counts as activity.
    fn score(&mut self, size: usize) -> bool {
        let size = size as f64;
        let Some(base) = self.baseline else {
            self.baseline = Some(size.max(1.0));
            self.stats.last_score = 1.0;
            self.consecutive = 0;
            return false;
        };
        let base = base.max(1.0);
        let threshold = f64::from(self.cfg.threshold);
        let (score, active) = if self.codec == Codec::Mjpeg {
            // Intra-only: any change of the picture changes its size, up or down.
            let dev = (size - base).abs() / base;
            (dev + 1.0, dev > threshold - 1.0)
        } else {
            let s = size / base;
            (s, s > threshold)
        };
        self.stats.last_score = score as f32;
        if !active {
            self.consecutive = 0;
            self.learn(size as usize, TRACK_ALPHA);
        }
        active
    }

    fn learn(&mut self, size: usize, alpha: f64) {
        let size = size as f64;
        self.baseline = Some(match self.baseline {
            None => size.max(1.0),
            Some(b) => (b + alpha * (size - b)).max(1.0),
        });
    }

    fn is_forced(&self, now_ns: u64) -> bool {
        now_ns < self.forced_until
    }

    fn activate(&mut self, now_ns: u64, out: &mut Vec<(EncodedPacket, bool)>) {
        self.state = GateState::Active;
        self.stats.activations += 1;
        self.last_active_ns = now_ns;
        self.consecutive = 0;
        if self.gop_broken {
            // The buffer overflowed: the reference chain cannot be rebuilt.
            self.need_key = true;
            self.gop_broken = false;
        }
        let n = self.gop.len();
        for (i, p) in self.gop.drain(..).enumerate() {
            out.push((p, i + 1 == n));
        }
        self.stats.decoded += n as u64;
        self.gop_bytes = 0;
    }

    fn maybe_deactivate(&mut self, now_ns: u64) {
        let hold_end =
            self.last_active_ns.saturating_add(self.cfg.hold_ms.saturating_mul(1_000_000));
        if now_ns >= hold_end && !self.is_forced(now_ns) {
            self.state = GateState::Idle;
            self.consecutive = 0;
            // Everything up to here was decoded: the chain is intact.
            self.gop_broken = false;
            // MJPEG: restart the idle rate limiter from the last emitted frame.
        }
    }

    fn drop_gop(&mut self) {
        self.stats.skipped += self.gop.len() as u64;
        self.gop.clear();
        self.gop_bytes = 0;
    }

    fn emit(&mut self, pkt: EncodedPacket, emit: bool, out: &mut Vec<(EncodedPacket, bool)>) {
        self.stats.decoded += 1;
        out.push((pkt, emit));
    }

    /// Records an emitted MJPEG frame's pts for the idle rate limiter.
    fn track_pts(&mut self, pts: i64) {
        let interval = self.idle_interval();
        self.last_pts = Some(pts);
        self.next_emit_pts = interval.map(|i| pts + i);
    }

    /// MJPEG rate limiter: emits at most `idle_fps` frames per second of
    /// pts, without drift (`next = previous due + interval`), resyncing after
    /// gaps or pts going backwards. A non-positive `idle_fps` disables it.
    fn rate_limit_due(&mut self, pts: i64) -> bool {
        let Some(interval) = self.idle_interval() else {
            return true;
        };
        let backwards = self.last_pts.is_some_and(|last| pts < last);
        self.last_pts = Some(pts);
        match self.next_emit_pts {
            Some(due) if !backwards && pts < due => false,
            Some(due) if !backwards && pts - due < interval => {
                self.next_emit_pts = Some(due + interval);
                true
            }
            _ => {
                self.next_emit_pts = Some(pts + interval);
                true
            }
        }
    }

    fn idle_interval(&self) -> Option<i64> {
        let fps = self.cfg.idle_fps;
        (fps.is_finite() && fps > 0.0).then(|| ((PTS_HZ / fps).round() as i64).max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::sync::Arc;

    const MS: u64 = 1_000_000;
    /// 25 fps.
    const FRAME_NS: u64 = 40 * MS;
    const FRAME_PTS: i64 = 3600;

    #[derive(Clone, Copy)]
    struct P {
        seq: u64,
        size: usize,
        key: bool,
        loss: bool,
    }

    fn k(seq: u64, size: usize) -> P {
        P { seq, size, key: true, loss: false }
    }
    fn p(seq: u64, size: usize) -> P {
        P { seq, size, key: false, loss: false }
    }
    fn lost(mut x: P) -> P {
        x.loss = true;
        x
    }

    fn pkt(x: P, codec: Codec) -> EncodedPacket {
        EncodedPacket {
            stream_id: Arc::from("t"),
            codec,
            data: Bytes::from(vec![0u8; x.size]),
            pts: x.seq as i64 * FRAME_PTS,
            keyframe: x.key,
            seq: x.seq,
            recv_time_ns: x.seq * FRAME_NS,
            capture_time_ns: None,
            loss: x.loss,
        }
    }

    /// Runs packets through the gate at 25 fps; returns `(seq, emit)` pairs.
    fn run(g: &mut Gate, codec: Codec, pkts: &[P]) -> Vec<(u64, bool)> {
        let mut out = Vec::new();
        for &x in pkts {
            g.push(pkt(x, codec), x.seq * FRAME_NS, &mut out);
        }
        out.into_iter().map(|(p, e)| (p.seq, e)).collect()
    }

    fn cfg(mode: GatingMode) -> GatingConfig {
        GatingConfig {
            mode,
            threshold: 2.0,
            trigger_frames: 2,
            hold_ms: 200,
            warmup_frames: 5,
            idle_fps: 1.0,
        }
    }

    /// Keyframe every `gop` packets, `size` bytes per P frame, from seq `from`.
    fn stream(from: u64, n: u64, gop: u64, size: usize) -> Vec<P> {
        (from..from + n).map(|s| if s % gop == 0 { k(s, size * 10) } else { p(s, size) }).collect()
    }

    fn all(seqs: impl IntoIterator<Item = u64>) -> Vec<(u64, bool)> {
        seqs.into_iter().map(|s| (s, true)).collect()
    }

    #[test]
    fn always_waits_for_first_keyframe() {
        let mut g = Gate::new(cfg(GatingMode::Always), Codec::H264);
        let out = run(&mut g, Codec::H264, &[p(0, 10), p(1, 10), k(2, 100), p(3, 10), p(4, 10)]);
        assert_eq!(out, all([2, 3, 4]));
        assert_eq!(g.stats().decoded, 3);
        assert_eq!(g.stats().skipped, 2);
        assert_eq!(g.state(), GateState::Active);
    }

    #[test]
    fn loss_drops_until_keyframe_in_all_modes() {
        let seq = [k(0, 100), p(1, 10), lost(p(2, 10)), p(3, 10), k(4, 100), p(5, 10)];
        let mut g = Gate::new(cfg(GatingMode::Always), Codec::H264);
        assert_eq!(run(&mut g, Codec::H264, &seq), all([0, 1, 4, 5]));
        // A lost keyframe flag is fine: the keyframe itself restarts decoding.
        let seq2 = [k(0, 100), p(1, 10), lost(k(2, 100)), p(3, 10)];
        let mut g = Gate::new(cfg(GatingMode::Always), Codec::H264);
        assert_eq!(run(&mut g, Codec::H264, &seq2), all([0, 1, 2, 3]));
        let mut g = Gate::new(cfg(GatingMode::Keyframes), Codec::H264);
        assert_eq!(run(&mut g, Codec::H264, &seq), all([0, 4]));
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        assert_eq!(run(&mut g, Codec::H264, &seq), all([0, 1, 4, 5]));
    }

    #[test]
    fn keyframes_mode_h26x() {
        let mut g = Gate::new(cfg(GatingMode::Keyframes), Codec::H265);
        let out = run(&mut g, Codec::H265, &stream(0, 30, 10, 50));
        assert_eq!(out, all([0, 10, 20]));
        assert_eq!(g.stats().skipped, 27);
        g.force_active(u64::MAX); // no effect in this mode
        assert_eq!(run(&mut g, Codec::H265, &[p(30, 50), k(31, 500)]), all([31]));
    }

    #[test]
    fn keyframes_mode_mjpeg_rate_limit() {
        let frames: Vec<P> = (0..101).map(|s| k(s, 1000)).collect();
        let mut g = Gate::new(cfg(GatingMode::Keyframes), Codec::Mjpeg);
        assert_eq!(run(&mut g, Codec::Mjpeg, &frames), all([0, 25, 50, 75, 100]));
        let mut c = cfg(GatingMode::Keyframes);
        c.idle_fps = 5.0;
        let mut g = Gate::new(c, Codec::Mjpeg);
        assert_eq!(run(&mut g, Codec::Mjpeg, &frames[..21]), all([0, 5, 10, 15, 20]));
        // Non-positive idle_fps disables the limiter.
        c.idle_fps = 0.0;
        let mut g = Gate::new(c, Codec::Mjpeg);
        assert_eq!(run(&mut g, Codec::Mjpeg, &frames[..4]).len(), 4);
    }

    #[test]
    fn mjpeg_rate_limit_resyncs() {
        let mut g = Gate::new(cfg(GatingMode::Keyframes), Codec::Mjpeg);
        let mut out = Vec::new();
        let mut push = |g: &mut Gate, pts: i64| {
            let mut x = pkt(k(0, 100), Codec::Mjpeg);
            x.pts = pts;
            out.clear();
            g.push(x, 0, &mut out);
            !out.is_empty()
        };
        assert!(push(&mut g, 1_000_000));
        assert!(!push(&mut g, 1_045_000));
        // Jittery frame slightly late: due advances by exactly one interval.
        assert!(push(&mut g, 1_090_100));
        assert!(!push(&mut g, 1_179_000));
        assert!(push(&mut g, 1_180_000));
        // Long gap: resync instead of bursting.
        assert!(push(&mut g, 2_000_000));
        assert!(!push(&mut g, 2_050_000));
        // pts going backwards (reconnect) resyncs immediately.
        assert!(push(&mut g, 10));
        assert!(!push(&mut g, 3610));
    }

    /// Warm-up then a static scene: only keyframes are decoded.
    #[test]
    fn auto_static_scene() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        assert_eq!(g.state(), GateState::Warmup);
        let out = run(&mut g, Codec::H264, &stream(0, 35, 10, 100));
        assert_eq!(out, all([0, 1, 2, 3, 4, 10, 20, 30]));
        assert_eq!(g.state(), GateState::Idle);
        let st = g.stats();
        assert_eq!(st.decoded, 8);
        assert_eq!(st.activations, 0);
        assert_eq!(st.baseline, 100.0);
        assert_eq!(st.last_score, 1.0);
        assert_eq!(g.buffered(), 4); // 31..=34
        assert_eq!(st.skipped, 35 - 8 - 4);
    }

    /// Motion burst: buffered packets replayed, newest emitted, then
    /// pass-through, then back to idle after `hold_ms`.
    #[test]
    fn auto_motion_burst_and_hold() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        let mut seq = stream(0, 14, 10, 100); // warm-up 0..=4, idle 5..=13 (key at 10)
        seq.push(p(14, 300)); // active #1 (score 3 > 2)
        seq.push(p(15, 300)); // active #2 -> trigger
        seq.push(p(16, 100)); // pass-through, quiet
        let out = run(&mut g, Codec::H264, &seq);
        let mut expect = all([0, 1, 2, 3, 4, 10]);
        expect.extend([(11, false), (12, false), (13, false), (14, false), (15, true), (16, true)]);
        assert_eq!(out, expect);
        assert_eq!(g.state(), GateState::Active);
        assert_eq!(g.stats().activations, 1);
        assert_eq!(g.buffered(), 0);

        // Last activity at seq 15 (600 ms); hold 200 ms -> idle once
        // now >= 800 ms (seq 20). Seq 20 itself is still decoded.
        let out = run(&mut g, Codec::H264, &[p(17, 100), p(18, 100), p(19, 100), k(20, 1000)]);
        assert_eq!(out, all([17, 18, 19, 20]));
        assert_eq!(g.state(), GateState::Idle);
        let out = run(&mut g, Codec::H264, &[p(21, 100), p(22, 100)]);
        assert!(out.is_empty());
        assert_eq!(g.buffered(), 2);
    }

    #[test]
    fn auto_activity_extends_hold() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        let mut seq = stream(0, 6, 100, 100);
        seq.extend([p(6, 500), p(7, 500)]); // trigger at 7 (280 ms)
        seq.extend([p(8, 100), p(9, 100), p(10, 100), p(11, 500)]); // activity at 11 (440 ms)
        seq.extend((12..17).map(|s| p(s, 100))); // idle again at 16 (640 ms)
        seq.push(p(17, 100));
        let out = run(&mut g, Codec::H264, &seq);
        let mut expect = all([0, 1, 2, 3, 4]);
        expect.push((5, false));
        expect.push((6, false));
        expect.extend(all(7..=16));
        assert_eq!(out, expect);
        assert_eq!(g.state(), GateState::Idle);
        assert_eq!(g.buffered(), 1);
    }

    #[test]
    fn auto_single_spike_does_not_trigger() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        let mut seq = stream(0, 8, 100, 100);
        seq.extend([p(8, 1000), p(9, 100), p(10, 1000), p(11, 100)]);
        let out = run(&mut g, Codec::H264, &seq);
        assert_eq!(out, all(0..5));
        assert_eq!(g.stats().activations, 0);
        // Spikes do not pollute the baseline.
        assert_eq!(g.stats().baseline, 100.0);
    }

    #[test]
    fn auto_baseline_adapts_slowly() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        let mut seq = stream(0, 5, 1000, 100);
        // Gradual brightness/noise change: 150 bytes is below 2x.
        seq.extend((5..105).map(|s| p(s, 150)));
        run(&mut g, Codec::H264, &seq);
        let b = g.stats().baseline;
        // 100 updates with alpha 0.01 towards 150: 150 - 50 * 0.99^100
        let expect = 150.0 - 50.0 * 0.99f64.powi(100);
        assert!((b - expect).abs() < 1e-6, "{b} vs {expect}");
    }

    #[test]
    fn auto_warmup_baseline_is_ema() {
        let mut c = cfg(GatingMode::Auto);
        c.warmup_frames = 4;
        let mut g = Gate::new(c, Codec::H264);
        run(&mut g, Codec::H264, &[k(0, 5000), p(1, 100), p(2, 200), p(3, 300)]);
        let expect = {
            let b = 100.0f64;
            let b = b + 0.05 * (200.0 - b);
            b + 0.05 * (300.0 - b)
        };
        assert!((g.stats().baseline - expect).abs() < 1e-9);
        assert_eq!(g.state(), GateState::Idle);
    }

    #[test]
    fn force_active_replays_and_holds() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        let out = run(&mut g, Codec::H264, &stream(0, 8, 100, 100));
        assert_eq!(out, all(0..5));
        // Force until 400 ms (seq 10).
        g.force_active(400 * MS);
        let out = run(&mut g, Codec::H264, &[p(8, 100)]);
        assert_eq!(out, vec![(5, false), (6, false), (7, false), (8, true)]);
        assert_eq!(g.state(), GateState::Active);
        // Forced until 400 ms, then hold counts from the activation at 320 ms
        // (seq 8): idle at max(400, 520) ms = seq 13.
        let out = run(&mut g, Codec::H264, &(9..15).map(|s| p(s, 100)).collect::<Vec<_>>());
        assert_eq!(out, all(9..=13));
        assert_eq!(g.state(), GateState::Idle);
    }

    #[test]
    fn force_active_on_keyframe_in_idle() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        run(&mut g, Codec::H264, &stream(0, 7, 100, 100));
        g.force_active(u64::MAX);
        let out = run(&mut g, Codec::H264, &[k(7, 1000), p(8, 100)]);
        assert_eq!(out, all([7, 8]));
        assert_eq!(g.stats().activations, 1);
        assert_eq!(g.state(), GateState::Active);
    }

    #[test]
    fn loss_in_idle_clears_buffer() {
        let mut g = Gate::new(cfg(GatingMode::Auto), Codec::H264);
        let mut seq = stream(0, 8, 100, 100); // buffer 5, 6, 7
        seq.push(lost(p(8, 300)));
        seq.push(p(9, 300));
        seq.push(k(10, 1000));
        seq.push(p(11, 300));
        seq.push(p(12, 300)); // trigger: replays only 11, 12
        let out = run(&mut g, Codec::H264, &seq);
        let mut expect = all([0, 1, 2, 3, 4, 10]);
        expect.extend([(11, false), (12, true)]);
        assert_eq!(out, expect);
    }

    #[test]
    fn gop_cap_waits_for_next_keyframe() {
        let mut g = Gate::with_gop_limits(cfg(GatingMode::Auto), Codec::H264, 4, usize::MAX);
        let mut seq = stream(0, 10, 100, 100); // idle 5..=9: 5 packets > cap 4
        assert_eq!(run(&mut g, Codec::H264, &seq), all(0..5));
        assert_eq!(g.buffered(), 0);
        // Motion now cannot be replayed: activation waits for the keyframe.
        seq = vec![p(10, 500), p(11, 500), p(12, 500), k(13, 5000), p(14, 500), p(15, 500)];
        let out = run(&mut g, Codec::H264, &seq);
        assert_eq!(out, all([13, 14, 15]));
        assert_eq!(g.state(), GateState::Active);
        assert_eq!(g.stats().activations, 1);
    }

    #[test]
    fn gop_cap_by_bytes() {
        let mut g = Gate::with_gop_limits(cfg(GatingMode::Auto), Codec::H264, 1000, 250);
        let seq = stream(0, 8, 100, 100); // 5, 6 buffered = 200, 7 -> 300 > 250
        run(&mut g, Codec::H264, &seq);
        assert_eq!(g.buffered(), 0);
        // After the next keyframe buffering resumes.
        let out = run(&mut g, Codec::H264, &[k(8, 1000), p(9, 100)]);
        assert_eq!(out, all([8]));
        assert_eq!(g.buffered(), 1);
    }

    #[test]
    fn warmup_zero_learns_in_idle() {
        let mut c = cfg(GatingMode::Auto);
        c.warmup_frames = 0;
        c.trigger_frames = 1;
        let mut g = Gate::new(c, Codec::H264);
        let out = run(&mut g, Codec::H264, &[k(0, 1000), p(1, 100), p(2, 100), p(3, 500)]);
        assert_eq!(out, vec![(0, true), (1, false), (2, false), (3, true)]);
    }

    #[test]
    fn mjpeg_auto() {
        let mut c = cfg(GatingMode::Auto);
        c.idle_fps = 5.0; // every 5th frame at 25 fps
        let mut g = Gate::new(c, Codec::Mjpeg);
        let mut seq: Vec<P> = (0..20).map(|s| k(s, 10_000)).collect();
        // Motion: sizes change by more than threshold - 1 = 100%.
        seq.extend([k(20, 25_000), k(21, 30_000), k(22, 10_000)]);
        seq.extend((23..40).map(|s| k(s, 10_000)));
        let out = run(&mut g, Codec::Mjpeg, &seq);
        // Warm-up 0..=4, idle every 5th frame counted from the last emitted
        // one (9, 14, 19), trigger at 21 (two consecutive deviations > 100%),
        // active until hold expires at 21 + 5 = 26 (decoded), then idle
        // again counted from 26: 31, 36.
        let mut expect = all(0..5);
        expect.extend(all([9, 14, 19]));
        expect.extend(all(21..=26));
        expect.extend(all([31, 36]));
        assert_eq!(out, expect);
        assert_eq!(g.stats().activations, 1);
        assert_eq!(g.state(), GateState::Idle);
    }

    #[test]
    fn stats_account_for_every_packet() {
        let mut g = Gate::with_gop_limits(cfg(GatingMode::Auto), Codec::H264, 8, usize::MAX);
        let mut seq = stream(0, 200, 25, 100);
        for s in [40u64, 41, 42, 90, 91, 150, 151, 152, 153] {
            seq[s as usize].size = 1000;
        }
        seq[120].loss = true;
        let out = run(&mut g, Codec::H264, &seq);
        let st = g.stats();
        assert_eq!(st.decoded as usize, out.len());
        assert_eq!(st.decoded + st.skipped + g.buffered() as u64, 200);
        // Emitted packets are strictly increasing in seq.
        assert!(out.windows(2).all(|w| w[0].0 < w[1].0));
    }
}
