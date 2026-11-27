//! Conversion of retina's output to camtensor's packet contract: stream
//! parameters, Annex B framing, 90 kHz timestamps and RTCP wall-clock
//! mapping. Kept free of I/O so it can be unit-tested.

use bytes::Bytes;
use camtensor_bitstream::annexb;
use camtensor_core::time::{PTS_CLOCK_HZ, ntp_to_unix_ns};
use camtensor_core::{Codec, StreamParams};
use retina::codec::{VideoParameters, VideoParametersCodec};

/// Maps an RTSP encoding name (lowercase) to a codec.
pub(crate) fn codec_from_encoding(name: &str) -> Option<Codec> {
    match name {
        "h264" => Some(Codec::H264),
        "h265" => Some(Codec::H265),
        "jpeg" => Some(Codec::Mjpeg),
        _ => None,
    }
}

/// Stream parameters plus the NAL length size of retina's frames.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParamState {
    pub params: StreamParams,
    pub length_size: usize,
}

impl ParamState {
    /// Parameters before anything is known (only the codec and maybe the
    /// SDP frame rate).
    pub fn unknown(codec: Codec, sdp_fps: Option<f64>) -> Self {
        Self {
            params: StreamParams {
                codec,
                width: 0,
                height: 0,
                extradata: Bytes::new(),
                fps: sdp_fps,
            },
            length_size: 4,
        }
    }

    /// `true` once the dimensions (and parameter sets for H.26x) are known.
    pub fn complete(&self) -> bool {
        self.params.width > 0
            && self.params.height > 0
            && (self.params.codec == Codec::Mjpeg || !self.params.extradata.is_empty())
    }

    /// Builds parameters from retina's `VideoParameters`.
    pub fn from_video_parameters(codec: Codec, vp: &VideoParameters, sdp_fps: Option<f64>) -> Self {
        let mut st = Self::from_parts(
            codec,
            vp.pixel_dimensions(),
            vp.extra_data(),
            vp.frame_rate(),
            sdp_fps,
        );
        if matches!(codec, Codec::H264 | Codec::H265) && st.params.extradata.is_empty() {
            // Fall back on the raw parameter sets if the record did not parse.
            let mut out = Vec::new();
            match vp.codec_params() {
                VideoParametersCodec::H264 { sps, pps } => {
                    annexb::push_nal(&mut out, sps);
                    annexb::push_nal(&mut out, pps);
                }
                VideoParametersCodec::H265 { vps, sps, pps } => {
                    annexb::push_nal(&mut out, vps);
                    annexb::push_nal(&mut out, sps);
                    annexb::push_nal(&mut out, pps);
                }
                _ => {}
            }
            st.params.extradata = Bytes::from(out);
            st.fill_from_sps();
        }
        st
    }

    /// Builds parameters from their parts. `extra_data` is an `avcC` /
    /// `hvcC` record for H.26x (retina's length-prefixed framing).
    /// `frame_rate` is a frame duration in seconds as `(num, den)`.
    pub fn from_parts(
        codec: Codec,
        (width, height): (u32, u32),
        extra_data: &[u8],
        frame_rate: Option<(u32, u32)>,
        sdp_fps: Option<f64>,
    ) -> Self {
        let mut st = Self::unknown(codec, None);
        st.params.width = width;
        st.params.height = height;
        let converted = match codec {
            Codec::H264 => annexb::avcc_config_to_annexb(extra_data).ok(),
            Codec::H265 => annexb::hvcc_config_to_annexb(extra_data).ok(),
            Codec::Mjpeg => None,
        };
        if let Some((ps, length_size)) = converted {
            st.params.extradata = Bytes::from(ps);
            st.length_size = length_size;
        }
        st.params.fps = frame_rate
            .filter(|&(num, den)| num > 0 && den > 0)
            .map(|(num, den)| f64::from(den) / f64::from(num));
        st.fill_from_sps();
        if st.params.fps.is_none() {
            st.params.fps = sdp_fps.filter(|f| f.is_finite() && *f > 0.0);
        }
        st
    }

    /// Fills missing dimensions / frame rate from the SPS.
    fn fill_from_sps(&mut self) {
        if self.params.codec == Codec::Mjpeg || self.params.extradata.is_empty() {
            return;
        }
        if let Ok(info) =
            camtensor_bitstream::probe_params(self.params.codec, &self.params.extradata)
        {
            if self.params.width == 0 || self.params.height == 0 {
                self.params.width = info.width;
                self.params.height = info.height;
            }
            if self.params.fps.is_none() {
                self.params.fps = info.fps;
            }
        }
    }

    /// Converts retina frame data to the packet payload: Annex B with the
    /// parameter sets prepended on keyframes for H.26x, unchanged JPEG for
    /// MJPEG.
    pub fn payload(&self, keyframe: bool, data: Vec<u8>) -> Result<Bytes, String> {
        match self.params.codec {
            Codec::Mjpeg => Ok(Bytes::from(data)),
            Codec::H264 | Codec::H265 => {
                let extra = if keyframe { self.params.extradata.len() } else { 0 };
                let mut out = Vec::with_capacity(extra + data.len() + 16);
                if keyframe {
                    out.extend_from_slice(&self.params.extradata);
                }
                annexb::avcc_to_annexb(&data, self.length_size, &mut out)
                    .map_err(|e| format!("bad {} frame: {e}", self.params.codec))?;
                Ok(Bytes::from(out))
            }
        }
    }
}

/// Rescales a timestamp from `clock_rate` to 90 kHz.
pub(crate) fn to_pts(ticks: i64, clock_rate: u32) -> i64 {
    if clock_rate as i64 == PTS_CLOCK_HZ || clock_rate == 0 {
        return ticks;
    }
    (i128::from(ticks) * i128::from(PTS_CLOCK_HZ) / i128::from(clock_rate)) as i64
}

/// Keeps `pts` continuous and monotonic across reconnects: each new
/// session is offset so it continues from the previous one, advanced by the
/// wall-clock time spent reconnecting.
#[derive(Debug, Default)]
pub(crate) struct PtsMapper {
    offset: i64,
    session_started: bool,
    max_pts: Option<i64>,
    last_wall_ns: u64,
}

impl PtsMapper {
    /// Marks the start of a new session.
    pub fn new_session(&mut self) {
        self.session_started = false;
    }

    /// Maps a session-relative 90 kHz timestamp to the output pts.
    pub fn map(&mut self, raw: i64, now_ns: u64) -> i64 {
        if !self.session_started {
            self.session_started = true;
            self.offset = match self.max_pts {
                None => -raw,
                Some(max) => {
                    let gap_ns = now_ns.saturating_sub(self.last_wall_ns);
                    let gap =
                        (i128::from(gap_ns) * i128::from(PTS_CLOCK_HZ) / 1_000_000_000) as i64;
                    max + gap.max(1) - raw
                }
            };
        }
        let pts = raw + self.offset;
        self.max_pts = Some(self.max_pts.map_or(pts, |m| m.max(pts)));
        self.last_wall_ns = now_ns;
        pts
    }
}

/// RTP to wall-clock mapping from the latest RTCP sender report.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SrClock {
    /// `(ntp timestamp, extended rtp timestamp, clock rate)`.
    last: Option<(u64, i64, u32)>,
}

impl SrClock {
    /// Records a sender report; NTP timestamp 0 means "unknown" and is ignored.
    pub fn on_sender_report(&mut self, ntp: u64, rtp: i64, clock_rate: u32) {
        if ntp != 0 && clock_rate > 0 {
            self.last = Some((ntp, rtp, clock_rate));
        }
    }

    /// Wall-clock capture time (ns since Unix epoch) of an extended RTP
    /// timestamp, `None` before the first sender report.
    pub fn capture_time_ns(&self, rtp: i64) -> Option<u64> {
        let (ntp, sr_rtp, rate) = self.last?;
        let base = i128::from(ntp_to_unix_ns(ntp));
        let delta = i128::from(rtp - sr_rtp) * 1_000_000_000 / i128::from(rate);
        u64::try_from(base + delta).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPS: [u8; 28] = [
        0x67, 0x64, 0x00, 0x1f, 0xac, 0xd9, 0x40, 0x50, 0x05, 0xbb, 0x01, 0x6e, 0x02, 0x02, 0x02,
        0x80, 0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x19, 0x07, 0x8c, 0x18, 0xcb,
    ];
    const PPS: [u8; 6] = [0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];

    fn avcc() -> Vec<u8> {
        let mut rec = vec![1, 0x64, 0x00, 0x1f, 0xff, 0xe1];
        rec.extend((SPS.len() as u16).to_be_bytes());
        rec.extend(SPS);
        rec.push(1);
        rec.extend((PPS.len() as u16).to_be_bytes());
        rec.extend(PPS);
        rec
    }

    #[test]
    fn encoding_names() {
        assert_eq!(codec_from_encoding("h264"), Some(Codec::H264));
        assert_eq!(codec_from_encoding("h265"), Some(Codec::H265));
        assert_eq!(codec_from_encoding("jpeg"), Some(Codec::Mjpeg));
        assert_eq!(codec_from_encoding("mpeg4-generic"), None);
    }

    #[test]
    fn params_from_avcc() {
        let st = ParamState::from_parts(Codec::H264, (1280, 720), &avcc(), Some((1, 25)), None);
        assert_eq!((st.params.width, st.params.height), (1280, 720));
        assert_eq!(st.params.fps, Some(25.0));
        assert_eq!(st.length_size, 4);
        let mut expect = vec![0, 0, 0, 1];
        expect.extend(SPS);
        expect.extend([0, 0, 0, 1]);
        expect.extend(PPS);
        assert_eq!(&st.params.extradata[..], &expect[..]);
        assert!(st.complete());

        // Unknown dimensions / frame rate come from the SPS.
        let st = ParamState::from_parts(Codec::H264, (0, 0), &avcc(), None, Some(10.0));
        assert_eq!((st.params.width, st.params.height, st.params.fps), (1280, 720, Some(25.0)));

        // MJPEG: SDP frame rate, no extradata.
        let st = ParamState::from_parts(Codec::Mjpeg, (640, 480), &[], None, Some(15.0));
        assert_eq!(st.params.fps, Some(15.0));
        assert!(st.params.extradata.is_empty() && st.complete());
        assert!(!ParamState::unknown(Codec::H264, None).complete());
    }

    #[test]
    fn payload_conversion() {
        let st = ParamState::from_parts(Codec::H264, (1280, 720), &avcc(), None, None);
        let frame = vec![0, 0, 0, 3, 0x65, 0x88, 0x84, 0, 0, 0, 2, 0x06, 0x05];
        let key = st.payload(true, frame.clone()).unwrap();
        assert!(key.starts_with(&st.params.extradata));
        assert_eq!(
            &key[st.params.extradata.len()..],
            &[0, 0, 0, 1, 0x65, 0x88, 0x84, 0, 0, 0, 1, 0x06, 0x05]
        );
        assert!(camtensor_bitstream::h264::is_keyframe(&key));
        let p = st.payload(false, vec![0, 0, 0, 2, 0x41, 0x9a]).unwrap();
        assert_eq!(&p[..], &[0, 0, 0, 1, 0x41, 0x9a]);
        assert!(st.payload(false, vec![0, 0, 0, 9, 0x41]).is_err());
        let mj = ParamState::unknown(Codec::Mjpeg, None);
        assert_eq!(
            &mj.payload(true, vec![0xff, 0xd8, 0xff, 0xd9]).unwrap()[..],
            &[0xff, 0xd8, 0xff, 0xd9]
        );
    }

    #[test]
    fn pts_rescale() {
        assert_eq!(to_pts(12345, 90_000), 12345);
        assert_eq!(to_pts(1000, 1000), 90_000);
        assert_eq!(to_pts(-8000, 8000), -90_000);
    }

    #[test]
    fn pts_continuity_across_sessions() {
        let mut m = PtsMapper::default();
        m.new_session();
        assert_eq!(m.map(1_000_000, 0), 0);
        assert_eq!(m.map(1_003_600, 40_000_000), 3600);
        // B-frame style reordering is preserved within a session.
        assert_eq!(m.map(1_001_800, 50_000_000), 1800);
        // Reconnect 2 s later; the new session starts at an unrelated value.
        m.new_session();
        assert_eq!(m.map(55, 2_050_000_000), 3600 + 180_000);
        assert_eq!(m.map(55 + 3600, 2_090_000_000), 3600 + 180_000 + 3600);
        // Even with no wall-clock gap pts still advances.
        m.new_session();
        let last = 3600 + 180_000 + 3600;
        assert_eq!(m.map(0, 2_090_000_000), last + 1);
    }

    #[test]
    fn sender_report_mapping() {
        let mut c = SrClock::default();
        assert_eq!(c.capture_time_ns(0), None);
        // 2020-01-01T00:00:00Z + 0.5 s at RTP 90_000.
        let ntp = (3_786_825_600u64 << 32) | (1u64 << 31);
        c.on_sender_report(ntp, 90_000, 90_000);
        assert_eq!(c.capture_time_ns(90_000), Some(1_577_836_800_500_000_000));
        assert_eq!(c.capture_time_ns(180_000), Some(1_577_836_801_500_000_000));
        assert_eq!(c.capture_time_ns(45_000), Some(1_577_836_800_000_000_000));
        // NTP 0 is ignored.
        c.on_sender_report(0, 0, 90_000);
        assert_eq!(c.capture_time_ns(90_000), Some(1_577_836_800_500_000_000));
    }
}
