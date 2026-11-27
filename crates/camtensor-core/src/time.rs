//! Time helpers. All wall-clock timestamps in camtensor are nanoseconds since
//! the Unix epoch (`u64`), all media timestamps are 90 kHz ticks (`i64`).

use std::time::{SystemTime, UNIX_EPOCH};

/// Media clock rate used for `pts` values (RTP video clock).
pub const PTS_CLOCK_HZ: i64 = 90_000;

/// Current wall-clock time in nanoseconds since the Unix epoch.
pub fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

/// Converts an NTP 64-bit timestamp (seconds since 1900 in the high 32 bits,
/// fraction in the low 32 bits) to nanoseconds since the Unix epoch.
pub fn ntp_to_unix_ns(ntp: u64) -> u64 {
    const NTP_UNIX_OFFSET_SECS: u64 = 2_208_988_800;
    let secs = ntp >> 32;
    let frac = ntp & 0xffff_ffff;
    let nanos = (frac * 1_000_000_000) >> 32;
    secs.saturating_sub(NTP_UNIX_OFFSET_SECS) * 1_000_000_000 + nanos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntp_conversion() {
        // 2020-01-01T00:00:00Z = 1577836800 unix = 3786825600 ntp
        let ntp = 3_786_825_600u64 << 32 | (1u64 << 31); // + 0.5 s
        assert_eq!(ntp_to_unix_ns(ntp), 1_577_836_800_500_000_000);
    }
}
