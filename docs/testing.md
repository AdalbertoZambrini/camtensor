# Test media and camera simulation

camtensor's tests and benchmarks run against real RTSP streams. This page
explains how to get the test media, how to simulate cameras with MediaMTX on
Windows and Linux, and what each clip is for.

| Path | What |
|------|------|
| `testdata/manifest.json` | Every clip: source URL + license + SHA-256 (downloads), exact ffmpeg arguments (generated), expected codec/profile/size/fps/GOP/B-frames/duration, activity intervals (gating clips). Committed. |
| `testdata/media/` | The media itself. Git-ignored, never commit it. |
| `scripts/fetch-test-media.{ps1,sh}` | Download + generate + verify `testdata/media/`. |
| `scripts/sim-cameras.{ps1,sh}` | Start MediaMTX and publish looping cameras; probe; stop. |
| `deploy/mediamtx.yml` | MediaMTX config used by the scripts and docker compose. |
| `deploy/docker-compose.yml` | Linux alternative: MediaMTX + 4 cameras in containers. |

The PowerShell and bash scripts behave identically and share state (stamps,
PID files), so either can be used on the same checkout.

## Requirements

- **ffmpeg + ffprobe** with libx264, libx265 and the mjpeg encoder (any recent
  static build).
- **MediaMTX** (tested with v1.21.4) for the camera simulation.
- **curl** (Windows 10+ ships `curl.exe`).
- bash scripts only: bash 4+, `sha256sum` or `shasum`, and `jq` or `python3`
  to read the manifest.

Tool lookup order: environment variable, then `PATH`; the PowerShell scripts
finally fall back to `C:\mediamtx-ffmpeg\ffmpeg\bin\` and
`C:\mediamtx-ffmpeg\mediamtx\`.

| Variable | Default | Used by |
|----------|---------|---------|
| `CAMTENSOR_FFMPEG` | `ffmpeg` | all scripts |
| `CAMTENSOR_FFPROBE` | `ffprobe` | all scripts |
| `CAMTENSOR_MEDIAMTX` | `mediamtx` | sim-cameras |
| `CAMTENSOR_MEDIAMTX_CONFIG` | `deploy/mediamtx.yml` | sim-cameras |
| `CAMTENSOR_MEDIA_DIR` | `testdata/media` | all scripts |
| `CAMTENSOR_RTSP_URL` | `rtsp://127.0.0.1:8554` | sim-cameras |
| `CAMTENSOR_API_URL` | `http://127.0.0.1:9997` | sim-cameras |
| `PROBE_SECONDS` | `10` | `sim-cameras probe` |

## Fetching the media

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\fetch-test-media.ps1 -Quick   # core set
powershell -ExecutionPolicy Bypass -File scripts\fetch-test-media.ps1          # everything
```

```bash
# Linux
scripts/fetch-test-media.sh --quick
scripts/fetch-test-media.sh
```

| | Files | Download | On disk | Time (8-core desktop) |
|---|---|---|---|---|
| quick | 14 | 31 MB (Big Buck Bunny trailer) | ~85 MB | < 1 min |
| full | 37 | 53 MB (4 clips) | ~350 MB | a few minutes (4K HEVC dominates) |

Options (bash spelling / PowerShell spelling):

- `--quick` / `-Quick`: only entries with `"quick": true`.
- `--force` / `-Force`: re-encode generated files even if up to date.
- `--verify-only` / `-VerifyOnly`: only run the ffprobe checks.
- `--only GLOB` / `-Only GLOB`: e.g. `'gating/*'` (inputs must already exist).
- `--list` / `-List`, `--media-dir DIR` / `-MediaDir DIR`.

The scripts are idempotent:

- Downloads go to `NAME.part`, resume with HTTP ranges, are checked against the
  manifest size and SHA-256, and fall back to a pinned Wayback Machine copy if
  the primary URL fails or serves different bytes.
- Generated files are skipped when they exist, their ffmpeg recipe is unchanged
  (stamp in `testdata/media/.stamps/`) and they pass verification. Change the
  arguments in the manifest and the file is regenerated on the next run.
- Every file is verified with ffprobe: codec, profile, size, pixel format, fps,
  B-frames (`has_b_frames > 0`), duration (+-0.15 s), frame count and keyframe
  count (= ceil(frames / GOP)). ffmpeg logs are kept in `testdata/media/.logs/`.
- Exit code is non-zero if anything failed.

Generated clips are not bit-identical across ffmpeg/x264/x265 versions; tests
must rely on the manifest properties, not on checksums of generated files.

### Adding a clip

Add an entry to `testdata/manifest.json` (`kind: "download"` with `source.urls`,
`size`, `sha256`, license; or `kind: "ffmpeg"` with the argument list, where
`{media:NAME}` refers to an earlier entry) plus the expected `video` block,
then run the fetch script. Only openly licensed media (CC0, CC-BY) is allowed.

## The clips

All synthetic clips are derived from the Big Buck Bunny trailer (25 fps,
full-frame), segment 8-28 s, scaled to the target size. Unless noted: 20 s,
25 fps, fixed GOP (scene-cut detection off, closed GOP / IDR), SPS/PPS repeated
before every H.264 IDR, video only. *quick* marks the `--quick` set.

### Downloaded (CC-BY 3.0, Blender Foundation)

| File | Content | Purpose |
|------|---------|---------|
| `real/bbb-trailer-1080p25-h264.mov` *quick* | BBB trailer, H.264 Main, B-frames, 7 Mb/s, 33 s, AAC | Real content; source of all synthetic clips |
| `real/sintel-trailer-1080p24-h264.mp4` | Sintel trailer, H.264 High, 24 fps, 52 s | Letterboxed (black bars), 24 fps |
| `real/sintel-trailer-854x480-h264.mp4` | Same at 854x480 | Non-mod16 width |
| `real/bbb-1080p30-10s-h265.mp4` | BBB, HEVC Main, 30 fps, 10 s (test-videos.co.uk) | Third-party HEVC; keyframes only at 0 and 8.3 s |

### H.264 / H.265 matrix

| File | Profile / B | GOP | Rate | Purpose |
|------|-------------|-----|------|---------|
| `h264/h264-baseline-640x360-25fps-gop1s-vbr-500k.mp4` *quick* | Constrained Baseline, no B | 1 s | VBR 0.5 Mb/s | Sub-stream, cheapest decode |
| `h264/h264-main-1280x720-30fps-gop2s-vbr-3m.mp4` | Main, B | 2 s | VBR 3 Mb/s | 30 fps |
| `h264/h264-high-1920x1080-25fps-gop2s-vbr-6m.mp4` *quick* | High, B | 2 s | VBR 6 Mb/s | Reference 1080p stream |
| `h264/h264-high-1920x1080-25fps-gop2s-cbr-2m.mp4` | High, B | 2 s | CBR 2 Mb/s (filler) | Strict CBR |
| `h264/h264-main-1920x1080-25fps-gop4s-nob-vbr-4m.mkv` | Main, no B | 4 s | VBR 4 Mb/s | Typical IP camera; MKV |
| `h264/h264-high-2560x1440-25fps-gop2s-vbr-10m.mp4` | High, B | 2 s | VBR 10 Mb/s | 4 MP load |
| `h264/h264-high-3840x2160-25fps-gop2s-vbr-20m.mp4` | High, B | 2 s | VBR 20 Mb/s | 8 MP (4K) load |
| `h265/h265-main-1280x720-25fps-gop2s-vbr-1500k.mp4` | Main, B | 2 s | VBR 1.5 Mb/s | HEVC 720p |
| `h265/h265-main-1920x1080-25fps-gop2s-vbr-4m.mp4` *quick* | Main, B | 2 s | VBR 4 Mb/s | Reference HEVC stream |
| `h265/h265-main-1920x1080-25fps-gop4s-cbr-2m.mp4` | Main, B | 4 s | CBR 2 Mb/s | HEVC strict CBR |
| `h265/h265-main-1920x1080-25fps-gop1s-nob-vbr-3m.mkv` | Main, no B | 1 s | VBR 3 Mb/s | Low-latency profile; MKV |
| `h265/h265-main-2560x1440-25fps-gop2s-vbr-6m.mp4` | Main, B | 2 s | VBR 6 Mb/s | HEVC 4 MP load |
| `h265/h265-main-3840x2160-25fps-gop2s-vbr-12m.mp4` | Main, B | 2 s | VBR 12 Mb/s | HEVC 4K load |

### MJPEG

Encoded with `-huffman default -force_duplicated_matrix 1` so they can be sent
over RTSP with `-c copy` (see [MJPEG over RTSP](#mjpeg-over-rtsp)). 10 s each.

| File | Purpose |
|------|---------|
| `mjpeg/mjpeg-640x360-25fps-q12.mkv` | Low quality / bitrate |
| `mjpeg/mjpeg-1280x720-25fps-q5.mkv` *quick* | Reference MJPEG camera |
| `mjpeg/mjpeg-1920x1080-10fps-q4.mkv` | 1080p at a low frame rate, high quality |

### Edge cases

| File | Purpose |
|------|---------|
| `edge/h265-main10-1920x1080-25fps-gop2s-vbr-4m.mp4` *quick* | 10-bit HEVC (Main 10, `yuv420p10le`): the "unsupported 10-bit" error path. 10 s. |
| `edge/h264-high-642x362-25fps-gop2s-vbr-1m.mp4` *quick* | Non-mod16 size (cropping window, pitch/alignment). 10 s. |
| `edge/h264-main-1280x720-25fps-gop2s-vbr-3m-fullrange.mp4` | Full-range (`yuvj420p`) H.264 like many IP cameras: colour-range handling. 10 s. |

### Unit-test inputs

| File | Purpose |
|------|---------|
| `annexb/h264-baseline-640x360-25fps-gop1s-2s.h264` *quick* | Raw Annex B, no B-frames, 50 frames |
| `annexb/h264-high-1280x720-25fps-gop2s-4s.h264` | Raw Annex B with B-frames (reordering), 100 frames |
| `annexb/h265-main-640x360-25fps-gop1s-2s.hevc` *quick* | Raw HEVC, no B-frames, VPS/SPS/PPS before every IDR |
| `annexb/h265-main-1280x720-25fps-gop2s-4s.hevc` | Raw HEVC with B-frames |
| `frames/bbb-meadow-1920x1080.jpg` *quick* | 4:2:0 JPEG; gating background |
| `frames/bbb-rabbit-1920x1080.jpg` *quick* | 4:2:0 JPEG; gating sprite |
| `frames/bbb-rabbit-1280x720-422.jpg` *quick* | 4:2:2 JPEG (common camera snapshot layout) |
| `frames/bbb-rabbit-642x362-420.jpg` *quick* | Non-mod16 JPEG (partial MCUs) |
| `frames/bbb-rabbit-640x360-444.jpg` | 4:4:4 JPEG |

Raw Annex B has no container timing: ffprobe guesses the H.264 frame rate as
50 fps, so the scripts only check frame and keyframe counts for these files.

### Gating clips

For `GatingConfig` (compressed-domain motion gating): a static scene (a still
of the Big Buck Bunny meadow with light temporal sensor noise,
`noise=alls=3:allf=t`) and a moving object (the rabbit still, scaled) only
during known intervals. 60 s, 1920x1080, 25 fps (MJPEG: 640x360, 15 fps).

| Interval (s, inclusive) | Frames @25 fps | Object | Motion |
|-------------------------|----------------|--------|--------|
| 10-16 | 250-400 | 480x270 | left to right across the frame |
| 25-28 | 625-700 | 128x72 (small, "distant") | right to left |
| 40-50 | 1000-1250 | 256x144 | left to right, slow |

The intervals are also in the manifest (`activity`, with frame indices per
clip). Mean size of non-key packets, idle vs. active (measured):

| File | Rate control | Idle | Active | Ratio |
|------|--------------|------|--------|-------|
| `gating/gating-h264-1080p25-gop2s-vbr.mp4` *quick* | capped CRF 23 (max 6 Mb/s) | 392 B | 1019 B | 2.6 |
| `gating/gating-h265-1080p25-gop2s-vbr.mp4` | capped CRF 26 | 209 B | 918 B | 4.4 |
| `gating/gating-h264-1080p25-gop2s-cbr-4m.mp4` | strict CBR 4 Mb/s, filler NALs | 13.9 kB | 14.4 kB | 1.04 |
| `gating/gating-h265-1080p25-gop4s-cbr-3m.mp4` | strict CBR 3 Mb/s, 4 s GOP | 13.3 kB | 13.9 kB | 1.04 |
| `gating/gating-mjpeg-640x360-15fps-q10.mkv` | MJPEG q10 (intra only) | - | - | - |

VBR clips show a clear signal at the default `threshold: 1.6`. The CBR clips
are the worst case real CBR cameras produce: padding hides the activity in raw
packet sizes, so a detector must look past filler data (or treat such streams
as "always decode"). The MJPEG clip exercises `idle_fps` decimation.

## Simulating cameras

`sim-cameras` starts MediaMTX with `deploy/mediamtx.yml` (RTSP on 8554 TCP+UDP,
API on 127.0.0.1:9997, other protocols off, any path publishable) and one
`ffmpeg -re -stream_loop -1 -i FILE -map 0:v:0 -c copy -f rtsp` publisher per
camera at `rtsp://127.0.0.1:8554/camN`. Files are assigned round-robin.

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 start -Count 4 -Preset mixed
powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 probe
powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 status
powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 stop
# explicit files (comma-separated, relative to testdata/media or absolute)
powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 start -Count 2 -Files 'edge/h265-main10-1920x1080-25fps-gop2s-vbr-4m.mp4,h264/h264-high-3840x2160-25fps-gop2s-vbr-20m.mp4'
```

```bash
# Linux
scripts/sim-cameras.sh start -n 4 -p mixed
scripts/sim-cameras.sh probe
scripts/sim-cameras.sh status
scripts/sim-cameras.sh stop
scripts/sim-cameras.sh start -n 8 gating/gating-h264-1080p25-gop2s-vbr.mp4
```

Presets (all files of `mixed` are in the quick set):

| Preset | Files |
|--------|-------|
| `h264-1080p` | H.264 High 1080p 6 Mb/s |
| `h265-1080p` | HEVC Main 1080p 4 Mb/s |
| `mjpeg-720p` | MJPEG 720p q5 |
| `mixed` (default) | H.264 1080p, HEVC 1080p, MJPEG 720p, H.264 Baseline 360p |
| `gating` | the five gating clips (missing ones are skipped) |

Other options: `--prefix`/`-Prefix` (path names, default `cam`),
`--transport`/`-Transport` (`tcp`/`udp` for publishing), `--mjpeg`/`-Mjpeg`
(`auto` copies MJPEG files from `testdata/media` and re-encodes other MJPEG
sources RFC 2435-compatibly; `copy`; `reencode`), `--external`/`-External`
(publish to an already running MediaMTX, e.g. the docker one). Codecs other
than H.264/H.265/MJPEG are re-encoded to H.264.

`probe` runs ffprobe over RTSP/TCP against every camera in parallel and decodes
`PROBE_SECONDS` (10) of video; a camera fails if no frame decodes. Decoder
messages such as `Could not find ref with POC ...` are expected for HEVC: the
reader joins mid-GOP and the leading frames reference pictures it never
received. They are counted and written to `target/sim-logs/probe-NAME.log`.

Logs (`mediamtx.log`, `camN.log`) and the PID file live in `target/sim-logs/`.
`stop` only kills PIDs that still belong to ffmpeg/MediaMTX. `start` refuses
to run if a MediaMTX API already answers on port 9997.

### Docker (Linux)

```bash
scripts/fetch-test-media.sh --quick
docker compose -f deploy/docker-compose.yml up -d
scripts/sim-cameras.sh probe cam1 cam2 cam3 cam4
docker compose -f deploy/docker-compose.yml down
```

`bluenviron/mediamtx:1.21.4` plus four `linuxserver/ffmpeg` publishers with
host networking (Linux engine required), mounting `testdata/media` read-only.
Change files with `CAM1_FILE`..`CAM4_FILE`. A commented `camtensor` service
is reserved for `camtensor serve`.

## Codec support through MediaMTX v1.21.4

Verified on Windows 11 with MediaMTX v1.21.4 and an ffmpeg 2026-11 build:
published with `-c copy` over RTSP/TCP, read back with
`ffprobe -rtsp_transport tcp` (decoding frames, not only the SDP).

| Stream | Result |
|--------|--------|
| H.264 Baseline / Main / High, 360p to 4K, 25/30 fps, CBR and VBR, MP4/MKV/MOV | OK |
| H.264 642x362 and 854x480, full-range `yuvj420p` | OK (size and range preserved) |
| HEVC Main 720p to 4K, with/without B-frames, MP4/MKV | OK |
| HEVC Main 10 (`yuv420p10le`) | OK, MediaMTX forwards it unchanged |
| MJPEG 640x360, 1280x720, 1920x1080 (files from fetch-test-media) | OK |
| MJPEG from ffmpeg with default options | **Fails**, see below |

MediaMTX logs `RTP packets are too big (1460 > 1440), remuxing them into
smaller ones` for the ffmpeg publishers; this is harmless (ffmpeg's default
RTP payload exceeds the limit derived from MediaMTX's `udpMaxPayloadSize`,
so MediaMTX repacketizes).

### MJPEG over RTSP

M-JPEG over RTSP works with MediaMTX v1.21.4, but only with RTP-compatible
JPEGs (RFC 2435):

1. ffmpeg's mjpeg encoder uses optimized Huffman tables by default. ffmpeg's
   RTP muxer then refuses every frame (`RFC 2435 requires standard Huffman
   tables for jpeg`) and the publisher dies. Fix: `-huffman default`.
2. With standard tables but a single quantisation table (ffmpeg writes one
   when luma and chroma tables are equal), the publish succeeds and the path
   is "ready" with an `M-JPEG` track, but MediaMTX rejects every RTP packet
   (`processing errors, last was: received wrong fragment`), readers see
   `width=0 height=0` and never get a frame. Fix: `-force_duplicated_matrix 1`.

With both options (`-c:v mjpeg -huffman default -force_duplicated_matrix 1`)
frames flow end to end at 360p, 720p and 1080p. RFC 2435 also limits frames
to 2040x2040 and to 4:2:0/4:2:2, so 4K MJPEG cannot be carried over RTSP.
Real cameras already produce compatible JPEGs; the constraint matters only
for simulated sources. `sim-cameras --mjpeg auto` re-encodes any MJPEG file
from outside `testdata/media` for this reason.

## Licensing

Big Buck Bunny (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org.
Sintel (c) copyright Blender Foundation | durian.blender.org. Both CC-BY 3.0.
All generated clips are derived from Big Buck Bunny and carry the same
license. Source URLs, mirrors and checksums are in `testdata/manifest.json`.
