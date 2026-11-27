# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

### Added
- RTSP ingest (H.264, H.265, MJPEG; TCP/UDP) with reconnects, RTCP capture time and keyframe-aware backpressure.
- MediaMTX auto-discovery through its control API.
- NVDEC decoding through a runtime-loaded `nvcuvid`, a libavcodec decoder loaded at runtime (FFmpeg 5–8) and a pure-Rust MJPEG decoder.
- A fused preprocessing kernel (CUDA PTX) and an equivalent CPU sampler. Both handle colour conversion, resize, letterbox, crop, affine warps, normalisation, dtype and layout.
- Multi-camera batching with deadlines, padding, backpressure policies and pooled buffers.
- Region crops from boxes, affine matrices or five keypoints (similarity alignment), on the GPU in one launch.
- Compressed-domain motion gating: keyframes only while idle, with GOP replay on activity.
- Python bindings with DLPack, `__cuda_array_interface__` and NumPy support, plus a C ABI, a standalone service (REST and shared memory), a client crate and a CLI (`probe`, `run`, `serve`, `bench`).
- Prometheus metrics, test-media tooling, camera simulation scripts and CI for Linux, Windows and ARM64.
