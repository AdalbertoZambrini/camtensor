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
- Video files as sources (`file:` URLs, FFmpeg 6+), with real-time or as-fast-as-possible playback and looping.
- Webcams and capture cards (`device:` URLs via libavdevice; raw YUYV/UYVY/NV12/I420/grey and MJPEG), `camtensor devices`, `ct.list_devices()`.
- Multi-GPU engines: least-loaded (pixel-rate) or pinned placement, per-device output shards, `recv_on` / `subscribe_on` / `device=` in Python, C (`ct_engine_recv_device`), the service data plane and the CLI (`--devices`).
- Free-threaded CPython (3.14t, 3.15t) support: the extension keeps the GIL disabled; `cp314t` / `cp315t` wheels.
- Fix: with `block` backpressure, removing a stream or shutting down could deadlock.
- Python asyncio support: `async for batch in eng.abatches(...)`, `await eng.arecv(...)`.
- `camtensor client` to read batches from the service's shared-memory data plane.
- Performance: asynchronous CUDA fences (one wait per batch instead of one per image) and page-locked host memory for CPU outputs.
- Per-stream regions of interest in normalised coordinates.
- JPEG snapshots of kept frames (Rust, Python, C).
- ONNX Runtime (IOBinding) and TensorRT zero-copy examples.
- Docker image and a compose file with MediaMTX, simulated cameras and the service (GPU override included).
- Prometheus metrics, test-media tooling, camera simulation scripts and CI for Linux, Windows and ARM64.
