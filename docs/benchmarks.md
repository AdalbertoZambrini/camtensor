# Benchmarks

All numbers were measured on one laptop. They show orders of magnitude, not guarantees.

| | |
|---|---|
| CPU | Intel Core i9-14900HX (24 cores / 32 threads), Windows 11, balanced power plan |
| GPU | NVIDIA GeForce RTX 5070 Laptop, 8 GB, driver 621.24 |
| Source | MediaMTX v1.21.4 on localhost, FFmpeg publishing in real time (`-re`) |
| Clip | 1920x1080 at 25 fps, H.264 High, VBR 6 Mbit/s, GOP 2 s, **with B-frames** |
| Output | 640x640 RGB, f16, NCHW, letterbox, batches of 8 (20 ms deadline) |

Each stream is a separate RTSP session, so every camera is decoded independently. CPU is the process total as a share of one core: 100% means one core fully busy.

## Many cameras, one output

| Pipeline | Streams | Images/s | Per stream | CPU | RAM | GPU |
|---|---:|---:|---:|---:|---:|---|
| OpenCV `VideoCapture` + NumPy (typical Python) | 4 | 93 | 23.3 fps | 303% | 638 MB | none |
| OpenCV `VideoCapture` + NumPy (typical Python) | 16 | 156 | **9.8 fps** (falls behind) | 762% | 2,426 MB | none |
| camtensor, CPU decode (libavcodec) | 16 | 389 | 24.3 fps | 454% | 519 MB | none |
| **camtensor, NVDEC** | 16 | 400 | 25.0 fps | **25%** | 488 MB | NVDEC 73%*, SM 10% |
| **camtensor, NVDEC** | 32 | 801 | 25.0 fps | 51% | 841 MB | NVDEC 48%, SM 6% |
| **camtensor, NVDEC** | 48 | 1,182 | 24.6 fps | 89% | 1,206 MB | NVDEC 50%, SM 7% |

\* NVDEC utilisation is a single `nvidia-smi` sample taken halfway through the run, so it can vary.

* With 16 cameras, the OpenCV pipeline can no longer keep up in real time, and uses 30x the CPU and 5x the memory of camtensor on NVDEC.
* camtensor on NVDEC keeps 48 full-HD cameras in real time on less than one CPU core.

Reproduce with:

```
camtensor bench rtsp://127.0.0.1:8554/cam1 --streams 16 --markdown
camtensor bench rtsp://127.0.0.1:8554/cam1 --streams 16 --device cpu --markdown
python bench/baseline_opencv.py rtsp://127.0.0.1:8554/cam1 --streams 16
```

## Latency

Latency is measured from the moment the RTSP packet is received to the moment its image is ready in the batch tensor.

| Stream | Decoder | p50 |
|---|---|---:|
| H.264 1080p, **no B-frames** | NVDEC | **1.4 ms** |
| MJPEG 720p | NVDEC | 1.8 ms |
| H.264 1080p with B-frames | NVDEC / CPU | ~80 ms |

With B-frames, about 80 ms are spent in frame reordering, which any decoder has to do: a picture cannot be shown before the later frame it references has arrived. Most IP cameras do not use B-frames.

## Compressed-domain gating

The test camera shows a static scene, with movement at 10–16 s, 25–28 s and 40–50 s of every minute. The run uses 16 streams on the CPU decoder for 60 s.

| Gating | Images decoded and delivered | CPU |
|---|---:|---:|
| `always` | 400/s | 361% |
| `auto` | 160/s (−60%) | **166% (−54%)** |

While the scene is idle, only keyframes are decoded. When packet sizes show activity, the buffered GOP is replayed through the decoder and full-rate decoding resumes until `hold_ms` after the movement stops.

Gating relies on packet sizes, so it does not work on cameras configured for constant bitrate (CBR); use VBR.

## Kernels and decoders (micro-benchmarks)

| Operation | Time |
|---|---:|
| CUDA fused kernel, 1080p NV12 → 640x640 f16 letterbox | 12.6 µs / frame (at launch-overhead floor) |
| CUDA, 32 keypoint-aligned 112x112 crops in one launch | ~21 µs |
| CPU fused sampler, 1080p NV12 → 640x640 f16, one core | 1.22 ms |
| CPU fused sampler, same, Rayon | ~0.19 ms |
| NVDEC decode + copy, 1080p H.264, one stream | ~1,685 fps |
| libavcodec, 1080p H.264, 1 thread | ~212 fps |
