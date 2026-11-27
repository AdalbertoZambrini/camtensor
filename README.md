# camtensor

**Camera streams to inference-ready tensors.** RTSP in; batched, normalised, letterboxed tensors out, already on the GPU.

camtensor replaces the glue every video-AI project rewrites (`VideoCapture` → decode → BGR → resize → normalise → transpose → upload). It is a single lightweight Rust engine with Python, C and REST/shared-memory interfaces.

[![CI](https://github.com/AdalbertoZambrini/camtensor/actions/workflows/ci.yml/badge.svg)](https://github.com/AdalbertoZambrini/camtensor/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

```python
import camtensor as ct, torch

eng = ct.Engine(device="auto")                       # NVDEC + CUDA when available, CPU otherwise
eng.add_output("detector", 640, 640, dtype="f16", batch=8, max_wait_ms=15, keep_frame=True)
eng.add_stream("cam1", "rtsp://127.0.0.1:8554/cam1", ["detector"], gating="auto")

for batch in eng.batches("detector"):
    x = torch.from_dlpack(batch)                     # [N, 3, 640, 640] float16 on cuda:0, zero-copy
    for item in batch.items:
        box = item.box_to_source([10, 20, 110, 220]) # map detections back to camera pixels
```

## Why

Video inference is often limited by everything that happens *before* the model runs. A typical pipeline decodes on the CPU and touches every full-resolution pixel 4–6 times. It then copies frames across PCIe, and often decodes every frame of cameras that show an empty corridor.

| 16 × 1080p H.264 cameras | Images/s | CPU | RAM |
|---|---:|---:|---:|
| OpenCV `VideoCapture` + NumPy | 156 (cannot keep up) | 762% | 2.4 GB |
| **camtensor (NVDEC)** | **400 (real time)** | **25%** | **0.5 GB** |

camtensor on NVDEC keeps **48 full-HD cameras in real time on less than one CPU core**. Methodology and more numbers are in [docs/benchmarks.md](docs/benchmarks.md).

## Features

- **RTSP ingest:** H.264, H.265 and MJPEG over TCP or UDP. Reconnects with back-off, takes the capture time from RTCP, and drops frames until the next keyframe when the consumer falls behind.
- **MediaMTX integration:** discovers paths through its API (glob filters) and reacts when publishers appear or disappear. It also works with any RTSP camera directly.
- **Hardware decode:** NVDEC through a runtime-loaded `nvcuvid`, so frames never leave the GPU. CPU fallback uses a runtime-loaded libavcodec (FFmpeg 5–8) or a pure-Rust MJPEG decoder. No video library is linked at build time.
- **Fused preprocessing:** one pass per output pixel covers colour conversion (BT.601/709, limited/full range), resize, letterbox, crop and arbitrary affine warps. It also handles normalisation presets (`unit`, `imagenet`, `symmetric`, custom), `u8`/`f16`/`f32` output, NCHW/NHWC layout and RGB/BGR/GRAY channel order. CUDA kernel and SIMD-friendly CPU code implement the same math.
- **Multi-camera batching:** batches close when full or at a deadline. A static batch size can be padded, backpressure is configurable (`drop_oldest`, `latest_only`, `block`), and buffer pools are recycled.
- **Region crops for multi-stage pipelines:** keep the full-resolution frame, then cut **boxes, affine warps or crops aligned from 5 keypoints** for a second model, on the GPU, in one launch, with no frame leaving the device.
- **Compressed-domain motion gating:** packet sizes are read without decoding. Idle cameras decode keyframes only, and the buffered GOP is replayed the moment activity starts. This cut CPU by 54% on a mostly static scene.
- **Everywhere:** Rust crate, Python wheel (`abi3`, 1.6 MB, DLPack / `__cuda_array_interface__` / NumPy), C ABI, a standalone service (REST control plane and shared-memory data plane), and a CLI (`probe`, `run`, `serve`, `bench`). Prometheus metrics are built in.
- **Linux and Windows,** on x86_64 and ARM64. The GPU path needs only the NVIDIA driver at runtime; the CPU path runs on any processor.

## Install

```bash
pip install camtensor          # Python (wheels for Linux and Windows)
cargo add camtensor            # Rust
cargo install camtensor-cli    # CLI: camtensor probe / run / serve / bench
```

Runtime requirements:

| Path | Needs |
|---|---|
| GPU (NVDEC + CUDA) | NVIDIA driver ≥ 525 (Maxwell or newer). No CUDA toolkit. |
| CPU H.264/H.265 | FFmpeg shared libraries: `apt install libavcodec-dev` (Linux), or an LGPL "shared" build with `CAMTENSOR_FFMPEG_DIR` pointing at its `bin` folder (Windows). |
| CPU MJPEG | Nothing. A pure-Rust decoder is built in. |

## Configuration

Everything can be declared in one YAML file, which is shared by the CLI, the service, Python (`ct.Engine("camtensor.yaml")`), Rust and C.

```yaml
device: auto                    # auto | cpu | cuda:N
outputs:
  detector:
    tensor: { width: 640, height: 640, dtype: f16, layout: nchw, color: rgb,
              resize: letterbox, normalize: unit, pad: [114, 114, 114] }
    batch: { max_size: 8, max_wait_ms: 15, pad_to_max: false }
    keep_frame: true            # enables Engine.crop on delivered items
    max_fps: 10                 # per-stream cap for this output
    backpressure: drop_oldest
streams:
  - id: entrance
    url: rtsp://user:pass@10.0.0.20:554/stream1
    outputs: [detector]
    gating: { mode: auto, threshold: 1.6, hold_ms: 2000 }
mediamtx:                       # optional auto-discovery
  api_url: http://127.0.0.1:9997
  rtsp_url: rtsp://127.0.0.1:8554
  paths: ["cam*"]
  outputs: [detector]
```

## Two-stage pipelines

Your first model returns coordinates. camtensor cuts, aligns and normalises those regions from the full-resolution frame, which never leaves the GPU; only the coordinates travel back.

```python
eng.add_output("stage1", 640, 640, dtype="f16", keep_frame=True)
for batch in eng.batches("stage1"):
    results = first_model(torch.from_dlpack(batch))             # boxes and/or 5 keypoints per object
    for i, item in enumerate(batch.items):
        pts = [[item.to_source(x, y) for x, y in kp] for kp in results[i].keypoints]
        if pts:
            crops = eng.crop(item.frame, pts, 112, 112, normalize="symmetric")  # aligned, [K, 3, 112, 112]
            out = second_model(torch.from_dlpack(crops))
```

See [examples/python/keypoint_crops.py](examples/python/keypoint_crops.py).

## Rust

```rust
use camtensor::{Engine, EngineConfig};

let engine = Engine::new(EngineConfig::from_path("camtensor.yaml")?)?;
for batch in engine.subscribe("detector")? {
    let t = batch.tensor();            // t.data_ptr() is a CUdeviceptr on cuda:N
    println!("{:?} {:?} from {} cameras", t.shape(), t.device(), batch.len());
}
```

## Service

```bash
camtensor serve -c camtensor.yaml --http 127.0.0.1:7070 --data 127.0.0.1:7071
curl -X POST localhost:7070/v1/streams -d '{"id":"cam9","url":"rtsp://...","outputs":["detector"]}'
curl localhost:7070/metrics
```

Consumers in other processes receive batches through shared memory (`camtensor-client`).

## Repository layout

| Crate | Role |
|---|---|
| `camtensor-core` | Output contract, configuration, geometry (letterbox, affine, keypoint alignment) |
| `camtensor-ingest` | RTSP client (retina), reconnects, Annex B conversion |
| `camtensor-bitstream` | H.264/H.265/JPEG parsing, motion gate |
| `camtensor-decode` | NVDEC, libavcodec and zune-jpeg decoders |
| `camtensor-cuda` | Runtime-loaded CUDA driver and nvcuvid bindings, fused PTX kernel |
| `camtensor-preprocess` | CPU fused sampler |
| `camtensor` | Engine: streams, batching, crops, stats |
| `camtensor-py`, `camtensor-capi`, `camtensor-service`, `camtensor-client`, `camtensor-cli` | Interfaces |

See [docs/architecture.md](docs/architecture.md) for the design and [docs/testing.md](docs/testing.md) for test media and camera simulation.

## Development

```bash
cargo test --workspace            # unit + integration (MediaMTX/FFmpeg tests auto-skip when missing)
scripts/fetch-test-media.sh       # real and synthetic clips (H.264/H.265/MJPEG, 360p–4K, gating clips)
scripts/sim-cameras.sh start --preset mixed --count 8
cargo run -p xtask -- ptx         # regenerate the CUDA PTX (needs NVRTC; the PTX is committed)
```

## Roadmap

- Intel/AMD GPU decode (VAAPI, D3D11VA, oneVPL) and Jetson
- Zero-copy GPU IPC for the service (CUDA VMM handles)
- 10-bit HEVC and NV16 output for MJPEG 4:2:2
- AVX2/NEON CPU sampler paths

## License

Apache-2.0
