# Architecture

```
            ┌────────────── per stream ──────────────┐
RTSP ──► ingest (tokio + retina) ──► gate ──► decoder ──► preprocess ──┐
 (MediaMTX or camera)   Annex B AUs   │      NVDEC / CPU   fused kernel  │
                                      │                                 ▼
                         compressed-domain           batcher (one per output, shared by streams)
                         activity score              pooled buffers ─► consumer (Rust / Python / C / service)
```

## Stages

| Stage | Crate | What it does |
|---|---|---|
| Ingest | `camtensor-ingest` | RTSP session (retina), reconnect with back-off, Annex B conversion, parameter sets on keyframes, RTCP-based capture time, drop-until-keyframe when the decoder falls behind. |
| Bitstream & gate | `camtensor-bitstream` | NAL/SPS parsing (H.264, H.265), JPEG SOF, and the motion gate: keyframes only while the scene is static, full decode while packet sizes show activity; the GOP buffer lets the decoder catch up without artefacts. |
| Decode | `camtensor-decode` | NVDEC through a runtime-loaded `nvcuvid` (frames stay in device memory), libavcodec loaded at runtime, or zune-jpeg. Nothing is linked at build time. |
| Preprocess | `camtensor-preprocess`, `camtensor-cuda` | One fused pass per output image: affine sampling (resize, letterbox, crop, rotation), YUV→RGB/BGR/GRAY (BT.601/709, limited/full), normalisation, dtype conversion and layout, written directly into the batch slot. CPU and CUDA implement identical math. |
| Batching | `camtensor` | Multi-camera batches with deadline, padding to static shapes, drop-oldest / latest-only / block backpressure, buffer pool recycled once every holder drops its tensor. |
| Delivery | `camtensor`, `camtensor-py`, `camtensor-capi`, `camtensor-service` | Rust API, Python (DLPack, `__cuda_array_interface__`, NumPy), C ABI, and a standalone service with REST control plane and shared-memory data plane. |

## Why it is cheap

1. **One write per output pixel.** A typical pipeline (decode → BGR frame → resize → normalise → transpose → upload) touches every pixel 4–6 times and moves full-resolution frames across PCIe. camtensor samples the decoded YUV planes once per *output* pixel (640×640 instead of 1920×1080) and writes the final tensor.
2. **Frames stay on the GPU.** With NVDEC, the NV12 surface is copied device-to-device into a pooled buffer and preprocessed in place; nothing crosses PCIe.
3. **Not decoding is cheaper than decoding.** For surveillance cameras, most frames are static. The gate reads only packet sizes and NAL headers, which costs almost nothing, so static streams decode one frame per GOP.
4. **Batching across cameras** gives the inference engine large, regular batches.

## Threads

* 2 tokio I/O threads for every RTSP session.
* 1 decode thread per stream (decode + preprocess + batch write).
* 1 flush thread per output with a batch deadline.
* Consumers call `recv` from their own threads.

## Coordinates

All transforms are *continuous* (`pixel i` covers `[i, i+1)`), the convention under which scaling is exact. `FrameMeta::transform` maps source → tensor; `to_source` / `box_to_source` invert it. ROIs given as OpenCV `warpAffine` matrices are converted internally.
