# Python API

```bash
pip install camtensor
```

```python
import camtensor as ct
```

Wheels are `abi3` (Python ≥ 3.9) for Linux and Windows. The GPU path needs
only the NVIDIA driver. CPU decoding of H.264/H.265 and file or device
sources need FFmpeg shared libraries; see the README.

## Module

| Name | Description |
|---|---|
| `ct.Engine(config=None, device=None, devices=None)` | Creates an engine. `config` is `None`, a dict, a YAML/JSON string or a path to a YAML/JSON file. `device` is `"auto"`, `"cpu"` or `"cuda:N"`. `devices` (e.g. `["auto"]` for every GPU, or `["cuda:0", "cuda:1"]`) enables multi-GPU. |
| `ct.capabilities()` | `{"cuda": bool, "ffmpeg": bool, "jpeg": bool}` |
| `ct.list_devices(backend=None)` | Capture devices as `[(name, description)]`. On Windows, use `"device:" + description`. |
| `ct.KEYPOINT_TEMPLATE_112` | Default 5-keypoint layout for aligned 112×112 crops |
| `ct.__version__` | Version string |

## Engine

### Outputs

```python
eng.add_output(
    name, width=640, height=640, *,
    dtype="f32",            # "u8" | "f16" | "f32"
    layout="nchw",          # "nchw" | "nhwc"
    color="rgb",            # "rgb" | "bgr" | "gray"
    resize="letterbox",     # "letterbox" | "letterbox_top_left" | "stretch" | "crop_center" | "none"
    interp="bilinear",      # "bilinear" | "nearest"
    normalize=None,         # "unit" (default) | "imagenet" | "symmetric" | "none" | {"scale":..,"mean":[..],"std":[..]}
    pad=None,               # padding colour, a number or [r, g, b] (default 114)
    batch=1, max_wait_ms=0, pad_to_max=False,
    keep_frame=False,       # keep the full-resolution frame for crop()/snapshot()
    max_fps=None,           # per-stream rate cap for this output
    backpressure="drop_oldest",  # "drop_oldest" | "latest_only" | "block"
    queue_depth=4,
    device=None,            # "cpu" to receive host tensors from a CUDA engine
)
```

Every stream that lists an output feeds the same batches.

### Streams

```python
eng.add_stream(
    id, url, outputs, *,
    gating=None,            # "always" | "keyframes" | "auto" | dict
    transport="tcp",        # RTSP: "tcp" | "udp"
    decoder=None,           # "auto" | "nvdec" | "cpu"
    username=None, password=None,
    region=None,            # (x, y, w, h) normalised 0..1
    realtime=True, loop=True,   # file sources
    device=None,            # pin to one engine device (multi-GPU); default: least loaded
)
eng.remove_stream(id)
eng.force_active(id, seconds)   # bypass gating, e.g. while something is tracked
```

| URL | Source |
|---|---|
| `rtsp://user:pass@host:554/path` | Camera or RTSP server |
| `file:videos/a.mp4` | Video file (MP4, MKV, MOV, raw H.264/H.265, MJPEG) |
| `device:USB Camera` / `device:/dev/video0` | Webcam or capture card |

Capture options (`width`, `height`, `fps`, `format`, `backend`) can be set
through the config dict/YAML (`capture:` on the stream).

### Receiving

```python
batch = eng.recv("detector", timeout=None)      # blocking, GIL released
batch = eng.recv("detector", device="cuda:1")   # multi-GPU: only that GPU's batches
for batch in eng.batches("detector", device="cuda:1"): ...
batch = eng.try_recv("detector")                # None if nothing is ready
for batch in eng.batches("detector"): ...       # blocking iterator
async for batch in eng.abatches("detector"): ...   # asyncio
batch = await eng.arecv("detector", timeout=1.0)
```

The engine has fully written a batch by the time it is delivered: any CUDA
stream may read it.

### Crops and snapshots

```python
t = eng.crop(item.frame, rois, 112, 112, normalize="symmetric")  # Tensor [K, 3, 112, 112]
jpeg = eng.snapshot(item.frame, region=(x, y, w, h), width=640, quality=90)  # bytes
```

`rois` takes any mix of:
- `(x, y, w, h)` boxes in source pixels;
- 5×2 keypoint lists (similarity alignment onto `KEYPOINT_TEMPLATE_112`);
- 6-value affine matrices in OpenCV `warpAffine` convention;
- dicts `{"kind": "box" | "affine" | "keypoints", ...}`.

### Introspection

`eng.device`, `eng.devices`, `eng.outputs`, `eng.streams`, `eng.stats()` (dict),
`eng.prometheus()` (text), `eng.config()` (dict), `eng.close()`. The engine
is also a context manager.

## Batch, Item, Tensor, Frame

| Object | Members |
|---|---|
| `Batch` | `tensor`, `items`, `output`, `seq`, `valid_count`, `len()`, `[i]`, `numpy()`, `__dlpack__` (so `torch.from_dlpack(batch)` works) |
| `Item` | `stream_id`, `valid`, `seq`, `pts`, `keyframe`, `src_width`, `src_height`, `transform`, `capture_time_ns`, `recv_time_ns`, `ready_time_ns`, `latency_ms`, `dropped_since_last`, `frame`, `to_source(x, y)`, `box_to_source(box)`, `boxes_to_source(boxes)` |
| `Tensor` | `shape`, `dtype`, `device`, `data_ptr`, `nbytes`, `__dlpack__`, `__dlpack_device__`, `__cuda_array_interface__` (CUDA), `__array_interface__` (CPU), `numpy()` |
| `Frame` | `width`, `height`, `format`, `device`, `pts` (pass to `crop` / `snapshot`) |

## Free-threaded Python

On free-threaded CPython (3.14t, 3.15t) camtensor keeps the GIL disabled
(`sys._is_gil_enabled()` stays `False` after import). Batches, items,
frames and tensors are immutable and safe to share between threads, and the
engine can be used from any number of threads at once. These builds need the
version-specific wheels (`cp314t`, `cp315t`); the `abi3` wheel covers
regular builds from 3.9 on.

## Errors

| Exception | When |
|---|---|
| `ValueError` | Invalid configuration or argument |
| `KeyError` | Unknown output or stream |
| `TimeoutError` | `recv` timed out |
| `RuntimeError` | Runtime failures (device, decoder, closed engine) |
