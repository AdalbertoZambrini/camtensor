"""Zero-copy ONNX Runtime (CUDA EP) inference on camtensor batches via IOBinding.

    python examples/python/onnxruntime_iobinding.py rtsp://127.0.0.1:8554/cam1 [--model m.onnx] [--batch 4] [--seconds 10]

camtensor delivers each batch as a contiguous NCHW tensor already in GPU memory.
IOBinding hands that device pointer straight to ONNX Runtime (``bind_input`` with
``device_type="cuda"`` and ``buffer_ptr=batch.tensor.data_ptr``), and the outputs
are bound to preallocated torch CUDA tensors, so no image ever crosses PCIe.

camtensor batches are fully written when they are delivered, so any CUDA stream
(ONNX Runtime's included) may read them without extra synchronisation. Keep the
``batch`` object alive until the run that reads it has finished: its buffer goes
back to camtensor's pool when the object is released.

Without ``--model`` the script exports a tiny CNN classifier (dynamic batch) with
``torch.onnx.export`` and validates one batch against the same model in PyTorch.
With ``--model`` it validates against the ONNX Runtime CPU provider when possible.
The camtensor output dtype (f16/f32) is taken from the model's input type.

Needs: pip install camtensor torch "onnxruntime-gpu[cuda,cudnn]" onnx onnxscript
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time

try:
    import camtensor as ct
except ImportError:
    sys.exit("camtensor is not installed: pip install camtensor")
try:
    import torch
except ImportError:
    sys.exit("PyTorch is required (outputs and validation): pip install torch")
try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    sys.exit('onnxruntime-gpu is not installed: pip install "onnxruntime-gpu[cuda,cudnn]"')

# camtensor dtype name -> (ONNX tensor type string, torch dtype, numpy dtype)
DTYPES = {
    "float16": ("tensor(float16)", torch.float16, np.float16),
    "float32": ("tensor(float)", torch.float32, np.float32),
}
ONNX_TO_CT = {"tensor(float16)": "f16", "tensor(float)": "f32"}


class TinyNet(torch.nn.Module):
    """A small, deterministic CNN classifier: [N, 3, H, W] -> [N, classes] logits."""

    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def make_model() -> TinyNet:
    torch.manual_seed(0)
    return TinyNet().eval()


def export_onnx(model: torch.nn.Module, path: str, size: int, half: bool, max_batch: int) -> None:
    m = model.half() if half else model.float()
    dtype = torch.float16 if half else torch.float32
    x = torch.rand(2, 3, size, size, dtype=dtype)
    batch = torch.export.Dim("batch", min=1, max=max(max_batch, 64))
    torch.onnx.export(
        m, (x,), path, input_names=["images"], output_names=["logits"],
        dynamic_shapes={"x": {0: batch}}, dynamo=True, verbose=False,  # verbose prints emoji (breaks cp1252 consoles)
    )
    model.float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?", default="rtsp://127.0.0.1:8554/cam1")
    ap.add_argument("--model", help="ONNX model with one NCHW float input (default: export a tiny CNN)")
    ap.add_argument("--batch", type=int, default=4, help="max images per batch")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to run")
    ap.add_argument("--size", type=int, default=224, help="input size for the built-in model")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="precision of the built-in model")
    ap.add_argument("--cache-dir", default=os.path.join(tempfile.gettempdir(), "camtensor-examples"),
                    help="where the built-in model is exported")
    ap.add_argument("--device", default="cuda:0", help="camtensor device (must be CUDA for IOBinding)")
    args = ap.parse_args()

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        sys.exit(f"CUDAExecutionProvider unavailable (have {ort.get_available_providers()}); "
                 'install onnxruntime-gpu, not onnxruntime')
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # pick up CUDA/cuDNN from the nvidia-* pip wheels

    torch_model = None
    model_path = args.model
    if model_path is None:
        os.makedirs(args.cache_dir, exist_ok=True)
        model_path = os.path.join(args.cache_dir, f"tinynet_{args.size}_{args.dtype}.onnx")
        torch_model = make_model()
        if not os.path.exists(model_path):
            export_onnx(torch_model, model_path, args.size, args.dtype == "f16", args.batch)
            print(f"exported {model_path}")

    gpu_index = int(args.device.split(":")[1]) if ":" in args.device else 0
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # Fail at load time if any node would be placed on the CPU EP instead of CUDA.
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = ort.InferenceSession(
        model_path, so,
        providers=[("CUDAExecutionProvider", {"device_id": gpu_index})],
    )
    providers = sess.get_providers()
    print("ORT", ort.__version__, "providers:", providers)
    assert providers[0] == "CUDAExecutionProvider", "CUDA EP was not used (silent CPU fallback)"

    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    if len(inp.shape) != 4 or not all(isinstance(d, int) for d in inp.shape[1:]):
        sys.exit(f"input {inp.name} {inp.shape}: expected NCHW with fixed C, H, W")
    _, c, h, w = inp.shape
    if inp.type not in ONNX_TO_CT:
        sys.exit(f"unsupported model input type {inp.type}")
    ct_dtype = ONNX_TO_CT[inp.type]
    print(f"model input {inp.name} {inp.shape} {inp.type} -> camtensor dtype {ct_dtype}")

    # Preallocated GPU output buffers sized for the largest batch; each run binds a
    # leading slice of them, which is still contiguous.
    by_onnx = {v[0]: v for v in DTYPES.values()}
    out_bufs = {}  # name -> (torch buffer, numpy dtype)
    for o in outs:
        if o.type not in by_onnx:
            sys.exit(f"unsupported output type {o.type}")
        if not all(isinstance(d, int) for d in o.shape[1:]):
            sys.exit(f"output {o.name} {o.shape}: only the batch dimension may be dynamic")
        _, tdt, ndt = by_onnx[o.type]
        buf = torch.empty([args.batch, *o.shape[1:]], dtype=tdt, device=f"cuda:{gpu_index}")
        out_bufs[o.name] = (buf, ndt)
    io = sess.io_binding()

    def run(ptr: int, shape, dtype: str):
        """Run on a device buffer (camtensor dtype name, e.g. "float16"); returns output views."""
        onnx_type, _, np_dtype = DTYPES[dtype]
        assert onnx_type == inp.type, f"tensor {dtype} does not match model input {inp.type}"
        n = shape[0]
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        io.bind_input(inp.name, device_type="cuda", device_id=gpu_index, element_type=np_dtype,
                      shape=tuple(shape), buffer_ptr=ptr)
        views = {}
        for name, (buf, ndt) in out_bufs.items():
            v = buf[:n]
            io.bind_output(name, device_type="cuda", device_id=gpu_index, element_type=ndt,
                           shape=tuple(v.shape), buffer_ptr=v.data_ptr())
            views[name] = v
        sess.run_with_iobinding(io)  # synchronises the EP stream before returning
        return views

    # Warm up every batch size before the camera starts: the first run per shape
    # includes kernel loading and cuDNN algorithm search, which would otherwise stall
    # the stream (camtensor drops the oldest batches while the consumer is busy).
    t0 = time.perf_counter()
    ct_name = {"f16": "float16", "f32": "float32"}[ct_dtype]
    for n in range(1, args.batch + 1):
        dummy = torch.zeros((n, int(c), int(h), int(w)), dtype=DTYPES[ct_name][1], device=f"cuda:{gpu_index}")
        run(dummy.data_ptr(), dummy.shape, ct_name)
    print(f"warm-up (batch sizes 1..{args.batch}): {(time.perf_counter() - t0) * 1e3:.0f} ms")

    eng = ct.Engine(device=args.device)
    eng.add_output("model", int(w), int(h), dtype=ct_dtype, normalize="unit",
                   color="gray" if c == 1 else "rgb", batch=args.batch, max_wait_ms=100)
    eng.add_stream("cam1", args.url, ["model"])

    lat, sizes, start = [], [], None
    try:
        for batch in eng.batches("model"):
            t = batch.tensor
            assert t.device.startswith("cuda"), f"batch is on {t.device}; use a CUDA device"
            t0 = time.perf_counter()
            res = run(t.data_ptr, t.shape, t.dtype)  # zero-copy: camtensor's buffer is the input
            dt = time.perf_counter() - t0
            if start is None:  # validate the first batch, then start the clock
                validate(batch, res[outs[0].name], torch_model, model_path, inp.name)
                start = time.perf_counter()
                continue
            lat.append(dt)
            sizes.append(len(batch))
            print(f"batch {batch.seq}: n={len(batch)} infer {dt * 1e3:6.2f} ms "
                  f"({len(batch) / dt:6.0f} img/s), capture->tensor {batch[0].latency_ms:5.1f} ms")
            if time.perf_counter() - start >= args.seconds:
                break
    finally:
        eng.close()

    if lat:
        total = sum(sizes)
        wall = time.perf_counter() - start
        print(f"\n{len(lat)} batches, {total} images in {wall:.1f} s: {total / wall:.1f} img/s end-to-end "
              f"(bounded by the camera frame rate)")
        print(f"inference per batch (bind + run + sync): mean {statistics.mean(lat) * 1e3:.2f} ms, "
              f"median {statistics.median(lat) * 1e3:.2f} ms, max {max(lat) * 1e3:.2f} ms; "
              f"compute throughput {total / sum(lat):.0f} img/s, mean batch {total / len(lat):.1f}")


def validate(batch, gpu_out: torch.Tensor, torch_model, model_path: str, input_name: str) -> bool:
    x = torch.from_dlpack(batch)  # zero-copy view of the same GPU buffer
    got = gpu_out.float()
    if torch_model is not None:
        with torch.no_grad():
            ref = torch_model.to(x.device)(x.float()).float()
        what = "PyTorch fp32"
    else:
        try:
            cpu = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            ref = torch.from_numpy(cpu.run(None, {input_name: x.cpu().numpy()})[0]).float().to(x.device)
        except Exception as e:  # e.g. fp16 kernels missing on the CPU provider
            print(f"validation skipped: ONNX Runtime CPU could not run the model ({e})")
            return True
        what = "ONNX Runtime CPU"
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    tol = 2e-2 * max(scale, 1.0) if gpu_out.dtype == torch.float16 else 1e-3 * max(scale, 1.0)
    print(f"validation vs {what}: max |diff| {err:.3e} (max |ref| {scale:.3e}, tol {tol:.1e}), "
          f"argmax agree {(got.argmax(1) == ref.argmax(1)).float().mean().item() * 100:.0f}%")
    assert err <= tol, "ONNX Runtime CUDA output does not match the reference"
    return True


if __name__ == "__main__":
    main()
