"""Zero-copy TensorRT inference on camtensor batches (execute_async_v3).

    python examples/python/tensorrt_infer.py rtsp://127.0.0.1:8554/cam1 [--model m.onnx] [--batch 4] [--seconds 10]

camtensor delivers each batch as a contiguous NCHW tensor already in GPU memory.
Its device pointer is given to TensorRT as the input address
(``context.set_tensor_address(input_name, batch.tensor.data_ptr)``); outputs are
written into preallocated torch CUDA tensors. No image is copied.

CUDA streams: camtensor batches are fully written when they are delivered, so any
stream may read them; here TensorRT runs on a torch-created stream and the script
synchronises it before reading the outputs. Keep the ``batch`` object alive until
that synchronisation: its buffer returns to camtensor's pool when it is released.

The engine is built from ONNX with a dynamic-batch optimisation profile
(min 1, opt/max ``--batch``) and cached on disk (``--cache-dir``), keyed by the
ONNX contents, the TensorRT version, the GPU and the batch size.

FP16: without ``--model`` the script exports a tiny CNN with FP16 weights and an
FP16 input, and builds a strongly typed network, so TensorRT runs it in FP16.
TensorRT 11 only builds strongly typed networks (precision comes from the ONNX
types; ``BuilderFlag.FP16`` no longer exists). On TensorRT 10 an FP32 ONNX is
built with ``BuilderFlag.FP16`` instead. The camtensor output dtype (f16/f32)
follows the engine's input type.

Validation compares one batch with the same model in PyTorch (built-in model) or
with ONNX Runtime on the CPU (``--model``, when onnxruntime is installed).

Needs: pip install camtensor torch tensorrt-cu13 onnx onnxscript
"""

from __future__ import annotations

import argparse
import hashlib
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
    sys.exit("PyTorch is required (CUDA stream, outputs, validation): pip install torch")
try:
    import tensorrt as trt
except ImportError:
    sys.exit("TensorRT is not installed: pip install tensorrt-cu13 (CUDA 13) or tensorrt-cu12 (CUDA 12)")

TRT_TO_TORCH = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
TRT_TO_CT = {trt.DataType.HALF: "f16", trt.DataType.FLOAT: "f32"}
CT_TO_TRT = {"float16": trt.DataType.HALF, "float32": trt.DataType.FLOAT}


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
    x = torch.rand(2, 3, size, size, dtype=torch.float16 if half else torch.float32)
    batch = torch.export.Dim("batch", min=1, max=max(max_batch, 64))
    torch.onnx.export(
        m, (x,), path, input_names=["images"], output_names=["logits"],
        dynamic_shapes={"x": {0: batch}}, dynamo=True,
        verbose=False,  # the verbose exporter prints emoji, which breaks cp1252 consoles
    )
    model.float()


def build_engine(onnx_path: str, cache_dir: str, max_batch: int, logger: trt.Logger) -> bytes:
    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()
    key = hashlib.sha1(onnx_bytes + trt.__version__.encode()
                       + torch.cuda.get_device_name().encode() + str(max_batch).encode()).hexdigest()[:16]
    plan_path = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(onnx_path))[0]}.b{max_batch}.{key}.engine")
    if os.path.exists(plan_path):
        print(f"loading cached engine {plan_path}")
        with open(plan_path, "rb") as f:
            return f.read()

    t0 = time.perf_counter()
    builder = trt.Builder(logger)
    flags = 0
    use_fp16_flag = hasattr(trt.BuilderFlag, "FP16")  # TensorRT <= 10
    if not use_fp16_flag:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes, onnx_path):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        sys.exit(f"ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    if use_fp16_flag:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        c, h, w = t.shape[1:]
        if min(c, h, w) < 1:
            sys.exit(f"input {t.name} {t.shape}: only the batch dimension may be dynamic")
        profile.set_shape(t.name, (1, c, h, w), (max_batch, c, h, w), (max_batch, c, h, w))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        sys.exit("TensorRT engine build failed (see the log above)")
    os.makedirs(cache_dir, exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(plan)
    print(f"built engine in {time.perf_counter() - t0:.1f} s -> {plan_path}")
    return bytes(plan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?", default="rtsp://127.0.0.1:8554/cam1")
    ap.add_argument("--model", help="ONNX model with one NCHW float input (default: export a tiny CNN)")
    ap.add_argument("--batch", type=int, default=4, help="max images per batch (profile max)")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to run")
    ap.add_argument("--size", type=int, default=224, help="input size for the built-in model")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="precision of the built-in model")
    ap.add_argument("--cache-dir", default=os.path.join(tempfile.gettempdir(), "camtensor-examples"),
                    help="where the built-in ONNX model and the engine are cached")
    ap.add_argument("--device", default="cuda:0", help="camtensor/TensorRT CUDA device")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("PyTorch has no CUDA device; TensorRT needs an NVIDIA GPU")
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    logger = trt.Logger(trt.Logger.WARNING)

    torch_model = None
    model_path = args.model
    if model_path is None:
        os.makedirs(args.cache_dir, exist_ok=True)
        model_path = os.path.join(args.cache_dir, f"tinynet_{args.size}_{args.dtype}.onnx")
        torch_model = make_model()
        if not os.path.exists(model_path):
            export_onnx(torch_model, model_path, args.size, args.dtype == "f16", args.batch)
            print(f"exported {model_path}")

    print("TensorRT", trt.__version__, "on", torch.cuda.get_device_name(dev))
    plan = build_engine(model_path, args.cache_dir, args.batch, logger)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1:
        sys.exit(f"expected one input, got {inputs}")
    inp = inputs[0]
    in_dtype = engine.get_tensor_dtype(inp)
    if in_dtype not in TRT_TO_CT:
        sys.exit(f"unsupported engine input type {in_dtype}")
    _, c, h, w = engine.get_tensor_shape(inp)
    ct_dtype = TRT_TO_CT[in_dtype]
    print(f"engine input {inp} {tuple(engine.get_tensor_shape(inp))} {in_dtype} -> camtensor dtype {ct_dtype}")

    # Output buffers for the largest batch; each run uses a leading (contiguous) slice.
    context.set_input_shape(inp, (args.batch, c, h, w))
    out_bufs = {}
    for name in outputs:
        dt = engine.get_tensor_dtype(name)
        if dt not in TRT_TO_TORCH:
            sys.exit(f"unsupported output type {name}: {dt}")
        out_bufs[name] = torch.empty(tuple(context.get_tensor_shape(name)), dtype=TRT_TO_TORCH[dt], device=dev)
    stream = torch.cuda.Stream(device=dev)

    def run(ptr: int, shape, dtype: str):
        """Run on a device buffer (camtensor dtype name, e.g. "float16"); returns output views."""
        assert CT_TO_TRT[dtype] == in_dtype, f"tensor {dtype} does not match engine input {in_dtype}"
        n = shape[0]
        context.set_input_shape(inp, tuple(shape))
        context.set_tensor_address(inp, ptr)  # zero-copy: TensorRT reads camtensor's buffer
        views = {}
        for name, buf in out_bufs.items():
            v = buf[:n]
            assert tuple(v.shape) == tuple(context.get_tensor_shape(name))
            context.set_tensor_address(name, v.data_ptr())
            views[name] = v
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        stream.synchronize()
        return views

    t0 = time.perf_counter()
    for n in range(1, args.batch + 1):
        dummy = torch.zeros((n, c, h, w), dtype=TRT_TO_TORCH[in_dtype], device=dev)
        run(dummy.data_ptr(), dummy.shape, {"f16": "float16", "f32": "float32"}[ct_dtype])
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
            res = run(t.data_ptr, t.shape, t.dtype)
            dt = time.perf_counter() - t0
            if start is None:  # validate the first batch, then start the clock
                validate(batch, res[outputs[0]], torch_model, model_path)
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
        print(f"inference per batch (enqueue + sync): mean {statistics.mean(lat) * 1e3:.2f} ms, "
              f"median {statistics.median(lat) * 1e3:.2f} ms, max {max(lat) * 1e3:.2f} ms; "
              f"compute throughput {total / sum(lat):.0f} img/s, mean batch {total / len(lat):.1f}")


def validate(batch, gpu_out: torch.Tensor, torch_model, model_path: str) -> None:
    x = torch.from_dlpack(batch)  # zero-copy view of the same GPU buffer
    got = gpu_out.float()
    if torch_model is not None:
        with torch.no_grad():
            ref = torch_model.to(x.device)(x.float()).float()
        what = "PyTorch fp32"
    else:
        try:
            import onnxruntime as ort

            cpu = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            feed = {cpu.get_inputs()[0].name: x.cpu().numpy()}
            ref = torch.from_numpy(cpu.run(None, feed)[0]).float().to(x.device)
        except Exception as e:  # onnxruntime missing, or no fp16 CPU kernels
            print(f"validation skipped: no CPU reference ({e})")
            return
        what = "ONNX Runtime CPU"
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    # FP16 engines (TensorRT 10 FP16 flag, or an FP16 graph) get a looser bound.
    half = gpu_out.dtype == torch.float16 or x.dtype == torch.float16 or hasattr(trt.BuilderFlag, "FP16")
    tol = (2e-2 if half else 1e-3) * max(scale, 1.0)
    print(f"validation vs {what}: max |diff| {err:.3e} (max |ref| {scale:.3e}, tol {tol:.1e}), "
          f"argmax agree {(got.argmax(1) == ref.argmax(1)).float().mean().item() * 100:.0f}%")
    assert err <= tol, "TensorRT output does not match the reference"


if __name__ == "__main__":
    main()
