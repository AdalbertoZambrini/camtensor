"""Two-stage pipeline skeleton: a first model runs on the full frame, then
the regions it found are cut from the original resolution on the GPU,
aligned and normalised for a second model.

Plug in your models where marked, e.g. with onnxruntime-gpu or TensorRT.

    python examples/python/keypoint_crops.py rtsp://127.0.0.1:8554/cam1
"""

import sys

import numpy as np

import camtensor as ct


def first_stage(batch_tensor):
    """Replace with a real detector. Returns, per image, a list of
    (box[x1, y1, x2, y2], keypoints[5][2]) in *tensor* pixel coordinates."""
    n = batch_tensor.shape[0]
    pts = np.array(ct.KEYPOINT_TEMPLATE_112) * 2 + 200  # a fake detection in the middle
    return [[((180, 180, 420, 420), pts.tolist())] for _ in range(n)]


def second_stage(crops_tensor):
    """Replace with the second model; input is [K, 3, 112, 112] in [-1, 1]."""
    return crops_tensor.shape[0]


url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://127.0.0.1:8554/cam1"
with ct.Engine(device="auto") as eng:
    eng.add_output("stage1", 640, 640, dtype="f32", batch=8, max_wait_ms=15, keep_frame=True)
    eng.add_stream("cam1", url, ["stage1"], gating="auto")
    for k, batch in enumerate(eng.batches("stage1")):
        detections = first_stage(batch.tensor)
        for item, found in zip(batch.items, detections):
            if not item.valid or not found:
                continue
            # Results are in letterboxed tensor pixels: map them back to source
            # pixels, then crop and align at full resolution.
            boxes = [item.box_to_source(list(box)) for box, _ in found]
            aligned = [[list(item.to_source(x, y)) for x, y in pts] for _, pts in found]
            crops = eng.crop(item.frame, aligned, 112, 112, normalize="symmetric")
            boxed = eng.crop(item.frame, [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in boxes], 224, 224)
            print(f"{item.stream_id}: {second_stage(crops)} aligned crop(s) {crops.shape}, "
                  f"box crops {boxed.shape} on {crops.device}")
            # Keep decoding every frame while something of interest is in view.
            eng.force_active(item.stream_id, 3.0)
        if k == 30:
            break
