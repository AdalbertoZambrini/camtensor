"""Two-stage face recognition skeleton: detector on the full frame, then
ArcFace-aligned 112x112 crops cut on the GPU from the original resolution.

Plug in your models where marked (SCRFD/RetinaFace/YOLO-face for detection,
ArcFace/AdaFace for embeddings), e.g. with onnxruntime-gpu or TensorRT.

    python examples/python/face_pipeline.py rtsp://127.0.0.1:8554/cam1
"""

import sys

import numpy as np

import camtensor as ct


def detect(batch_tensor):
    """Replace with a real detector. Returns, per image, a list of
    (box[x1, y1, x2, y2], landmarks[5][2]) in *tensor* pixel coordinates."""
    n = batch_tensor.shape[0]
    tpl = np.array(ct.ARCFACE_112_TEMPLATE) * 2 + 200  # a fake face in the middle
    return [[((180, 180, 420, 420), tpl.tolist())] for _ in range(n)]


def embed(faces_tensor):
    """Replace with an embedding model; input is [F, 3, 112, 112] in [-1, 1]."""
    return faces_tensor.shape[0]


url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://127.0.0.1:8554/cam1"
with ct.Engine(device="auto") as eng:
    eng.add_output("detector", 640, 640, dtype="f32", batch=8, max_wait_ms=15, keep_frame=True)
    eng.add_stream("cam1", url, ["detector"], gating="auto")
    for k, batch in enumerate(eng.batches("detector")):
        dets = detect(batch.tensor)
        for item, faces in zip(batch.items, dets):
            if not item.valid or not faces:
                continue
            # Landmarks were found on the letterboxed tensor: map them back to
            # source pixels, then align at full resolution.
            rois = [[list(item.to_source(x, y)) for x, y in lm] for _, lm in faces]
            crops = eng.crop(item.frame, rois, 112, 112, normalize="arcface")
            print(f"{item.stream_id}: {embed(crops)} face(s), crops {crops.shape} on {crops.device}")
            # Keep decoding every frame while someone is in view.
            eng.force_active(item.stream_id, 3.0)
        if k == 30:
            break
