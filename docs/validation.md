# Validation

Beyond unit tests, camtensor is checked against the situations real
deployments run into. Every check below is automated and lives in the
repository.

## Real-camera behaviour (emulated)

`crates/camtensor/tests/camera_quirks.rs` reproduces what IP cameras and
networks do, using MediaMTX, FFmpeg and a hostile TCP proxy written for the test:

| Scenario | What is emulated | Expected and verified |
|---|---|---|
| Digest authentication | Read access requires `viewer:secret` (RTSP digest, as most cameras use) | Credentials work in the URL or as separate fields. A wrong password delivers nothing, keeps retrying and reports the 401 in `last_error`. |
| Audio track | H.264 video plus G.711 A-law audio, the usual camera pairing | The video track is selected and the audio ignored |
| Camera reboot | The camera disappears for 3 s and comes back at a different resolution (640x360 → 1280x720) | Reconnects, reopens the decoder and delivers frames at the new size |
| Hostile network | 1 in 40 RTP packets corrupted in flight, and the connection dropped every 5 s | Keeps delivering (222 images in 18 s at 15 fps, through 55 corrupted packets and 3 drops); reconnects every time |
| Slow camera | 5 fps with a keyframe every 10 s (low-bandwidth setups) | First image arrives at the next keyframe (about 8 s here), then a steady 5 fps |

These are emulations. Camera firmware has its own quirks, so testing against
your own cameras with `camtensor probe rtsp://...` is still recommended.

## PyTorch on the GPU

`crates/camtensor-py/tests/test_torch_gpu.py` (PyTorch 2.14 + CUDA 13, RTX 5070):

- `torch.from_dlpack(batch)` returns a `cuda:0` tensor whose `data_ptr()`
  equals camtensor's buffer: zero-copy, with no synchronisation needed by the
  consumer.
- A torch tensor that outlives its batch stays valid: the buffer pool does
  not reuse memory while any holder remains.
- On the same still image, the CUDA kernel output (via DLPack) and the CPU
  sampler output (independent decoder and implementation) agree within the
  frame-to-frame noise of the encoder.
- Aligned crops go straight into torch operations.

## C ABI

`examples/c/consumer.c` is compiled in CI with gcc (`-Wall -Wextra -Werror`)
and MSVC (`/W4 /WX`), and was run against a live stream on CUDA.

## Long-running stability (soak)

`bench/soak.py` runs many streams for a long time with churn: one camera
reboot and one stream removed and re-added every 5 minutes. It records memory,
threads, handles, GPU memory and throughput.

Run: 16 streams of 1080p H.264, H.265 and MJPEG, decoded by NVDEC into
640x640 f16 batches, for **60 minutes**. Over that time there were **12
churn events and 80 reconnections**, with a sample every 30 s (119 samples).

| Metric | Start (first 20%) | End (last 20%) | Range over the run |
|---|---:|---:|---|
| Process RSS | 582 MB | 603 MB | 549–606 MB; the last sample is 564 MB, below the start |
| Threads | 32 | 31 | constant |
| OS handles | 496 | 395 | do not grow |
| GPU memory (whole device) | 2,442 MiB | 2,371 MiB | does not grow |
| Delivered images/s | 398.7 on average | | real time is 400 |
| Decode errors | 0 | | |
| Images dropped by backpressure | 0 | | |
| Batch buffers allocated | 3–4 | | the pool stays bounded |

Memory moves within a band and does not trend upward, threads and handles
stay flat, and GPU memory does not grow. Throughput stays at real time,
apart from the samples taken while a camera was rebooting.

Reproduce:

```
python bench/soak.py --minutes 60 --streams 16 --churn-min 5
```

## Linux and Docker

Validated with Docker Desktop (WSL2) on the same machine:

- **Image:** `docker build` produces a 378 MB image (Debian trixie + FFmpeg 7). `camtensor caps` reports CUDA available with `--gpus all` and falls back to the CPU without it.
- **NVDEC on Linux:** with `deploy/docker-compose.yml` plus the GPU override, the service discovered the 4 simulated cameras through the MediaMTX API and decoded H.264, H.265 and MJPEG on NVDEC at about 25 fps each, with no errors.
- **Shared-memory data plane on Linux:** `camtensor client detector` inside the service container received multi-camera batches (`[4, 3, 640, 640]` f16) through `/dev/shm`.
- **Scripts on a clean Debian container:** `fetch-test-media.sh --quick` (14/14 files verified) and `sim-cameras.sh start / status / probe / stop` all work.
