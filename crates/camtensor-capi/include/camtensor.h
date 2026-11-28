/*
 * camtensor C API: camera streams to inference-ready tensors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Link against the `camtensor_c` shared library (camtensor_c.dll / libcamtensor_c.so)
 * built from crates/camtensor-capi.
 *
 *   ct_engine *e = ct_engine_from_file("camtensor.yaml");
 *   ct_batch *b;
 *   while (ct_engine_recv(e, "detector", -1, &b) == CT_OK) {
 *       ct_tensor t; ct_batch_tensor(b, &t);   // t.data = CUDA or host pointer
 *       ... run inference on t ...
 *       ct_batch_free(b);                       // returns the buffer to the pool
 *   }
 *   ct_engine_free(e);
 *
 * All functions are thread-safe. Error details: ct_last_error() (thread-local).
 */
#ifndef CAMTENSOR_H
#define CAMTENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CT_OK 0
#define CT_TIMEOUT 1
#define CT_CLOSED 2
#define CT_ERROR (-1)
#define CT_INVALID (-2)
#define CT_NOT_FOUND (-3)

/* DLPack-compatible device types. */
#define CT_DEVICE_CPU 1
#define CT_DEVICE_CUDA 2

/* Element types. */
#define CT_DTYPE_U8 0
#define CT_DTYPE_F16 1
#define CT_DTYPE_F32 2

typedef struct CtEngine ct_engine;
typedef struct CtBatch ct_batch;

typedef struct ct_tensor {
    int32_t device_type;   /* CT_DEVICE_* */
    int32_t device_id;     /* CUDA ordinal */
    int32_t dtype;         /* CT_DTYPE_* */
    int32_t ndim;          /* always 4 */
    int64_t shape[4];      /* NCHW or NHWC */
    int64_t strides[4];    /* in elements */
    uint64_t data;         /* host pointer or CUdeviceptr; fully written when delivered */
    uint64_t nbytes;
} ct_tensor;

typedef struct ct_item {
    const char *stream_id;      /* valid until ct_batch_free */
    uint64_t seq;
    int64_t pts;                /* 90 kHz */
    int32_t valid;              /* 0 for padding slots */
    int32_t keyframe;
    uint32_t src_width;
    uint32_t src_height;
    float transform[6];         /* continuous dst_from_src affine: x' = a x + b y + c, y' = d x + e y + f */
    uint64_t capture_time_ns;   /* 0 if unknown (RTCP) */
    uint64_t recv_time_ns;
    uint64_t ready_time_ns;
    uint64_t dropped_since_last;
} ct_item;

const char *ct_version(void);
const char *ct_last_error(void);

/* Create from a YAML/JSON string (NULL = empty config) or a file. NULL on error. */
ct_engine *ct_engine_new(const char *config_yaml);
ct_engine *ct_engine_from_file(const char *path);
void ct_engine_free(ct_engine *engine);

/* Configure at runtime with JSON (same schema as the YAML config). */
int ct_engine_add_output_json(const ct_engine *engine, const char *name, const char *output_json);
int ct_engine_add_stream_json(const ct_engine *engine, const char *stream_json);
int ct_engine_remove_stream(const ct_engine *engine, const char *stream_id);
int ct_engine_force_active(const ct_engine *engine, const char *stream_id, double seconds);

/* Wait for a batch; timeout_ms < 0 waits forever. Returns CT_OK, CT_TIMEOUT, CT_CLOSED or an error. */
int ct_engine_recv(const ct_engine *engine, const char *output, int timeout_ms, ct_batch **out);
/* Multi-GPU: only batches produced on `device` ("cuda:1", "cpu", ...). */
int ct_engine_recv_device(const ct_engine *engine, const char *output, const char *device,
                          int timeout_ms, ct_batch **out);
size_t ct_batch_len(const ct_batch *batch);
int ct_batch_tensor(const ct_batch *batch, ct_tensor *out);
int ct_batch_item(const ct_batch *batch, size_t index, ct_item *out);
void ct_batch_free(ct_batch *batch);

/* JPEG of the full-resolution frame kept with item `index` (output needs keep_frame).
 * max_width 0 keeps the original size. Free *out with ct_bytes_free(*out, *out_len). */
int ct_batch_snapshot(const ct_engine *engine, const ct_batch *batch, size_t index, int quality,
                      uint32_t max_width, uint8_t **out, size_t *out_len);
void ct_bytes_free(uint8_t *p, size_t len);

/* Statistics as JSON; free with ct_string_free. */
char *ct_engine_stats_json(const ct_engine *engine);
void ct_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* CAMTENSOR_H */
