/* Minimal C consumer.
 * Build (Linux):  cc consumer.c -I../../crates/camtensor-capi/include -L../../target/release -lcamtensor_c
 * Build (MSVC):   cl consumer.c /I..\..\crates\camtensor-capi\include ..\..\target\release\camtensor_c.dll.lib
 */
#include <stdio.h>
#include "camtensor.h"

int main(int argc, char **argv) {
    const char *url = argc > 1 ? argv[1] : "rtsp://127.0.0.1:8554/cam1";
    char yaml[1024];
    snprintf(yaml, sizeof yaml,
             "outputs:\n  det:\n    tensor: {width: 640, height: 640, dtype: f16}\n"
             "    batch: {max_size: 4, max_wait_ms: 20}\n"
             "streams:\n  - {id: cam1, url: \"%s\", outputs: [det]}\n", url);
    ct_engine *e = ct_engine_new(yaml);
    if (!e) { fprintf(stderr, "error: %s\n", ct_last_error()); return 1; }
    for (int i = 0; i < 20; i++) {
        ct_batch *b;
        int rc = ct_engine_recv(e, "det", 10000, &b);
        if (rc != CT_OK) { fprintf(stderr, "recv: %d %s\n", rc, ct_last_error()); break; }
        ct_tensor t; ct_item it;
        ct_batch_tensor(b, &t);
        ct_batch_item(b, 0, &it);
        printf("batch of %zu [%lld,%lld,%lld,%lld] on %s:%d from %s (%ux%u)\n", ct_batch_len(b),
               (long long)t.shape[0], (long long)t.shape[1], (long long)t.shape[2], (long long)t.shape[3],
               t.device_type == CT_DEVICE_CUDA ? "cuda" : "cpu", t.device_id, it.stream_id,
               it.src_width, it.src_height);
        /* t.data is a CUdeviceptr (CUDA) or host pointer: feed it to TensorRT / ONNX Runtime here. */
        ct_batch_free(b);
    }
    ct_engine_free(e);
    return 0;
}
