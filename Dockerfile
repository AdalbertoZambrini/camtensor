# camtensor service image.
#
#   docker build -t camtensor .
#   docker run --rm --gpus all -p 7070:7070 -p 7071:7071 -v ./camtensor.yaml:/config/camtensor.yaml camtensor
#
# The GPU path needs only the NVIDIA driver on the host (NVIDIA Container
# Toolkit / Docker Desktop with WSL2); CUDA and NVDEC are loaded at runtime.
# Without a GPU the same image decodes on the CPU with libavcodec.

FROM rust:1-slim-trixie AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p camtensor-cli \
    && cp target/release/camtensor /usr/local/bin/camtensor

FROM debian:trixie-slim
LABEL org.opencontainers.image.title="camtensor" \
      org.opencontainers.image.description="Camera streams to inference-ready tensors" \
      org.opencontainers.image.source="https://github.com/AdalbertoZambrini/camtensor" \
      org.opencontainers.image.licenses="Apache-2.0"
# libavcodec/libavformat/libavdevice (FFmpeg 7): CPU decoding, files, capture devices.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libavcodec61 libavformat61 libavdevice61 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 camtensor
COPY --from=build /usr/local/bin/camtensor /usr/local/bin/camtensor
COPY deploy/camtensor.yaml /config/camtensor.yaml
# Expose the driver's CUDA and video (NVDEC) libraries to the container.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    RUST_LOG=info
USER camtensor
EXPOSE 7070 7071
HEALTHCHECK --interval=30s --timeout=3s CMD ["camtensor", "--log", "error", "caps"]
ENTRYPOINT ["camtensor"]
CMD ["serve", "-c", "/config/camtensor.yaml", "--http", "0.0.0.0:7070", "--data", "0.0.0.0:7071"]
