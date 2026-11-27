# Contributing

Issues and pull requests are welcome.

1. Fork the repository and create a feature branch (`feat/...`, `fix/...`).
2. Keep `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` clean.
3. Add tests:
   - Unit tests next to the code.
   - Integration tests in `crates/*/tests`. Tests that need MediaMTX, FFmpeg or a GPU must skip when those are missing.
4. Run `cargo test --workspace`. To run the end-to-end tests locally, fetch test media with `scripts/fetch-test-media.*` (see [docs/testing.md](docs/testing.md)).
5. If you change `crates/camtensor-cuda/kernels/*.cu`, regenerate the committed PTX with `cargo run -p xtask -- ptx` and keep the CPU sampler (`camtensor-preprocess`) bit-compatible.
6. Use [Conventional Commits](https://www.conventionalcommits.org/) for messages.

By contributing you agree that your contributions are licensed under Apache-2.0.
