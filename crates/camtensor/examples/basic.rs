//! Minimal consumer: `cargo run -p camtensor --example basic -- rtsp://127.0.0.1:8554/cam1`

use std::time::Duration;

use camtensor::{BatchConfig, DType, Engine, EngineConfig, OutputConfig, StreamConfig, TensorSpec};

fn main() -> camtensor::Result<()> {
    let url = std::env::args().nth(1).unwrap_or_else(|| "rtsp://127.0.0.1:8554/cam1".into());
    let mut cfg = EngineConfig::default();
    cfg.outputs.insert(
        "detector".into(),
        OutputConfig {
            tensor: TensorSpec { width: 640, height: 640, dtype: DType::F16, ..Default::default() },
            batch: BatchConfig { max_size: 4, max_wait_ms: 20, pad_to_max: false },
            ..Default::default()
        },
    );
    cfg.streams.push(StreamConfig::new("cam1", url, &["detector"]));
    let engine = Engine::new(cfg)?;
    println!("device: {}", engine.device());
    for _ in 0..50 {
        let batch = engine.recv("detector", Some(Duration::from_secs(10)))?;
        let t = batch.tensor();
        let m = &batch.items()[0].meta;
        println!(
            "batch #{:<4} {:?} {:?} on {}  src {}x{}  latency {:.2} ms",
            batch.seq(),
            t.shape(),
            t.dtype(),
            t.device(),
            m.src_width,
            m.src_height,
            m.ready_time_ns.saturating_sub(m.recv_time_ns) as f64 / 1e6
        );
    }
    engine.shutdown();
    Ok(())
}
