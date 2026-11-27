//! `camtensor` command line tool.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use camtensor::{
    BatchConfig, DType, DecoderPref, Device, Engine, EngineConfig, GatingConfig, GatingMode,
    OutputConfig, StreamConfig, TensorSpec,
};

#[derive(Parser)]
#[command(name = "camtensor", version, about = "Camera streams to inference-ready tensors")]
struct Cli {
    /// Log filter (e.g. `info`, `camtensor=debug`). Also read from RUST_LOG.
    #[arg(long, global = true, default_value = "info")]
    log: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show available backends (CUDA/NVDEC, libavcodec, JPEG).
    Caps,
    /// Connect to a camera and report codec, resolution, decoder and rates.
    Probe {
        /// rtsp:// URL.
        url: String,
        /// Seconds to observe.
        #[arg(long, default_value_t = 5)]
        seconds: u64,
        /// Device (auto, cpu, cuda:N).
        #[arg(long, default_value = "auto")]
        device: String,
    },
    /// Run a pipeline from a config file and print statistics.
    Run {
        /// YAML/JSON engine configuration.
        #[arg(short, long)]
        config: String,
        /// Seconds between statistics lines.
        #[arg(long, default_value_t = 2)]
        interval: u64,
        /// Stop after this many seconds (0 = until Ctrl+C).
        #[arg(long, default_value_t = 0)]
        duration: u64,
    },
    /// Run the standalone service (REST control plane + shared-memory data plane).
    Serve {
        /// YAML/JSON engine configuration (optional: streams can be added over REST).
        #[arg(short, long)]
        config: Option<String>,
        /// Control plane address.
        #[arg(long, default_value = "127.0.0.1:7070")]
        http: String,
        /// Data plane address.
        #[arg(long, default_value = "127.0.0.1:7071")]
        data: String,
        /// Device override (auto, cpu, cuda:N).
        #[arg(long)]
        device: Option<String>,
    },
    /// Benchmark: N streams into one batched output, report throughput,
    /// latency and resource usage.
    Bench(BenchArgs),
    /// Receive batches from a running service through its shared-memory
    /// data plane (must run on the same machine / IPC namespace).
    Client {
        /// Output name.
        output: String,
        /// Data plane address.
        #[arg(long, default_value = "127.0.0.1:7071")]
        data: String,
        /// Number of batches to receive.
        #[arg(long, default_value_t = 10)]
        count: usize,
    },
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Camera URLs (repeated round-robin to reach --streams).
    #[arg(required = true)]
    urls: Vec<String>,
    /// Number of streams to open.
    #[arg(long, default_value_t = 1)]
    streams: usize,
    /// Device (auto, cpu, cuda:N).
    #[arg(long, default_value = "auto")]
    device: String,
    /// Decoder (auto, nvdec, cpu).
    #[arg(long, default_value = "auto")]
    decoder: String,
    /// Output width and height.
    #[arg(long, default_value_t = 640)]
    size: u32,
    /// Output dtype (u8, f16, f32).
    #[arg(long, default_value = "f16")]
    dtype: String,
    /// Batch size.
    #[arg(long, default_value_t = 8)]
    batch: usize,
    /// Batch deadline in milliseconds.
    #[arg(long, default_value_t = 20)]
    max_wait_ms: u64,
    /// Gating mode (always, keyframes, auto).
    #[arg(long, default_value = "always")]
    gating: String,
    /// Warm-up seconds (not measured).
    #[arg(long, default_value_t = 3)]
    warmup: u64,
    /// Measured seconds.
    #[arg(long, default_value_t = 15)]
    seconds: u64,
    /// Print a Markdown table row.
    #[arg(long)]
    markdown: bool,
    /// Write the result as JSON to this file.
    #[arg(long)]
    json: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let filter = std::env::var("RUST_LOG").unwrap_or(cli.log.clone());
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .with_target(false)
        .init();
    match cli.cmd {
        Cmd::Caps => {
            println!("camtensor {}", camtensor::VERSION);
            println!("{}", serde_json::to_string_pretty(&camtensor::capabilities())?);
            Ok(())
        }
        Cmd::Probe { url, seconds, device } => probe(&url, seconds, &device),
        Cmd::Run { config, interval, duration } => run(&config, interval, duration),
        Cmd::Serve { config, http, data, device } => serve(config, &http, &data, device),
        Cmd::Bench(args) => bench(args),
        Cmd::Client { output, data, count } => client(&output, &data, count),
    }
}

fn ctrl_c_flag() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let f = flag.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
        if let Ok(rt) = rt {
            rt.block_on(async {
                let _ = tokio::signal::ctrl_c().await;
            });
            f.store(true, Ordering::Relaxed);
        }
    });
    flag
}

fn probe(url: &str, seconds: u64, device: &str) -> Result<()> {
    let mut cfg = EngineConfig { device: device.parse()?, ..Default::default() };
    cfg.outputs.insert(
        "probe".into(),
        OutputConfig {
            tensor: TensorSpec { width: 64, height: 64, dtype: DType::U8, ..Default::default() },
            ..Default::default()
        },
    );
    cfg.streams.push(StreamConfig::new("probe", url, &["probe"]));
    let engine = Engine::new(cfg)?;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut images = 0u64;
    let mut first: Option<Duration> = None;
    let t0 = Instant::now();
    while Instant::now() < deadline {
        if let Ok(b) = engine.recv("probe", Some(Duration::from_millis(200))) {
            images += b.valid_count() as u64;
            first.get_or_insert(t0.elapsed());
        }
    }
    let st = engine.stats();
    let s = st.streams.first().context("no stream")?;
    println!("url          {}", s.url);
    println!("connected    {}", s.connected);
    println!("codec        {}", s.codec.as_deref().unwrap_or("?"));
    println!("resolution   {}x{}", s.width, s.height);
    println!("decoder      {}", s.decoder.as_deref().unwrap_or("?"));
    println!("device       {}", st.device);
    println!("packets      {} ({:.1}/s)", s.packets, s.packets as f64 / seconds as f64);
    println!("bitrate      {:.2} Mbit/s", s.bytes as f64 * 8.0 / seconds as f64 / 1e6);
    println!("frames       {} ({:.1}/s)", s.frames, s.frames as f64 / seconds as f64);
    println!("delivered    {images}");
    if let Some(f) = first {
        println!("first image  {:.0} ms after start", f.as_secs_f64() * 1e3);
    }
    println!("latency      {:.2} ms (receive -> tensor ready)", s.avg_latency_ms);
    if let Some(e) = &s.last_error {
        println!("last error   {e}");
    }
    engine.shutdown();
    if s.frames == 0 {
        bail!("no frame decoded");
    }
    Ok(())
}

fn drain_all(engine: &Engine, stop: Arc<AtomicBool>) -> Vec<std::thread::JoinHandle<()>> {
    engine
        .outputs()
        .into_iter()
        .map(|o| {
            let e = engine.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _ = e.recv(&o, Some(Duration::from_millis(200)));
                }
            })
        })
        .collect()
}

fn run(config: &str, interval: u64, duration: u64) -> Result<()> {
    let engine = Engine::from_config_file(config)?;
    let stop = ctrl_c_flag();
    let drains = drain_all(&engine, stop.clone());
    let t0 = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_secs(interval.max(1)));
        let st = engine.stats();
        for s in &st.streams {
            println!(
                "{:<16} {:<5} {:>5}x{:<5} dec={:<6} gate={:<7} in={:>5.1}fps dec={:>5.1}fps lat={:>6.2}ms err={}",
                s.id,
                s.codec.as_deref().unwrap_or("-"),
                s.width,
                s.height,
                s.decoder.as_deref().unwrap_or("-"),
                s.gate,
                s.input_fps,
                s.decode_fps,
                s.avg_latency_ms,
                s.decode_errors
            );
        }
        for (name, o) in &st.outputs {
            println!(
                "  output {name}: batches={} images={} dropped={} queued={}",
                o.batches, o.images, o.dropped_images, o.queued
            );
        }
        if duration > 0 && t0.elapsed() >= Duration::from_secs(duration) {
            stop.store(true, Ordering::Relaxed);
        }
    }
    engine.shutdown();
    for d in drains {
        let _ = d.join();
    }
    Ok(())
}

fn serve(config: Option<String>, http: &str, data: &str, device: Option<String>) -> Result<()> {
    let mut cfg = match config {
        Some(p) => EngineConfig::from_path(p)?,
        None => EngineConfig::default(),
    };
    if let Some(d) = device {
        cfg.device = d.parse()?;
    }
    let engine = Engine::new(cfg)?;
    let opts = camtensor_service::ServeOptions {
        http: http.parse().context("--http")?,
        data: data.parse().context("--data")?,
        ..Default::default()
    };
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(camtensor_service::serve(engine.clone(), opts, async {
        let _ = tokio::signal::ctrl_c().await;
    }))?;
    engine.shutdown();
    Ok(())
}

#[derive(serde::Serialize)]
struct BenchResult {
    streams: usize,
    device: String,
    decoders: Vec<String>,
    codec: String,
    resolution: String,
    dtype: String,
    size: u32,
    batch: usize,
    seconds: f64,
    images_per_sec: f64,
    per_stream_fps: f64,
    latency_avg_ms: f64,
    latency_p50_ms: f64,
    latency_p99_ms: f64,
    dropped: u64,
    cpu_percent: f64,
    rss_mb: f64,
    gpu: Option<String>,
}

fn bench(a: BenchArgs) -> Result<()> {
    let dtype = match a.dtype.as_str() {
        "u8" => DType::U8,
        "f16" => DType::F16,
        "f32" => DType::F32,
        o => bail!("invalid dtype {o}"),
    };
    let gating = match a.gating.as_str() {
        "always" => GatingMode::Always,
        "keyframes" => GatingMode::Keyframes,
        "auto" => GatingMode::Auto,
        o => bail!("invalid gating {o}"),
    };
    let decoder = match a.decoder.as_str() {
        "auto" => DecoderPref::Auto,
        "nvdec" => DecoderPref::Nvdec,
        "cpu" => DecoderPref::Cpu,
        o => bail!("invalid decoder {o}"),
    };
    let device: Device = a.device.parse()?;
    let mut cfg = EngineConfig { device, decoder, ..Default::default() };
    cfg.outputs.insert(
        "bench".into(),
        OutputConfig {
            tensor: TensorSpec { width: a.size, height: a.size, dtype, ..Default::default() },
            batch: BatchConfig { max_size: a.batch, max_wait_ms: a.max_wait_ms, pad_to_max: false },
            queue_depth: 8,
            ..Default::default()
        },
    );
    for i in 0..a.streams {
        let url = &a.urls[i % a.urls.len()];
        let mut s = StreamConfig::new(format!("s{i:03}"), url.clone(), &["bench"]);
        s.gating = GatingConfig { mode: gating, ..Default::default() };
        cfg.streams.push(s);
    }
    let engine = Engine::new(cfg)?;
    eprintln!("warming up {} s on {} ...", a.warmup, engine.device());
    let warm_end = Instant::now() + Duration::from_secs(a.warmup);
    while Instant::now() < warm_end {
        let _ = engine.recv("bench", Some(Duration::from_millis(100)));
    }

    let pid = sysinfo::get_current_pid().ok();
    let mut sys = sysinfo::System::new().ok();
    let cpu_snapshot = |sys: &mut Option<sysinfo::System>| -> (f64, f64) {
        match (sys.as_mut(), pid) {
            (Some(s), Some(p)) => {
                s.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[p]), true);
                s.process(p)
                    .map(|pr| (pr.cpu_usage() as f64, pr.memory() as f64 / 1048576.0))
                    .unwrap_or((0.0, 0.0))
            }
            _ => (0.0, 0.0),
        }
    };
    let _ = cpu_snapshot(&mut sys);
    let before = engine.stats();
    let mut lat: Vec<f64> = Vec::with_capacity(100_000);
    let mut images = 0u64;
    let mut cpu_samples = Vec::new();
    let mut rss = 0.0f64;
    let t0 = Instant::now();
    let mut next_sample = Instant::now() + Duration::from_secs(1);
    let mut gpu: Option<String> = None;
    while t0.elapsed() < Duration::from_secs(a.seconds) {
        if let Ok(b) = engine.recv("bench", Some(Duration::from_millis(100))) {
            for it in b.items().iter().filter(|i| i.valid) {
                images += 1;
                // Receive -> tensor ready (the camera clock from RTCP is not reliable
                // for file-based publishers).
                lat.push(it.meta.ready_time_ns.saturating_sub(it.meta.recv_time_ns) as f64 / 1e6);
            }
        }
        if Instant::now() >= next_sample {
            let (c, m) = cpu_snapshot(&mut sys);
            cpu_samples.push(c);
            rss = rss.max(m);
            if gpu.is_none() && t0.elapsed() >= Duration::from_secs(a.seconds / 2) {
                gpu = nvidia_smi();
            }
            next_sample += Duration::from_secs(1);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let after = engine.stats();
    engine.shutdown();

    lat.sort_by(|x, y| x.total_cmp(y));
    let pct =
        |p: f64| if lat.is_empty() { 0.0 } else { lat[((lat.len() - 1) as f64 * p) as usize] };
    let first = after.streams.first();
    let mut decoders: Vec<String> =
        after.streams.iter().filter_map(|s| s.decoder.clone()).collect();
    decoders.sort();
    decoders.dedup();
    let dropped = after.outputs.get("bench").map_or(0, |o| o.dropped_images)
        - before.outputs.get("bench").map_or(0, |o| o.dropped_images);
    let r = BenchResult {
        streams: a.streams,
        device: after.device.clone(),
        decoders,
        codec: first.and_then(|s| s.codec.clone()).unwrap_or_default(),
        resolution: first.map(|s| format!("{}x{}", s.width, s.height)).unwrap_or_default(),
        dtype: a.dtype.clone(),
        size: a.size,
        batch: a.batch,
        seconds: secs,
        images_per_sec: images as f64 / secs,
        per_stream_fps: images as f64 / secs / a.streams as f64,
        latency_avg_ms: if lat.is_empty() {
            0.0
        } else {
            lat.iter().sum::<f64>() / lat.len() as f64
        },
        latency_p50_ms: pct(0.5),
        latency_p99_ms: pct(0.99),
        dropped,
        cpu_percent: if cpu_samples.is_empty() {
            0.0
        } else {
            cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64
        },
        rss_mb: rss,
        gpu,
    };
    if a.markdown {
        println!(
            "| {} | {} {} | {} | {} | {:.0} | {:.1} | {:.1} / {:.1} | {:.0}% | {:.0} MB | {} |",
            r.streams,
            r.codec,
            r.resolution,
            r.decoders.join("+"),
            r.device,
            r.images_per_sec,
            r.per_stream_fps,
            r.latency_p50_ms,
            r.latency_p99_ms,
            r.cpu_percent,
            r.rss_mb,
            r.gpu.as_deref().unwrap_or("-")
        );
    } else {
        println!("{}", serde_json::to_string_pretty(&r)?);
    }
    if let Some(p) = a.json {
        std::fs::write(p, serde_json::to_string_pretty(&r)?)?;
    }
    Ok(())
}

/// GPU utilisation snapshot (`gpu%, dec%, MiB`) if nvidia-smi is available.
fn nvidia_smi() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.decoder,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let v: Vec<&str> = s.lines().next()?.split(',').map(str::trim).collect();
    (v.len() == 3).then(|| format!("gpu {}% / nvdec {}% / {} MiB", v[0], v[1], v[2]))
}

fn client(output: &str, data: &str, count: usize) -> Result<()> {
    let mut c = camtensor_client::Client::connect(data, output)
        .with_context(|| format!("connect to data plane {data}"))?;
    println!(
        "subscribed to '{}': {}x{} {:?}, up to {} per batch",
        c.output, c.tensor.width, c.tensor.height, c.tensor.dtype, c.max_batch
    );
    for _ in 0..count {
        let b = c.recv()?;
        let now = camtensor::core::time::now_ns();
        let lat: Vec<String> = b
            .items
            .iter()
            .filter(|i| i.valid)
            .map(|i| {
                format!(
                    "{}:{:.1}ms",
                    i.meta.stream_id,
                    now.saturating_sub(i.meta.recv_time_ns) as f64 / 1e6
                )
            })
            .collect();
        println!(
            "batch #{:<5} {:?} {} {} bytes  [{}]",
            b.seq,
            b.shape,
            b.dtype,
            b.data().len(),
            lat.join(" ")
        );
    }
    Ok(())
}
