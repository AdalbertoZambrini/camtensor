//! Service test: REST control plane + shared-memory data plane.

#[path = "../../camtensor-ingest/tests/common/mod.rs"]
#[allow(dead_code)]
mod common;

use std::io::{Read, Write};
use std::time::Duration;

use camtensor::{Device, Engine, EngineConfig, OutputConfig, TensorSpec};

fn http(port: u16, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let code = r.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, r.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

#[test]
fn rest_and_shared_memory() {
    let Some((mtx_exe, ffmpeg)) = common::tools("service") else { return };
    let mtx = common::MediaMtx::start(&mtx_exe, 18660);
    let _p = common::publish(&ffmpeg, &mtx.url("cam"), "640x360", common::H264);
    mtx.wait_ready("cam", Duration::from_secs(15));

    let mut cfg = EngineConfig { device: Device::Auto, ..Default::default() };
    cfg.outputs.insert(
        "det".into(),
        OutputConfig {
            tensor: TensorSpec { width: 128, height: 128, ..Default::default() },
            ..Default::default()
        },
    );
    let engine = Engine::new(cfg).unwrap();
    let opts = camtensor_service::ServeOptions {
        http: "127.0.0.1:18670".parse().unwrap(),
        data: "127.0.0.1:18671".parse().unwrap(),
        ..Default::default()
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = rt.spawn(camtensor_service::serve(engine.clone(), opts, async {
        let _ = rx.await;
    }));
    std::thread::sleep(Duration::from_millis(300));

    assert_eq!(http(18670, "GET", "/health", "").0, 200);
    let body = format!(r#"{{"id":"cam","url":"{}","outputs":["det"]}}"#, mtx.url("cam"));
    assert_eq!(http(18670, "POST", "/v1/streams", &body).0, 201);
    assert_eq!(http(18670, "POST", "/v1/streams", &body).0, 400, "duplicate id");
    assert_eq!(http(18670, "GET", "/v1/streams/nope", "").0, 404);

    let mut client = camtensor_client::Client::connect("127.0.0.1:18671", "det").unwrap();
    assert_eq!(client.tensor.width, 128);
    for _ in 0..5 {
        let b = client.recv().unwrap();
        assert_eq!(b.shape, [1, 3, 128, 128]);
        assert_eq!(b.items[0].meta.stream_id.as_ref(), "cam");
        let v = b.as_f32().unwrap();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
    }

    let (code, stats) = http(18670, "GET", "/v1/streams/cam", "");
    assert_eq!(code, 200, "{stats}");
    assert!(stats.contains("\"connected\":true"));
    let (_, metrics) = http(18670, "GET", "/metrics", "");
    assert!(metrics.contains("camtensor_output_images_total{output=\"det\"}"));
    assert_eq!(http(18670, "DELETE", "/v1/streams/cam", "").0, 204);

    drop(client);
    let _ = tx.send(());
    rt.block_on(server).unwrap().unwrap();
    engine.shutdown();
}
