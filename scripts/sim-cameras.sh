#!/usr/bin/env bash
# Simulates RTSP cameras: starts MediaMTX with deploy/mediamtx.yml and publishes
# looping media files as rtsp://127.0.0.1:8554/cam1..camN.
#
# Usage:
#   scripts/sim-cameras.sh start [-n COUNT] [-p PRESET | FILE...] [options]
#   scripts/sim-cameras.sh probe     # ffprobe every published camera (decodes ~2 s)
#   scripts/sim-cameras.sh status    # processes + MediaMTX path list
#   scripts/sim-cameras.sh stop      # stop publishers and MediaMTX
#
# start options:
#   -n, --count N        number of cameras (default 4; files are assigned round-robin)
#   -p, --preset NAME    h264-1080p | h265-1080p | mjpeg-720p | mixed (default) | gating
#   --prefix NAME        path prefix (default: cam -> cam1, cam2, ...)
#   --transport T        RTSP transport used for publishing: tcp (default) | udp
#   --mjpeg MODE         auto (default: copy files from testdata/media, which are RFC 2435
#                        compatible, re-encode others) | copy | reencode
#   --external           do not start MediaMTX (publish to an already running server)
#   FILE...              media files (absolute, or relative to testdata/media) instead of a preset
#
# Environment:
#   CAMTENSOR_FFMPEG, CAMTENSOR_FFPROBE, CAMTENSOR_MEDIAMTX   tool paths (default: on PATH)
#   CAMTENSOR_MEDIAMTX_CONFIG   MediaMTX config (default deploy/mediamtx.yml)
#   CAMTENSOR_MEDIA_DIR         media directory (default testdata/media)
#   CAMTENSOR_RTSP_URL          RTSP base URL (default rtsp://127.0.0.1:8554)
#   CAMTENSOR_API_URL           MediaMTX API (default http://127.0.0.1:9997)
#   PROBE_SECONDS               seconds decoded per camera by "probe" (default 10)
#
# Logs and PIDs: target/sim-logs/ (mediamtx.log, camN.log, pids).
# Keep this script behaviourally identical to sim-cameras.ps1.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFMPEG="${CAMTENSOR_FFMPEG:-ffmpeg}"
FFPROBE="${CAMTENSOR_FFPROBE:-ffprobe}"
MEDIAMTX="${CAMTENSOR_MEDIAMTX:-mediamtx}"
MTX_CONFIG="${CAMTENSOR_MEDIAMTX_CONFIG:-$ROOT/deploy/mediamtx.yml}"
MEDIA_DIR="${CAMTENSOR_MEDIA_DIR:-$ROOT/testdata/media}"
RTSP_URL="${CAMTENSOR_RTSP_URL:-rtsp://127.0.0.1:8554}"
API_URL="${CAMTENSOR_API_URL:-http://127.0.0.1:9997}"
LOG_DIR="$ROOT/target/sim-logs"
PID_FILE="$LOG_DIR/pids"

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found (see the CAMTENSOR_* variables in --help)"; }

# Git Bash / MSYS (Windows smoke tests only): native tools need Windows paths.
NATIVE_PATH() { printf '%s' "$1"; }
if command -v cygpath >/dev/null 2>&1; then
  export MSYS2_ARG_CONV_EXCL='*'
  NATIVE_PATH() { cygpath -m "$1"; }
fi

preset_files() {
  case "$1" in
    h264-1080p) echo "h264/h264-high-1920x1080-25fps-gop2s-vbr-6m.mp4" ;;
    h265-1080p) echo "h265/h265-main-1920x1080-25fps-gop2s-vbr-4m.mp4" ;;
    mjpeg-720p) echo "mjpeg/mjpeg-1280x720-25fps-q5.mkv" ;;
    mixed)
      echo "h264/h264-high-1920x1080-25fps-gop2s-vbr-6m.mp4"
      echo "h265/h265-main-1920x1080-25fps-gop2s-vbr-4m.mp4"
      echo "mjpeg/mjpeg-1280x720-25fps-q5.mkv"
      echo "h264/h264-baseline-640x360-25fps-gop1s-vbr-500k.mp4" ;;
    gating)
      echo "gating/gating-h264-1080p25-gop2s-vbr.mp4"
      echo "gating/gating-h265-1080p25-gop2s-vbr.mp4"
      echo "gating/gating-h264-1080p25-gop2s-cbr-4m.mp4"
      echo "gating/gating-h265-1080p25-gop4s-cbr-3m.mp4"
      echo "gating/gating-mjpeg-640x360-15fps-q10.mkv" ;;
    *) die "unknown preset '$1' (h264-1080p, h265-1080p, mjpeg-720p, mixed, gating)" ;;
  esac
}

alive() { kill -0 "$1" 2>/dev/null; }

# Accepts only PIDs that still belong to our tools (guards against PID reuse).
ours() {
  local pid="$1" comm
  alive "$pid" || return 1
  comm="$(ps -p "$pid" -o comm= 2>/dev/null || true)"
  [[ -z "$comm" || "$comm" == *ffmpeg* || "$comm" == *mediamtx* ]]
}

api_get() { curl -fsS --max-time 2 "$API_URL$1" 2>/dev/null; }

# Prints "name ready tracks readers" for every MediaMTX path.
api_paths() {
  local json
  json="$(api_get /v3/paths/list)" || return 1
  if command -v jq >/dev/null 2>&1; then
    jq -r '.items[] | "\(.name) ready=\(.ready) tracks=\(.tracks | join(",")) readers=\(.readers | length)"' <<<"$json"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -I -c 'import json,sys
for p in json.load(sys.stdin)["items"]:
    print("%s ready=%s tracks=%s readers=%d" % (p["name"], str(p["ready"]).lower(), ",".join(p["tracks"]), len(p["readers"])))' <<<"$json" | tr -d '\r'
  else
    echo "$json"
  fi
}

cmd_stop() {
  [[ -f "$PID_FILE" ]] || { echo "not running (no $PID_FILE)"; return 0; }
  local name pid
  # publishers first, MediaMTX last
  while read -r name pid; do
    [[ "$name" == mediamtx ]] && continue
    if ours "$pid"; then kill "$pid" 2>/dev/null && echo "stopped $name (pid $pid)"; fi
  done <"$PID_FILE"
  while read -r name pid; do
    [[ "$name" == mediamtx ]] || continue
    if ours "$pid"; then kill "$pid" 2>/dev/null && echo "stopped mediamtx (pid $pid)"; fi
  done <"$PID_FILE"
  local i
  for i in $(seq 1 20); do
    local any=0
    while read -r name pid; do alive "$pid" && any=1; done <"$PID_FILE"
    [[ $any == 0 ]] && break
    sleep 0.25
  done
  while read -r name pid; do
    if ours "$pid"; then kill -9 "$pid" 2>/dev/null && echo "killed $name (pid $pid)"; fi
  done <"$PID_FILE"
  rm -f "$PID_FILE"
}

cmd_status() {
  if [[ -f "$PID_FILE" ]]; then
    local name pid
    while read -r name pid; do
      printf '%-10s pid %-7s %s\n' "$name" "$pid" "$(alive "$pid" && echo running || echo DEAD)"
    done <"$PID_FILE"
  else
    echo "no processes tracked ($PID_FILE missing)"
  fi
  echo "MediaMTX paths ($API_URL):"
  api_paths | sed 's/^/  /' || echo "  API not reachable"
}

# probe [names...]: ffprobe every camera over RTSP/TCP in parallel, decoding
# PROBE_SECONDS (default 10) of video, enough to reach a keyframe for GOPs up
# to ~8 s. Decoder messages (expected when joining mid-GOP) go to
# target/sim-logs/probe-NAME.log.
cmd_probe() {
  local names=("$@") fails=0 n out frames nerr plog secs="${PROBE_SECONDS:-10}"
  if [[ ${#names[@]} -eq 0 ]]; then
    if [[ -f "$PID_FILE" ]]; then
      mapfile -t names < <(awk '$1 != "mediamtx" { print $1 }' "$PID_FILE")
    else
      mapfile -t names < <(api_paths | awk '{ print $1 }')
    fi
  fi
  [[ ${#names[@]} -gt 0 ]] || die "nothing to probe"
  mkdir -p "$LOG_DIR"
  local to=(); command -v timeout >/dev/null 2>&1 && to=(timeout $((secs + 20)))
  local -a jobs=()
  for n in "${names[@]}"; do
    ${to[@]+"${to[@]}"} "$FFPROBE" -v error -rtsp_transport tcp -timeout 5000000 -read_intervals "%+$secs" \
      -select_streams v:0 -count_frames \
      -show_entries stream=codec_name,profile,width,height,pix_fmt,nb_read_frames \
      -of default=noprint_wrappers=1 "$RTSP_URL/$n" </dev/null >"$LOG_DIR/probe-$n.out" 2>"$LOG_DIR/probe-$n.log" &
    jobs+=("$!")
  done
  wait "${jobs[@]}" 2>/dev/null || true
  for n in "${names[@]}"; do
    plog="$LOG_DIR/probe-$n.log"
    out="$(tr -d '\r' <"$LOG_DIR/probe-$n.out" | tr '\n' ' ')"
    rm -f "$LOG_DIR/probe-$n.out"
    nerr="$(wc -l <"$plog" | tr -d ' ')"
    frames="$(sed -n 's/.*nb_read_frames=\([0-9]*\).*/\1/p' <<<"$out")"
    if [[ -n "$frames" && "$frames" -gt 0 ]] && ! grep -q 'width=0 ' <<<"$out"; then
      echo "OK   $n: $out(decoder messages: $nerr)"
    else
      echo "FAIL $n: ${out:-no output} $(head -n 3 "$plog" | tr -d '\r' | tr '\n' ' ')"; fails=$((fails + 1))
    fi
  done
  [[ $fails == 0 ]]
}

cmd_start() {
  local count=4 preset="" prefix="cam" transport="tcp" mjpeg="auto" external=0
  local -a files=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--count) count="${2:?}"; shift ;;
      -p|--preset) preset="${2:?}"; shift ;;
      --prefix) prefix="${2:?}"; shift ;;
      --transport) transport="${2:?}"; shift ;;
      --mjpeg) mjpeg="${2:?}"; shift ;;
      --external) external=1 ;;
      -h|--help) usage; exit 0 ;;
      -*) die "unknown option $1" ;;
      *) files+=("$1") ;;
    esac
    shift
  done
  [[ "$count" =~ ^[0-9]+$ && "$count" -ge 1 ]] || die "--count must be >= 1"
  [[ "$transport" == tcp || "$transport" == udp ]] || die "--transport must be tcp or udp"
  [[ "$mjpeg" =~ ^(auto|copy|reencode)$ ]] || die "--mjpeg must be auto, copy or reencode"
  need "$FFMPEG"; need "$FFPROBE"; need curl
  if [[ ${#files[@]} -eq 0 ]]; then
    mapfile -t files < <(preset_files "${preset:-mixed}")
  elif [[ -n "$preset" ]]; then
    die "give either --preset or files, not both"
  fi
  # resolve paths, drop missing preset files
  local -a avail=() f p
  for f in "${files[@]}"; do
    p="$f"; [[ "$p" == /* || "$p" =~ ^[A-Za-z]:[/\\] ]] || p="$MEDIA_DIR/$f"
    if [[ -f "$p" ]]; then avail+=("$p"); else echo "warning: missing $p (run scripts/fetch-test-media.sh)" >&2; fi
  done
  [[ ${#avail[@]} -gt 0 ]] || die "no media files available"

  if [[ -f "$PID_FILE" ]]; then
    echo "a simulation is already tracked in $PID_FILE; stopping it first"
    cmd_stop
  fi
  mkdir -p "$LOG_DIR"
  : >"$PID_FILE"

  if [[ $external == 0 ]]; then
    need "$MEDIAMTX"
    [[ -f "$MTX_CONFIG" ]] || die "config not found: $MTX_CONFIG"
    if api_get /v3/paths/list >/dev/null; then
      die "a MediaMTX API already answers on $API_URL; stop it or use --external"
    fi
    "$MEDIAMTX" "$(NATIVE_PATH "$MTX_CONFIG")" >"$LOG_DIR/mediamtx.log" 2>&1 </dev/null &
    echo "mediamtx $!" >>"$PID_FILE"
    echo "started mediamtx (pid $!), log $LOG_DIR/mediamtx.log"
    local i ready=0
    for i in $(seq 1 40); do
      if api_get /v3/paths/list >/dev/null; then ready=1; break; fi
      alive "$!" || break
      sleep 0.25
    done
    [[ $ready == 1 ]] || { tail -n 20 "$LOG_DIR/mediamtx.log" >&2; cmd_stop; die "MediaMTX did not start"; }
  fi

  local idx name file codec
  local -a enc
  for ((idx = 1; idx <= count; idx++)); do
    name="$prefix$idx"
    file="${avail[$(((idx - 1) % ${#avail[@]}))]}"
    codec="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 \
        "$(NATIVE_PATH "$file")" </dev/null | tr -d '\r')"
    case "$codec" in
      h264|hevc) enc=(-c:v copy) ;;
      mjpeg)
        # RFC 2435 needs standard Huffman tables and both quantisation tables;
        # files from fetch-test-media are encoded that way.
        if [[ "$mjpeg" == copy || ("$mjpeg" == auto && "$file" == "$MEDIA_DIR/"*) ]]; then
          enc=(-c:v copy)
        else
          enc=(-c:v mjpeg -q:v 5 -huffman default -force_duplicated_matrix 1 -pix_fmt yuvj420p)
        fi ;;
      *) enc=(-c:v libx264 -preset veryfast -tune zerolatency -bf 0 -g 50 -pix_fmt yuv420p) ;;
    esac
    "$FFMPEG" -hide_banner -nostdin -nostats -loglevel warning -re -stream_loop -1 -i "$(NATIVE_PATH "$file")" \
      -map 0:v:0 "${enc[@]}" -f rtsp -rtsp_transport "$transport" "$RTSP_URL/$name" \
      >"$LOG_DIR/$name.log" 2>&1 </dev/null &
    echo "$name $!" >>"$PID_FILE"
    echo "started $name (pid $!): ${file#"$MEDIA_DIR/"} [$codec, ${enc[*]:0:2}] -> $RTSP_URL/$name"
  done

  # wait until every path is ready
  local i nready=0
  for i in $(seq 1 60); do
    nready="$(api_paths 2>/dev/null | grep -c "^$prefix[0-9]* ready=true" || true)"
    [[ "$nready" -ge "$count" ]] && break
    sleep 0.5
  done
  echo "$nready/$count cameras ready; logs in $LOG_DIR; stop with: $0 stop"
  [[ "$nready" -ge "$count" ]]
}

cmd="${1:-}"; shift || true
case "$cmd" in
  start) cmd_start "$@" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  probe) need "$FFPROBE"; cmd_probe "$@" ;;
  -h|--help|help|"") usage ;;
  *) die "unknown command '$cmd' (start, stop, status, probe)" ;;
esac
