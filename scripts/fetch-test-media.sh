#!/usr/bin/env bash
# Downloads real test clips and generates the synthetic test-media matrix
# described in testdata/manifest.json into testdata/media/ (git-ignored).
#
# Usage: scripts/fetch-test-media.sh [--quick] [--force] [--verify-only]
#                                    [--only GLOB] [--list] [--media-dir DIR]
#
#   --quick        only the small core set (entries with "quick": true)
#   --force        regenerate generated files even if they are up to date
#   --verify-only  do not download or encode, only verify existing files
#   --only GLOB    restrict to entries whose name matches GLOB (e.g. 'gating/*');
#                  inputs referenced with {media:...} must already exist
#   --list         print the selected entries and exit
#   --media-dir    output directory (default: testdata/media, env CAMTENSOR_MEDIA_DIR)
#
# Environment: CAMTENSOR_FFMPEG, CAMTENSOR_FFPROBE (default: ffmpeg/ffprobe on PATH),
#              CAMTENSOR_MEDIA_DIR.
# Requires: bash 4+, curl, sha256sum (or shasum), and jq or python3 to read the manifest.
#
# Idempotent: downloads resume (.part files) and are checked by size + SHA-256;
# generated files are skipped when they exist, their recipe is unchanged
# (stamp in .stamps/) and they pass ffprobe verification.
# Keep this script behaviourally identical to fetch-test-media.ps1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/testdata/manifest.json"
MEDIA_DIR="${CAMTENSOR_MEDIA_DIR:-$ROOT/testdata/media}"
FFMPEG="${CAMTENSOR_FFMPEG:-ffmpeg}"
FFPROBE="${CAMTENSOR_FFPROBE:-ffprobe}"
QUICK=0 FORCE=0 VERIFY_ONLY=0 LIST=0 ONLY=""

usage() { sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --force) FORCE=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --list) LIST=1 ;;
    --only) ONLY="${2:?--only needs a glob}"; shift ;;
    --media-dir) MEDIA_DIR="${2:?--media-dir needs a path}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found (set CAMTENSOR_FFMPEG / CAMTENSOR_FFPROBE or install it)"; }
need "$FFMPEG"; need "$FFPROBE"; need curl
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"

# Git Bash / MSYS (Windows smoke tests only): pass arguments to the native
# ffmpeg untouched and hand it Windows-style paths.
NATIVE_PATH() { printf '%s' "$1"; }
if command -v cygpath >/dev/null 2>&1; then
  export MSYS2_ARG_CONV_EXCL='*'
  NATIVE_PATH() { cygpath -m "$1"; }
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
else
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
fi

file_size() { wc -c <"$1" | tr -d ' '; }

# One record per manifest entry; fields separated by \x1e, lists joined by \x1f.
# name kind quick size sha256 urls ffmpeg codec profile width height fps pix_fmt
# bframes duration frames gop_frames
manifest_rows() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '
      def s: if . == null then "" else tostring end;
      .files[] | [ .name, .kind, (.quick|s), (.source.size|s), (.source.sha256|s),
        ((.source.urls // []) | join("\u001f")), ((.ffmpeg // []) | join("\u001f")),
        (.video.codec|s), (.video.profile|s), (.video.width|s), (.video.height|s),
        (.video.fps|s), (.video.pix_fmt|s), (.video.bframes|s), (.video.duration|s),
        (.video.frames|s), (.video.gop_frames|s) ] | join("\u001e")' "$MANIFEST" | tr -d '\r'
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import json' >/dev/null 2>&1; then
    python3 -I - "$(NATIVE_PATH "$MANIFEST")" <<'PY'
import json, sys
def s(v):
    if v is None: return ""
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, float) and v.is_integer(): return str(int(v))
    return str(v)
m = json.load(open(sys.argv[1], encoding="utf-8"))
for f in m["files"]:
    src, v = f.get("source") or {}, f.get("video") or {}
    row = [f["name"], f["kind"], s(f.get("quick")), s(src.get("size")), s(src.get("sha256")),
           "\x1f".join(src.get("urls") or []), "\x1f".join(f.get("ffmpeg") or [])]
    row += [s(v.get(k)) for k in ("codec", "profile", "width", "height", "fps", "pix_fmt",
                                  "bframes", "duration", "frames", "gop_frames")]
    sys.stdout.write("\x1e".join(row) + "\n")
PY
  else
    die "need jq or python3 to read $MANIFEST"
  fi
}

# ------------------------------------------------------------------ verify
# verify FILE codec profile width height fps pix_fmt bframes duration frames gop
# Prints a one-line summary, returns non-zero on mismatch.
VERIFY_SUMMARY=""
verify() {
  local f="$1" codec="$2" profile="$3" width="$4" height="$5" fps="$6" pix="$7"
  local bframes="$8" duration="$9" frames="${10}" gop="${11}"
  local out errs=() k v
  out="$("$FFPROBE" -v error -select_streams v:0 \
      -show_entries stream=codec_name,profile,width,height,pix_fmt,has_b_frames,r_frame_rate:format=duration \
      -of default=noprint_wrappers=1 "$(NATIVE_PATH "$f")" </dev/null 2>&1 | tr -d '\r')" || {
    VERIFY_SUMMARY="ffprobe failed: ${out%%$'\n'*}"; return 1; }
  declare -A p=()
  while IFS='=' read -r k v; do [[ -n "$k" ]] && p[$k]="$v"; done <<<"$out"
  [[ "${p[codec_name]:-}" == "$codec" ]] || errs+=("codec ${p[codec_name]:-?} != $codec")
  [[ -z "$profile" || "${p[profile]:-}" == "$profile" ]] || errs+=("profile '${p[profile]:-?}' != '$profile'")
  [[ "${p[width]:-}" == "$width" && "${p[height]:-}" == "$height" ]] ||
    errs+=("size ${p[width]:-?}x${p[height]:-?} != ${width}x${height}")
  [[ -z "$pix" || "${p[pix_fmt]:-}" == "$pix" ]] || errs+=("pix_fmt ${p[pix_fmt]:-?} != $pix")
  # Raw Annex B has no container timing; ffprobe guesses its frame rate.
  if [[ -n "$fps" && "$f" != *.h264 && "$f" != *.hevc ]]; then
    awk -v r="${p[r_frame_rate]:-0/1}" -v e="$fps" \
      'BEGIN { split(r, a, "/"); f = (a[2] > 0) ? a[1] / a[2] : 0; exit !(f - e < 0.01 && e - f < 0.01) }' ||
      errs+=("fps ${p[r_frame_rate]:-?} != $fps")
  fi
  if [[ -n "$bframes" ]]; then
    local hb="false" nb="${p[has_b_frames]:-0}"
    [[ "$nb" =~ ^[0-9]+$ && "$nb" -gt 0 ]] && hb="true"
    [[ "$hb" == "$bframes" ]] || errs+=("bframes $hb != $bframes")
  fi
  local dur="${p[duration]:-}"
  if [[ -n "$duration" ]]; then
    awk -v d="${dur:-0}" -v e="$duration" 'BEGIN { exit !(d - e <= 0.15 && e - d <= 0.15) }' ||
      errs+=("duration ${dur:-?} != $duration")
  fi
  local nframes="" nkey=""
  if [[ -n "$frames" || -n "$gop" ]]; then
    read -r nframes nkey < <("$FFPROBE" -v error -select_streams v:0 -show_entries packet=flags \
        -of csv=p=0 "$(NATIVE_PATH "$f")" </dev/null | awk '{ n++ } /K/ { k++ } END { print n+0, k+0 }')
    [[ -z "$frames" || "$nframes" == "$frames" ]] || errs+=("frames $nframes != $frames")
    if [[ -n "$gop" && -n "$frames" ]]; then
      local ekey=$(( (frames + gop - 1) / gop ))
      [[ "$nkey" == "$ekey" ]] || errs+=("keyframes $nkey != $ekey")
    fi
  fi
  VERIFY_SUMMARY="${p[codec_name]:-?} ${p[profile]:+${p[profile]} }${p[width]:-?}x${p[height]:-?} ${p[pix_fmt]:-?}"
  [[ -n "$dur" && "$dur" != "N/A" ]] && VERIFY_SUMMARY+=" $(printf '%.2f' "$dur")s"
  [[ -n "$nframes" ]] && VERIFY_SUMMARY+=" ${nframes}f/${nkey}k"
  if [[ ${#errs[@]} -gt 0 ]]; then
    VERIFY_SUMMARY+=" MISMATCH: $(IFS=';'; echo "${errs[*]}")"
    return 1
  fi
}

# --------------------------------------------------------------- download
download() {
  local dest="$1" size="$2" sha="$3" urls="$4" url part
  if [[ -f "$dest" && "$(file_size "$dest")" == "$size" && "$(sha256_of "$dest")" == "$sha" ]]; then
    echo "  cached"; return 0
  fi
  rm -f "$dest"
  part="$dest.part"
  local -a list
  IFS=$'\x1f' read -r -a list <<<"$urls"
  for url in "${list[@]}"; do
    if [[ -f "$part" && "$(file_size "$part")" -gt "$size" ]]; then rm -f "$part"; fi
    echo "  GET $url"
    if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -sS -C - -o "$(NATIVE_PATH "$part")" "$url"; then
      # a server that ignores ranges or a broken partial: start over once
      rm -f "$part"
      curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -sS -o "$(NATIVE_PATH "$part")" "$url" || { echo "  failed"; continue; }
    fi
    if [[ "$(file_size "$part")" != "$size" ]]; then
      echo "  size mismatch ($(file_size "$part") != $size)"; rm -f "$part"; continue
    fi
    if [[ "$(sha256_of "$part")" != "$sha" ]]; then
      echo "  sha256 mismatch"; rm -f "$part"; continue
    fi
    mv -f "$part" "$dest"
    return 0
  done
  return 1
}

# --------------------------------------------------------------- generate
generate() {
  local name="$1" args="$2" dest="$MEDIA_DIR/$1" stamp="$MEDIA_DIR/.stamps/$1.recipe"
  local log="$MEDIA_DIR/.logs/$1.log" a ref
  local -a raw=() argv=()
  IFS=$'\x1f' read -r -a raw <<<"$args"
  for a in "${raw[@]}"; do
    while [[ "$a" =~ \{media:([^}]+)\} ]]; do
      ref="${BASH_REMATCH[1]}"
      [[ -f "$MEDIA_DIR/$ref" ]] || { echo "  missing input $ref (run without --only/--quick first)"; return 1; }
      a="${a//\{media:$ref\}/$(NATIVE_PATH "$MEDIA_DIR/$ref")}"
    done
    argv+=("$a")
  done
  mkdir -p "$(dirname "$dest")" "$(dirname "$stamp")" "$(dirname "$log")"
  local tmp; tmp="$(dirname "$dest")/.partial-$(basename "$dest")"
  rm -f "$tmp" "$stamp"
  if ! "$FFMPEG" -hide_banner -nostdin -y -v error "${argv[@]}" "$(NATIVE_PATH "$tmp")" </dev/null >"$log" 2>&1; then
    echo "  ffmpeg failed, see $log:"; tail -n 5 "$log" | sed 's/^/    /'
    rm -f "$tmp"; return 1
  fi
  mv -f "$tmp" "$dest"
  printf '%s\n' "${raw[@]}" >"$stamp"
}

recipe_matches() { # name args
  local stamp="$MEDIA_DIR/.stamps/$1.recipe"
  local -a raw=()
  [[ -f "$stamp" ]] || return 1
  IFS=$'\x1f' read -r -a raw <<<"$2"
  [[ "$(printf '%s\n' "${raw[@]}")" == "$(tr -d '\r' <"$stamp")" ]]
}

# ------------------------------------------------------------------- main
mkdir -p "$MEDIA_DIR"
echo "manifest : $MANIFEST"
echo "media dir: $MEDIA_DIR"
echo "ffmpeg   : $("$FFMPEG" -hide_banner -version </dev/null | head -n1 | tr -d '\r')"
echo "mode     : $([[ $QUICK == 1 ]] && echo quick || echo full)$([[ $VERIFY_ONLY == 1 ]] && echo ', verify only')"
echo

ok=0 failed=0 total=0
failures=()
summary=()
while IFS=$'\x1e' read -r -u 3 name kind quick size sha urls args codec profile width height fps pix bframes duration frames gop; do
  [[ -z "$name" ]] && continue
  [[ $QUICK == 1 && "$quick" != "true" ]] && continue
  # shellcheck disable=SC2053
  [[ -n "$ONLY" && "$name" != $ONLY ]] && continue
  if [[ $LIST == 1 ]]; then printf '%-8s %-5s %s\n' "$kind" "$quick" "$name"; continue; fi
  total=$((total + 1))
  dest="$MEDIA_DIR/$name"
  echo "[$total] $name"
  mkdir -p "$(dirname "$dest")"
  status=0
  if [[ $VERIFY_ONLY == 1 ]]; then
    [[ -f "$dest" ]] || { echo "  missing"; status=1; }
  elif [[ "$kind" == "download" ]]; then
    download "$dest" "$size" "$sha" "$urls" || { echo "  all mirrors failed"; status=1; }
  elif [[ "$kind" == "ffmpeg" ]]; then
    if [[ $FORCE == 0 && -f "$dest" ]] && recipe_matches "$name" "$args" &&
       verify "$dest" "$codec" "$profile" "$width" "$height" "$fps" "$pix" "$bframes" "$duration" "$frames" "$gop"; then
      echo "  up to date"
    else
      echo "  encoding..."
      generate "$name" "$args" || status=1
    fi
  else
    echo "  unknown kind '$kind'"; status=1
  fi
  if [[ $status == 0 ]]; then
    if verify "$dest" "$codec" "$profile" "$width" "$height" "$fps" "$pix" "$bframes" "$duration" "$frames" "$gop"; then
      echo "  OK  $VERIFY_SUMMARY"
    else
      echo "  FAIL $VERIFY_SUMMARY"; status=1
    fi
  fi
  if [[ $status == 0 ]]; then
    ok=$((ok + 1))
    summary+=("$(printf '%8.1f MB  %s' "$(awk -v b="$(file_size "$dest")" 'BEGIN { print b / 1048576 }')" "$name")")
  else
    failed=$((failed + 1)); failures+=("$name")
  fi
done 3< <(manifest_rows | tr -d '\r')

[[ $LIST == 1 ]] && exit 0
echo
printf '%s\n' "${summary[@]}"
echo
echo "$ok/$total OK, $failed failed (media dir: $MEDIA_DIR)"
if [[ $failed -gt 0 ]]; then
  printf '  failed: %s\n' "${failures[@]}"
  exit 1
fi
