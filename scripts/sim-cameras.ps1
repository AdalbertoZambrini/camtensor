<#
.SYNOPSIS
  Simulates RTSP cameras: starts MediaMTX with deploy/mediamtx.yml and publishes
  looping media files as rtsp://127.0.0.1:8554/cam1..camN.

.DESCRIPTION
  Commands (same behaviour as scripts/sim-cameras.sh):
    start   start MediaMTX (unless -External) and N looping publishers
    probe   ffprobe every published camera over RTSP/TCP, decoding ~3 s of video
    status  tracked processes + MediaMTX path list
    stop    stop publishers and MediaMTX

  H.264/H.265 files are published with -c copy. MJPEG is copied when the file
  comes from testdata/media (encoded RFC 2435 compatible by fetch-test-media),
  re-encoded otherwise (-Mjpeg auto); other codecs are re-encoded to H.264.

  Tools: env CAMTENSOR_FFMPEG / CAMTENSOR_FFPROBE / CAMTENSOR_MEDIAMTX, else on
  PATH, else C:\mediamtx-ffmpeg\{ffmpeg\bin,mediamtx}. Other environment:
  CAMTENSOR_MEDIAMTX_CONFIG, CAMTENSOR_MEDIA_DIR, CAMTENSOR_RTSP_URL, CAMTENSOR_API_URL,
  PROBE_SECONDS (seconds decoded per camera by probe, default 10).
  Logs and PIDs: target/sim-logs/.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 start -Count 4 -Preset mixed
  powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 probe
  powershell -ExecutionPolicy Bypass -File scripts\sim-cameras.ps1 stop
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)][ValidateSet('start', 'stop', 'status', 'probe', 'help')][string]$Command = 'help',
  [int]$Count = 4,
  [ValidateSet('', 'h264-1080p', 'h265-1080p', 'mjpeg-720p', 'mixed', 'gating')][string]$Preset = '',
  [string[]]$Files = @(),
  [string]$Prefix = 'cam',
  [ValidateSet('tcp', 'udp')][string]$Transport = 'tcp',
  [ValidateSet('auto', 'copy', 'reencode')][string]$Mjpeg = 'auto',
  [switch]$External,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Names = @()
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
function Get-EnvOr([string]$Name, [string]$Default) { $v = [Environment]::GetEnvironmentVariable($Name); if ($v) { $v } else { $Default } }
$MtxConfig = Get-EnvOr 'CAMTENSOR_MEDIAMTX_CONFIG' (Join-Path $Root 'deploy/mediamtx.yml')
$MediaDir = Get-EnvOr 'CAMTENSOR_MEDIA_DIR' (Join-Path $Root 'testdata/media')
$RtspUrl = Get-EnvOr 'CAMTENSOR_RTSP_URL' 'rtsp://127.0.0.1:8554'
$ApiUrl = Get-EnvOr 'CAMTENSOR_API_URL' 'http://127.0.0.1:9997'
$LogDir = Join-Path $Root 'target/sim-logs'
$PidFile = Join-Path $LogDir 'pids'
if (Test-Path $MediaDir) { $MediaDir = (Resolve-Path $MediaDir).Path }

function Find-Tool([string]$EnvName, [string]$Name, [string]$Fallback) {
  $v = [Environment]::GetEnvironmentVariable($EnvName)
  if ($v) { return $v }
  $c = Get-Command $Name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  if ($Fallback -and (Test-Path $Fallback)) { return $Fallback }
  throw "'$Name' not found: set $EnvName or put it on PATH"
}

$Presets = @{
  'h264-1080p' = @('h264/h264-high-1920x1080-25fps-gop2s-vbr-6m.mp4')
  'h265-1080p' = @('h265/h265-main-1920x1080-25fps-gop2s-vbr-4m.mp4')
  'mjpeg-720p' = @('mjpeg/mjpeg-1280x720-25fps-q5.mkv')
  'mixed'      = @('h264/h264-high-1920x1080-25fps-gop2s-vbr-6m.mp4',
                   'h265/h265-main-1920x1080-25fps-gop2s-vbr-4m.mp4',
                   'mjpeg/mjpeg-1280x720-25fps-q5.mkv',
                   'h264/h264-baseline-640x360-25fps-gop1s-vbr-500k.mp4')
  'gating'     = @('gating/gating-h264-1080p25-gop2s-vbr.mp4',
                   'gating/gating-h265-1080p25-gop2s-vbr.mp4',
                   'gating/gating-h264-1080p25-gop2s-cbr-4m.mp4',
                   'gating/gating-h265-1080p25-gop4s-cbr-3m.mp4',
                   'gating/gating-mjpeg-640x360-15fps-q10.mkv')
}

# Quotes one argument for a Windows command line (CommandLineToArgvW rules).
function ConvertTo-Arg([string]$a) {
  if ($a -ne '' -and $a -notmatch '[\s"]') { return $a }
  $s = $a -replace '(\\*)"', '$1$1\"'
  $s = $s -replace '(\\+)$', '$1$1'
  return '"' + $s + '"'
}
function Join-Args([string[]]$List) { ($List | ForEach-Object { ConvertTo-Arg $_ }) -join ' ' }

function Invoke-Tool([string]$Exe, [string[]]$ArgList) {
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { $out = @(& $Exe @ArgList 2>$null); return @{ Code = $LASTEXITCODE; Out = $out } }
  finally { $ErrorActionPreference = $old }
}

function Read-Pids {
  if (-not (Test-Path $PidFile)) { return @() }
  @(Get-Content $PidFile | Where-Object { $_.Trim() } | ForEach-Object {
      $p = $_ -split '\s+'; [pscustomobject]@{ Name = $p[0]; Pid = [int]$p[1] } })
}

# Returns the process only if it is still one of our tools (guards against PID reuse).
function Get-OurProcess([int]$Id) {
  $p = Get-Process -Id $Id -ErrorAction SilentlyContinue
  if ($p -and ($p.ProcessName -match '^(ffmpeg|mediamtx)$')) { return $p }
  return $null
}

function Test-Api {
  try { Invoke-RestMethod -UseBasicParsing -TimeoutSec 2 -Uri "$ApiUrl/v3/paths/list" | Out-Null; return $true } catch { return $false }
}

# Path list (empty when the API is not reachable; check with Test-Api).
function Get-ApiPaths {
  try { $r = Invoke-RestMethod -UseBasicParsing -TimeoutSec 2 -Uri "$ApiUrl/v3/paths/list" } catch { return }
  foreach ($it in @($r.items)) {
    [pscustomobject]@{ Name = $it.name; Ready = [bool]$it.ready; Tracks = (@($it.tracks) -join ','); Readers = @($it.readers).Count }
  }
}

function Invoke-Stop {
  $pids = @(Read-Pids)
  if ($pids.Count -eq 0) { Write-Host "not running (no $PidFile)"; return }
  # publishers first, MediaMTX last
  foreach ($e in @($pids | Where-Object Name -ne 'mediamtx') + @($pids | Where-Object Name -eq 'mediamtx')) {
    $p = Get-OurProcess $e.Pid
    if ($p) {
      Stop-Process -Id $e.Pid -Force -ErrorAction SilentlyContinue
      Write-Host "stopped $($e.Name) (pid $($e.Pid))"
    }
  }
  foreach ($e in $pids) { try { Wait-Process -Id $e.Pid -Timeout 5 -ErrorAction SilentlyContinue } catch {} }
  Remove-Item -Force $PidFile
}

function Invoke-Status {
  $pids = @(Read-Pids)
  if ($pids.Count -eq 0) { Write-Host "no processes tracked ($PidFile missing)" }
  foreach ($e in $pids) {
    $state = if (Get-OurProcess $e.Pid) { 'running' } else { 'DEAD' }
    Write-Host ('{0,-10} pid {1,-7} {2}' -f $e.Name, $e.Pid, $state)
  }
  Write-Host "MediaMTX paths (${ApiUrl}):"
  if (-not (Test-Api)) { Write-Host '  API not reachable'; return }
  $paths = @(Get-ApiPaths)
  foreach ($p in $paths) { Write-Host "  $($p.Name) ready=$("$($p.Ready)".ToLower()) tracks=$($p.Tracks) readers=$($p.Readers)" }
}

# Probes every camera over RTSP/TCP in parallel, decoding PROBE_SECONDS (default
# 10) of video, enough to reach a keyframe for GOPs up to ~8 s. Decoder messages
# (expected when joining mid-GOP) go to target/sim-logs/probe-NAME.log.
function Invoke-Probe([string[]]$Which) {
  $ffprobe = Find-Tool 'CAMTENSOR_FFPROBE' 'ffprobe' 'C:\mediamtx-ffmpeg\ffmpeg\bin\ffprobe.exe'
  $secs = [int](Get-EnvOr 'PROBE_SECONDS' '10')
  if ($Which.Count -eq 0) {
    $Which = @(Read-Pids | Where-Object Name -ne 'mediamtx' | ForEach-Object Name)
    if ($Which.Count -eq 0) { $Which = @(Get-ApiPaths | ForEach-Object Name) }
  }
  if ($Which.Count -eq 0) { throw 'nothing to probe' }
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $procs = @{}
  foreach ($n in $Which) {
    $plog = Join-Path $LogDir "probe-$n.log"
    $argList = @('-v', 'error', '-rtsp_transport', 'tcp', '-timeout', '5000000', '-read_intervals', "%+$secs",
      '-select_streams', 'v:0', '-count_frames',
      '-show_entries', 'stream=codec_name,profile,width,height,pix_fmt,nb_read_frames',
      '-of', 'default=noprint_wrappers=1', "$RtspUrl/$n")
    $procs[$n] = Start-Process -FilePath $ffprobe -ArgumentList (Join-Args $argList) -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput "$plog.out" -RedirectStandardError $plog
  }
  $deadline = (Get-Date).AddSeconds($secs + 20)
  foreach ($n in $Which) {
    $ms = [int][math]::Max(0, ($deadline - (Get-Date)).TotalMilliseconds)
    if (-not $procs[$n].WaitForExit($ms)) { Stop-Process -Id $procs[$n].Id -Force -ErrorAction SilentlyContinue }
  }
  $fails = 0
  foreach ($n in $Which) {
    $plog = Join-Path $LogDir "probe-$n.log"
    $out = ((Get-Content "$plog.out" -ErrorAction SilentlyContinue) -join ' ').Trim()
    Remove-Item -Force -ErrorAction SilentlyContinue "$plog.out"
    $nerr = @(Get-Content $plog -ErrorAction SilentlyContinue).Count
    $frames = 0
    if ($out -match 'nb_read_frames=(\d+)') { $frames = [int]$Matches[1] }
    if ($frames -gt 0 -and $out -notmatch 'width=0 ') {
      Write-Host "OK   ${n}: $out (decoder messages: $nerr)"
    } else {
      $err = (@(Get-Content $plog -TotalCount 3 -ErrorAction SilentlyContinue) -join ' ')
      Write-Host "FAIL ${n}: $(if ($out) { $out } else { 'no output' }) $err"
      $fails++
    }
  }
  if ($fails -gt 0) { exit 1 }
}

function Invoke-Start {
  if ($Count -lt 1) { throw '-Count must be >= 1' }
  $ffmpeg = Find-Tool 'CAMTENSOR_FFMPEG' 'ffmpeg' 'C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe'
  $ffprobe = Find-Tool 'CAMTENSOR_FFPROBE' 'ffprobe' 'C:\mediamtx-ffmpeg\ffmpeg\bin\ffprobe.exe'
  # -Files accepts a list or a comma-separated string (powershell -File cannot pass arrays)
  $list = @(@($Files) + @($Names) | ForEach-Object { $_ -split "," } | Where-Object { $_ })
  if ($list.Count -eq 0) { $list = $Presets[$(if ($Preset) { $Preset } else { 'mixed' })] }
  elseif ($Preset) { throw 'give either -Preset or files, not both' }
  $avail = New-Object System.Collections.Generic.List[string]
  foreach ($f in $list) {
    $p = if ([IO.Path]::IsPathRooted($f)) { $f } else { Join-Path $MediaDir $f }
    if (Test-Path $p) { $avail.Add((Resolve-Path $p).Path) }
    else { Write-Warning "missing $p (run scripts\fetch-test-media.ps1)" }
  }
  if ($avail.Count -eq 0) { throw 'no media files available' }

  if (Test-Path $PidFile) { Write-Host "a simulation is already tracked in $PidFile; stopping it first"; Invoke-Stop }
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  Set-Content -Path $PidFile -Value @() -Encoding ASCII

  if (-not $External) {
    $mediamtx = Find-Tool 'CAMTENSOR_MEDIAMTX' 'mediamtx' 'C:\mediamtx-ffmpeg\mediamtx\mediamtx.exe'
    if (-not (Test-Path $MtxConfig)) { throw "config not found: $MtxConfig" }
    if (Test-Api) { throw "a MediaMTX API already answers on $ApiUrl; stop it or use -External" }
    $mlog = Join-Path $LogDir 'mediamtx.log'
    Remove-Item -Force -ErrorAction SilentlyContinue $mlog
    # Background processes are started without redirection (which would make
    # them inherit our stdout and keep a caller's pipe open): MediaMTX writes
    # its own log file, configured through MTX_* environment overrides.
    $env:MTX_LOGDESTINATIONS = 'file'; $env:MTX_LOGFILE = $mlog
    try {
      $mp = Start-Process -FilePath $mediamtx -ArgumentList (ConvertTo-Arg (Resolve-Path $MtxConfig).Path) -WindowStyle Hidden -PassThru
    } finally { Remove-Item Env:MTX_LOGDESTINATIONS, Env:MTX_LOGFILE -ErrorAction SilentlyContinue }
    Add-Content -Path $PidFile -Value "mediamtx $($mp.Id)" -Encoding ASCII
    Write-Host "started mediamtx (pid $($mp.Id)), log $mlog"
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
      if (Test-Api) { $ready = $true; break }
      if ($mp.HasExited) { break }
      Start-Sleep -Milliseconds 250
    }
    if (-not $ready) { Get-Content $mlog -Tail 20 -ErrorAction SilentlyContinue | Write-Host; Invoke-Stop; throw 'MediaMTX did not start' }
  }

  for ($idx = 1; $idx -le $Count; $idx++) {
    $name = "$Prefix$idx"
    $file = $avail[($idx - 1) % $avail.Count]
    $codec = "$((Invoke-Tool $ffprobe @('-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', $file)).Out | Select-Object -First 1)".Trim()
    switch ($codec) {
      { $_ -in 'h264', 'hevc' } { $enc = @('-c:v', 'copy') }
      'mjpeg' {
        # RFC 2435 needs standard Huffman tables and both quantisation tables;
        # files from fetch-test-media are encoded that way.
        if ($Mjpeg -eq 'copy' -or ($Mjpeg -eq 'auto' -and $file.StartsWith($MediaDir, [StringComparison]::OrdinalIgnoreCase))) {
          $enc = @('-c:v', 'copy')
        } else {
          $enc = @('-c:v', 'mjpeg', '-q:v', '5', '-huffman', 'default', '-force_duplicated_matrix', '1', '-pix_fmt', 'yuvj420p')
        }
      }
      default { $enc = @('-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency', '-bf', '0', '-g', '50', '-pix_fmt', 'yuv420p') }
    }
    $argList = @('-hide_banner', '-nostdin', '-nostats', '-loglevel', 'warning', '-re', '-stream_loop', '-1', '-i', $file,
      '-map', '0:v:0') + $enc + @('-f', 'rtsp', '-rtsp_transport', $Transport, "$RtspUrl/$name")
    $log = Join-Path $LogDir "$name.log"
    # ffmpeg writes its own log (FFREPORT; ':' must be escaped, use forward slashes)
    $env:FFREPORT = 'file=' + (($log -replace '\\', '/') -replace ':', '\:') + ':level=24'
    try {
      $pp = Start-Process -FilePath $ffmpeg -ArgumentList (Join-Args $argList) -WindowStyle Hidden -PassThru
    } finally { Remove-Item Env:FFREPORT -ErrorAction SilentlyContinue }
    Add-Content -Path $PidFile -Value "$name $($pp.Id)" -Encoding ASCII
    $rel = if ($file.StartsWith($MediaDir, [StringComparison]::OrdinalIgnoreCase)) { $file.Substring($MediaDir.Length + 1) } else { $file }
    Write-Host "started $name (pid $($pp.Id)): $rel [$codec, $($enc[0..1] -join ' ')] -> $RtspUrl/$name"
  }

  $nready = 0
  for ($i = 0; $i -lt 60; $i++) {
    $nready = @(Get-ApiPaths | Where-Object { $_.Ready -and $_.Name -match "^$([regex]::Escape($Prefix))\d+$" }).Count
    if ($nready -ge $Count) { break }
    Start-Sleep -Milliseconds 500
  }
  Write-Host "$nready/$Count cameras ready; logs in $LogDir; stop with: scripts\sim-cameras.ps1 stop"
  if ($nready -lt $Count) { exit 1 }
}

switch ($Command) {
  'start' { Invoke-Start }
  'stop' { Invoke-Stop }
  'status' { Invoke-Status }
  'probe' { Invoke-Probe $Names }
  default { Get-Help $PSCommandPath -Detailed }
}
