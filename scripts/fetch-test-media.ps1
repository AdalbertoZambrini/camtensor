<#
.SYNOPSIS
  Downloads real test clips and generates the synthetic test-media matrix
  described in testdata/manifest.json into testdata/media/ (git-ignored).

.DESCRIPTION
  Behaviour is identical to scripts/fetch-test-media.sh:
  - downloads resume (.part files) and are checked by size + SHA-256, trying
    each mirror in turn;
  - generated files are skipped when they exist, their recipe is unchanged
    (stamp in .stamps/) and they pass ffprobe verification;
  - every file is verified with ffprobe against the manifest (codec, profile,
    size, pixel format, fps, B-frames, duration, frame and keyframe counts).

  Tools: env CAMTENSOR_FFMPEG / CAMTENSOR_FFPROBE, else ffmpeg/ffprobe on PATH,
  else C:\mediamtx-ffmpeg\ffmpeg\bin. Output: env CAMTENSOR_MEDIA_DIR or -MediaDir.
  Works with Windows PowerShell 5.1 and PowerShell 7.

.PARAMETER Quick       Only the small core set (entries with "quick": true).
.PARAMETER Force       Regenerate generated files even if they are up to date.
.PARAMETER VerifyOnly  Do not download or encode, only verify existing files.
.PARAMETER Only        Restrict to entries whose name matches this wildcard (e.g. 'gating/*').
.PARAMETER List        Print the selected entries and exit.
.PARAMETER MediaDir    Output directory (default testdata/media).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\fetch-test-media.ps1 -Quick
#>
[CmdletBinding()]
param(
  [switch]$Quick,
  [switch]$Force,
  [switch]$VerifyOnly,
  [string]$Only = "",
  [switch]$List,
  [string]$MediaDir = ""
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$Inv = [Globalization.CultureInfo]::InvariantCulture

$Root = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $Root 'testdata/manifest.json'
if (-not $MediaDir) { $MediaDir = if ($env:CAMTENSOR_MEDIA_DIR) { $env:CAMTENSOR_MEDIA_DIR } else { Join-Path $Root 'testdata/media' } }
New-Item -ItemType Directory -Force -Path $MediaDir | Out-Null
$MediaDir = (Resolve-Path $MediaDir).Path

function Find-Tool([string]$EnvName, [string]$Name, [string]$Fallback) {
  $v = [Environment]::GetEnvironmentVariable($EnvName)
  if ($v) { return $v }
  $c = Get-Command $Name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  if ($Fallback -and (Test-Path $Fallback)) { return $Fallback }
  throw "'$Name' not found: set $EnvName or put it on PATH"
}
$FFmpeg = Find-Tool 'CAMTENSOR_FFMPEG' 'ffmpeg' 'C:\mediamtx-ffmpeg\ffmpeg\bin\ffmpeg.exe'
$FFprobe = Find-Tool 'CAMTENSOR_FFPROBE' 'ffprobe' 'C:\mediamtx-ffmpeg\ffmpeg\bin\ffprobe.exe'
if (-not (Test-Path $Manifest)) { throw "manifest not found: $Manifest" }

# Runs a native tool without PowerShell turning stderr into errors.
# Returns @{ Code; Out } (stdout lines); stderr goes to $LogFile when given.
function Invoke-Tool([string]$Exe, [string[]]$ArgList, [string]$LogFile = "") {
  $old = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    if ($LogFile) {
      & $Exe @ArgList 2>&1 | ForEach-Object { "$_" } | Set-Content -Encoding UTF8 -Path $LogFile
      $out = @()
    } else {
      $out = @(& $Exe @ArgList 2>$null)
    }
    return @{ Code = $LASTEXITCODE; Out = $out }
  } finally { $ErrorActionPreference = $old }
}

function Get-Prop($Obj, [string]$Name) {
  if ($null -eq $Obj) { return $null }
  $p = $Obj.PSObject.Properties[$Name]
  if ($p) { return $p.Value } else { return $null }
}

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant() }

# ------------------------------------------------------------------ verify
function Test-Media([string]$File, $V) {
  $errs = New-Object System.Collections.Generic.List[string]
  $r = Invoke-Tool $FFprobe @('-v', 'error', '-select_streams', 'v:0', '-show_entries',
    'stream=codec_name,profile,width,height,pix_fmt,has_b_frames,r_frame_rate:format=duration',
    '-of', 'default=noprint_wrappers=1', $File)
  if ($r.Code -ne 0) { return @{ Ok = $false; Summary = 'ffprobe failed' } }
  $p = @{}
  foreach ($line in $r.Out) {
    $i = "$line".IndexOf('=')
    if ($i -gt 0) { $p["$line".Substring(0, $i)] = "$line".Substring($i + 1).Trim() }
  }
  function P([string]$k) { if ($p.ContainsKey($k)) { $p[$k] } else { '' } }

  $codec = Get-Prop $V 'codec'; $profile = Get-Prop $V 'profile'
  $width = Get-Prop $V 'width'; $height = Get-Prop $V 'height'
  $fps = Get-Prop $V 'fps'; $pix = Get-Prop $V 'pix_fmt'; $bframes = Get-Prop $V 'bframes'
  $duration = Get-Prop $V 'duration'; $frames = Get-Prop $V 'frames'; $gop = Get-Prop $V 'gop_frames'
  $isRaw = $File -match '\.(h264|hevc)$'

  if ((P 'codec_name') -ne $codec) { $errs.Add("codec $(P 'codec_name') != $codec") }
  if ($profile -and (P 'profile') -ne $profile) { $errs.Add("profile '$(P 'profile')' != '$profile'") }
  if ((P 'width') -ne "$width" -or (P 'height') -ne "$height") { $errs.Add("size $(P 'width')x$(P 'height') != ${width}x${height}") }
  if ($pix -and (P 'pix_fmt') -ne $pix) { $errs.Add("pix_fmt $(P 'pix_fmt') != $pix") }
  # Raw Annex B has no container timing; ffprobe guesses its frame rate.
  if ($null -ne $fps -and -not $isRaw) {
    $rf = (P 'r_frame_rate') -split '/'
    $f = 0.0
    if ($rf.Count -eq 2 -and [double]$rf[1] -gt 0) { $f = [double]::Parse($rf[0], $Inv) / [double]::Parse($rf[1], $Inv) }
    if ([math]::Abs($f - [double]$fps) -ge 0.01) { $errs.Add("fps $(P 'r_frame_rate') != $fps") }
  }
  if ($null -ne $bframes) {
    $nb = 0; [void][int]::TryParse((P 'has_b_frames'), [ref]$nb)
    $hb = $nb -gt 0
    if ($hb -ne [bool]$bframes) { $errs.Add("bframes $("$hb".ToLower()) != $("$bframes".ToLower())") }
  }
  $dur = P 'duration'
  $durVal = $null
  if ($dur -and $dur -ne 'N/A') { $durVal = [double]::Parse($dur, $Inv) }
  if ($null -ne $duration) {
    if ($null -eq $durVal -or [math]::Abs($durVal - [double]$duration) -gt 0.15) { $errs.Add("duration $dur != $duration") }
  }
  $nframes = $null; $nkey = $null
  if ($null -ne $frames -or $null -ne $gop) {
    $pk = Invoke-Tool $FFprobe @('-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=flags', '-of', 'csv=p=0', $File)
    $nframes = 0; $nkey = 0
    foreach ($l in $pk.Out) { if ("$l".Trim()) { $nframes++; if ("$l" -match 'K') { $nkey++ } } }
    if ($null -ne $frames -and $nframes -ne [int]$frames) { $errs.Add("frames $nframes != $frames") }
    if ($null -ne $gop -and $null -ne $frames) {
      $ekey = [int][math]::Floor(([int]$frames + [int]$gop - 1) / [int]$gop)
      if ($nkey -ne $ekey) { $errs.Add("keyframes $nkey != $ekey") }
    }
  }
  $s = "$(P 'codec_name') "
  if (P 'profile') { $s += "$(P 'profile') " }
  $s += "$(P 'width')x$(P 'height') $(P 'pix_fmt')"
  if ($null -ne $durVal) { $s += ' ' + $durVal.ToString('0.00', $Inv) + 's' }
  if ($null -ne $nframes) { $s += " ${nframes}f/${nkey}k" }
  if ($errs.Count -gt 0) { return @{ Ok = $false; Summary = "$s MISMATCH: $($errs -join ';')" } }
  return @{ Ok = $true; Summary = $s }
}

# --------------------------------------------------------------- download
$Curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
function Get-Download([string]$Dest, $Src) {
  $size = [int64]$Src.size; $sha = $Src.sha256
  if ((Test-Path $Dest) -and (Get-Item $Dest).Length -eq $size -and (Get-Sha256 $Dest) -eq $sha) {
    Write-Host '  cached'; return $true
  }
  Remove-Item -Force -ErrorAction SilentlyContinue $Dest
  $part = "$Dest.part"
  foreach ($url in $Src.urls) {
    if ((Test-Path $part) -and (Get-Item $part).Length -gt $size) { Remove-Item -Force $part }
    Write-Host "  GET $url"
    $okDl = $false
    if ($Curl) {
      $r = Invoke-Tool $Curl.Source @('-fL', '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20', '-sS', '-C', '-', '-o', $part, $url)
      if ($r.Code -ne 0) {
        # a server that ignores ranges or a broken partial: start over once
        Remove-Item -Force -ErrorAction SilentlyContinue $part
        $r = Invoke-Tool $Curl.Source @('-fL', '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20', '-sS', '-o', $part, $url)
      }
      $okDl = $r.Code -eq 0
    } else {
      try {
        $pp = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $part
        $okDl = $true
      } catch { $okDl = $false } finally { $ProgressPreference = $pp }
    }
    if (-not $okDl) { Write-Host '  failed'; continue }
    $len = (Get-Item $part).Length
    if ($len -ne $size) { Write-Host "  size mismatch ($len != $size)"; Remove-Item -Force $part; continue }
    if ((Get-Sha256 $part) -ne $sha) { Write-Host '  sha256 mismatch'; Remove-Item -Force $part; continue }
    Move-Item -Force $part $Dest
    return $true
  }
  return $false
}

# --------------------------------------------------------------- generate
function Get-StampPath([string]$Name) { Join-Path $MediaDir ".stamps/$Name.recipe" }
function Get-RecipeText([string[]]$Raw) { ($Raw -join "`n") + "`n" }

function Test-Recipe([string]$Name, [string[]]$Raw) {
  $stamp = Get-StampPath $Name
  if (-not (Test-Path $stamp)) { return $false }
  $have = [IO.File]::ReadAllText($stamp).Replace("`r", '')
  return $have -eq (Get-RecipeText $Raw)
}

function New-Media([string]$Name, [string[]]$Raw) {
  $dest = Join-Path $MediaDir $Name
  $stamp = Get-StampPath $Name
  $log = Join-Path $MediaDir ".logs/$Name.log"
  $argv = New-Object System.Collections.Generic.List[string]
  foreach ($a in $Raw) {
    $s = $a
    while ($s -match '\{media:([^}]+)\}') {
      $ref = $Matches[1]
      $refPath = Join-Path $MediaDir $ref
      if (-not (Test-Path $refPath)) { Write-Host "  missing input $ref (run without -Only/-Quick first)"; return $false }
      $s = $s.Replace("{media:$ref}", (Resolve-Path $refPath).Path)
    }
    $argv.Add($s)
  }
  foreach ($d in @((Split-Path -Parent $dest), (Split-Path -Parent $stamp), (Split-Path -Parent $log))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
  }
  $tmp = Join-Path (Split-Path -Parent $dest) (".partial-" + (Split-Path -Leaf $dest))
  Remove-Item -Force -ErrorAction SilentlyContinue $tmp, $stamp
  $all = @('-hide_banner', '-nostdin', '-y', '-v', 'error') + $argv.ToArray() + @($tmp)
  $r = Invoke-Tool $FFmpeg $all $log
  if ($r.Code -ne 0) {
    Write-Host "  ffmpeg failed, see ${log}:"
    Get-Content $log -Tail 5 | ForEach-Object { Write-Host "    $_" }
    Remove-Item -Force -ErrorAction SilentlyContinue $tmp
    return $false
  }
  Move-Item -Force $tmp $dest
  [IO.File]::WriteAllText($stamp, (Get-RecipeText $Raw))
  return $true
}

# ------------------------------------------------------------------- main
$m = Get-Content -Raw -Encoding UTF8 $Manifest | ConvertFrom-Json
$ver = (Invoke-Tool $FFmpeg @('-hide_banner', '-version')).Out | Select-Object -First 1
Write-Host "manifest : $Manifest"
Write-Host "media dir: $MediaDir"
Write-Host "ffmpeg   : $ver"
$mode = if ($Quick) { 'quick' } else { 'full' }
if ($VerifyOnly) { $mode += ', verify only' }
Write-Host "mode     : $mode"
Write-Host ''

$ok = 0; $failed = 0; $total = 0
$failures = New-Object System.Collections.Generic.List[string]
$summary = New-Object System.Collections.Generic.List[string]
foreach ($e in $m.files) {
  $name = $e.name
  if ($Quick -and -not $e.quick) { continue }
  if ($Only -and -not ($name -like $Only)) { continue }
  if ($List) { Write-Host ('{0,-8} {1,-5} {2}' -f $e.kind, "$($e.quick)".ToLower(), $name); continue }
  $total++
  $dest = Join-Path $MediaDir $name
  Write-Host "[$total] $name"
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
  $raw = @(Get-Prop $e 'ffmpeg')
  $good = $true
  $result = $null
  if ($VerifyOnly) {
    if (-not (Test-Path $dest)) { Write-Host '  missing'; $good = $false }
  } elseif ($e.kind -eq 'download') {
    if (-not (Get-Download $dest $e.source)) { Write-Host '  all mirrors failed'; $good = $false }
  } elseif ($e.kind -eq 'ffmpeg') {
    $upToDate = $false
    if (-not $Force -and (Test-Path $dest) -and (Test-Recipe $name $raw)) {
      $result = Test-Media $dest $e.video
      $upToDate = $result.Ok
    }
    if ($upToDate) { Write-Host '  up to date' }
    else {
      Write-Host '  encoding...'
      $result = $null
      $good = New-Media $name $raw
    }
  } else { Write-Host "  unknown kind '$($e.kind)'"; $good = $false }
  if ($good) {
    if ($null -eq $result) { $result = Test-Media $dest $e.video }
    if ($result.Ok) { Write-Host "  OK  $($result.Summary)" } else { Write-Host "  FAIL $($result.Summary)"; $good = $false }
  }
  if ($good) {
    $ok++
    $mb = (Get-Item $dest).Length / 1MB
    $summary.Add(('{0,8} MB  {1}' -f $mb.ToString('0.0', $Inv), $name))
  } else { $failed++; $failures.Add($name) }
}

if ($List) { exit 0 }
Write-Host ''
$summary | ForEach-Object { Write-Host $_ }
Write-Host ''
Write-Host "$ok/$total OK, $failed failed (media dir: $MediaDir)"
if ($failed -gt 0) {
  $failures | ForEach-Object { Write-Host "  failed: $_" }
  exit 1
}
exit 0
