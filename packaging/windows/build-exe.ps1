<#
.SYNOPSIS
  Build the PdfCraft desktop executable.

.DESCRIPTION
  Runs `cargo build -p pdfcraft --release` from the repository root and copies
  pdfcraft.exe to dist/pdfcraft.exe. Honours CARGO_TARGET_DIR when set.
  Does not sign, package an installer, or copy the binary elsewhere.

.EXAMPLE
  pwsh packaging/windows/build-exe.ps1
#>
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

Write-Output '==> cargo build -p pdfcraft --release'
cargo build -p pdfcraft --release
if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }

$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root 'target' }
if (-not [System.IO.Path]::IsPathRooted($target)) {
  $target = Join-Path $Root $target
}
$exe = Join-Path $target 'release/pdfcraft.exe'
if (-not (Test-Path $exe)) {
  throw "expected $exe after a successful build"
}
$dist = Join-Path $Root 'dist'
New-Item -ItemType Directory -Force -Path $dist | Out-Null
$out = Join-Path $dist 'pdfcraft.exe'
Copy-Item -Force $exe $out
Write-Output "==> $out"
