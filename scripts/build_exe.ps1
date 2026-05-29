# SPDX-License-Identifier: AGPL-3.0-or-later
# Build a Windows .exe of Nocturne Data Forge with PyInstaller.
#
#   pwsh -File scripts/build_exe.ps1
#
# Output: dist\NocturneDataForge\NocturneDataForge.exe (one-dir)
$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> Ensuring build tools (pyinstaller)..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pyinstaller

Write-Host "==> Prefetching tiktoken encoding for offline use..." -ForegroundColor Cyan
$cache = Join-Path $root ".build\tiktoken_cache"
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$env:TIKTOKEN_CACHE_DIR = $cache
try {
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken cached')"
} catch {
    Write-Warning "tiktoken prefetch failed (no network?); exe will fetch it on first run."
}

Write-Host "==> Running PyInstaller..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean nocturne.spec

$exe = Join-Path $root "dist\NocturneDataForge\NocturneDataForge.exe"
if (Test-Path $exe) {
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "==> Built: $exe ($size MB)" -ForegroundColor Green
    Write-Host "    Ship the whole 'dist\NocturneDataForge' folder (zip it)."
} else {
    Write-Error "Build finished but exe not found at $exe"
    exit 1
}
