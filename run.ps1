# Quick Start Guide for RWA Rust Project (PowerShell)

Write-Host "RWA (Renewable Energy Wholesaler Arbitrage) - Rust Edition" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Rust is installed
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Rust is not installed!" -ForegroundColor Red
    Write-Host "📥 Install from https://rustup.rs/" -ForegroundColor Yellow
    exit 1
}

$rustVersion = cargo --version
Write-Host "✅ Rust toolchain found: $rustVersion" -ForegroundColor Green
Write-Host ""

# Build the project
Write-Host "📦 Building Rust project..." -ForegroundColor Cyan
Write-Host ""
cargo build --release

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Build successful!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Running simulation..." -ForegroundColor Cyan
Write-Host ""

# Run the simulation
.\target\release\rwa.exe

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Simulation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Results saved to:" -ForegroundColor Cyan
    $latestOutput = Get-ChildItem "output" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
    Write-Host "   $latestOutput" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 Output structure:" -ForegroundColor Cyan
    Write-Host "   - result.csv: Simulation results" -ForegroundColor Gray
    Write-Host "   - figure-1/: Reserved for visualizations" -ForegroundColor Gray
    Write-Host "   - figure-2/: Reserved for visualizations" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "❌ Simulation failed!" -ForegroundColor Red
    exit 1
}
