#!/bin/bash
# Quick Start Guide for RWA Rust Project

echo "RWA (Renewable Energy Wholesaler Arbitrage) - Rust Edition"
echo "=========================================================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is not installed!"
    echo "📥 Install from https://rustup.rs/"
    exit 1
fi

echo "✅ Rust toolchain found: $(rustc --version)"
echo ""

# Build the project
echo "📦 Building Rust project..."
echo ""
cargo build --release

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""
echo "🚀 Running simulation..."
echo ""

# Run the simulation
./target/release/rwa

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Simulation completed successfully!"
    echo ""
    echo "📊 Results saved to:"
    latest_output=$(ls -dt output/*/ | head -1)
    echo "   $latest_output"
    echo ""
    echo "📋 Output structure:"
    echo "   - result.csv: Simulation results"
    echo "   - figure-1/: Reserved for visualizations"
    echo "   - figure-2/: Reserved for visualizations"
    echo ""
else
    echo "❌ Simulation failed!"
    exit 1
fi
