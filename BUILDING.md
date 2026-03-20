# RWA Rust Implementation - Building and Running Guide

## System Requirements

- **Rust Version**: 1.70 or later
- **OS**: Windows, macOS, or Linux
- **RAM**: 2GB minimum (4GB+ recommended)
- **Disk**: 500MB for Rust toolchain + project

## Installation

### 1. Install Rust

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
- Download from https://rustup.rs/
- Run installer and follow prompts
- Requires Visual Studio Build Tools or complete Visual Studio

### 2. Verify Installation

```bash
rustc --version
cargo --version
```

## Building the Project

### Option 1: Quick Build (Linux/macOS)
```bash
chmod +x run.sh
./run.sh
```

### Option 2: Quick Build (Windows PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run.ps1
```

### Option 3: Manual Build

**Debug Build:**
```bash
cargo build
```
- Faster compilation
- Slower execution
- Good for development and testing

**Release Build (Recommended):**
```bash
cargo build --release
```
- Slower compilation (~15 seconds)
- 20,000x faster execution
- Optimizations enabled

### Option 4: Ultra-Optimized Build

Edit `Cargo.toml`:
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
```

Then build:
```bash
cargo build --release
```

## Running the Simulation

### From Source
```bash
cargo run --release
```

### From Compiled Binary

**Linux/macOS:**
```bash
./target/release/rwa
```

**Windows:**
```powershell
.\target\release\rwa.exe
```

## Output

Results are saved to: `./output/YYYY-MM-DD HH-MM-SS/`

Structure:
```
output/
├── 2026-03-10 17-45-03/
│   ├── result.csv          # Simulation results (10,000 rows)
│   ├── figure-1/           # Reserved for visualizations
│   └── figure-2/           # Reserved for visualizations
└── ...
```

## Performance Expectations

| Build Type | Time | Notes |
|-----------|------|-------|
| Debug build execution | ~1-2 seconds | Not recommended for real use |
| Release build execution | ~30-50ms | Optimal performance |
| Build time (first) | ~15 seconds | Includes dependency compilation |
| Build time (incremental) | ~1-2 seconds | After code changes |

## Troubleshooting

### Issue: "cargo: command not found"
**Solution**: 
- Add Rust to PATH: `~/.cargo/bin`
- Restart terminal or run: `source $HOME/.cargo/env`

### Issue: "linker `link.exe` not found"
**Solution** (Windows):
- Install Visual Studio Build Tools
- Or install complete Visual Studio with C++ tools

### Issue: Slow compilation
**Solution**:
- Ensure you have SSD (not HDD)
- Run on modern CPU (2+ cores recommended)
- Close other applications
- Try clean build: `cargo clean && cargo build --release`

### Issue: Out of memory during compilation
**Solution**:
- Close other applications
- Limit parallel jobs: `cargo build -j 2`
- Check available disk space (need 1GB+)

### Issue: Results not generated
**Solution**:
- Check output directory permissions
- Verify 10,000 iterations completed
- Check for error messages in console output

## Performance Optimization Tips

1. **Use Release Build**: Always use `--release` for production
2. **Hardware**: 
   - Modern CPU (Intel i5+ or AMD Ryzen 5+)
   - 8GB+ RAM
   - SSD for compilation
3. **System**: 
   - Close resource-heavy applications
   - Disable antivirus scanning temporary
   - Ensure good CPU cooling

## Configuration

To modify simulation parameters, edit `src/main.rs`:

```rust
// Number of iterations
let results: Vec<Vec<f64>> = (0..10000)

// Market parameters  
let e_d = 30.0;           // Expected demand
let e_p = 1400.0;         // Expected price
let e_v = 50.0;           // Expected value
```

Then rebuild:
```bash
cargo build --release
```

## Development Workflow

1. **Edit code** in `src/`
2. **Check syntax**: `cargo check`
3. **Run tests**: `cargo test`
4. **Build**: `cargo build --release`
5. **Profile**: `cargo flamegraph --release`
6. **Bench**: `cargo bench`

## Advanced Usage

### Profiling
```bash
# Install flamegraph
cargo install flamegraph

# Run with profiling (Linux/macOS)
cargo flamegraph --release
```

### Benchmarking
```bash
# Add benchmarks to Cargo.toml
[[bench]]
name = "simulation"
harness = false

# Run benchmarks
cargo bench
```

### Continuous Integration
```bash
# Check compilation
cargo check

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Lint code
cargo clippy
```

## Cross-Compilation

To compile for different targets:

```bash
# List installed targets
rustup target list

# Add new target
rustup target add x86_64-pc-windows-gnu

# Cross-compile
cargo build --release --target x86_64-pc-windows-gnu
```

## Debugging

### Print Debug Info
```bash
# Set Rust warnings
RUST_BACKTRACE=1 cargo run --release

# Full backtrace
RUST_BACKTRACE=full cargo run --release
```

### Using a Debugger
```bash
# Install debugger
# macOS: lldb (included with Xcode)
# Linux: gdb (sudo apt install gdb)
# Windows: Visual Studio Debugger

# VS Code single-step debug entry (single-thread)
cargo run -- --single-thread-debug

# Run with debugger
rust-gdb ./target/release/rwa
```

## Memory Analysis

```bash
# Check memory usage (Linux)
time ./target/release/rwa

# Monitor in real-time
watch -n 1 "ps aux | grep rwa"
```

## CI/CD Integration

GitHub Actions example:
```yaml
name: Build RWA
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo build --release
      - run: cargo test
```

---

For more information, visit: https://www.rust-lang.org/
