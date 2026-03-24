# RWA Rust - Building and Running Guide

## System Requirements

- Rust stable toolchain
- Cargo
- Windows / macOS / Linux

## Install Rust

Linux/macOS:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Windows:

1. Download installer from https://rustup.rs/
2. Install with default options
3. Ensure MSVC build tools are available

Verify:

```bash
rustc --version
cargo --version
```

## Build

Debug build:

```bash
cargo build
```

Release build (recommended):

```bash
cargo build --release
```

## Run

Run from source (recommended):

```bash
# Use logical CPU core count automatically
cargo run --release

# Specify worker threads
cargo run --release -- --workers 4
```

Run compiled binary:

Linux/macOS:

```bash
./target/release/rwa_rust
```

Windows PowerShell:

```powershell
.\target\release\rwa_rust.exe
```

Notes:

- Current implementation runs `100` iterations per execution.
- `--workers` (or `-w`) must be a positive integer.

## Quick Scripts

Linux/macOS:

```bash
chmod +x run.sh
./run.sh
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\run.ps1
```

## Output Files

Each run writes to a timestamped directory:

```text
output/
  YYYY-MM-DD HH-MM-SS/
    result.csv
    figure-1/
    figure-2/
```

At the same time, a global merged CSV is also updated:

```text
output/result.csv
```

## CSV Columns (Current)

The current output has 36 columns:

`T, c_t, k, f, E_D, E_P, E_V, E_DP, E_PV, E_cf, sigma_D, sigma_P, sigma_V, sigma_DP, sigma_PV, sigma_cf, m11, lambda1, theta1, m12, lambda2, theta2, m21, gamma1, mu1, m22, gamma2, mu2, q, r, p1, p2, cons_1, cons_2, cons_3, pi`

`cons_1`, `cons_2`, `cons_3` are upstream constraints and are positioned between `p2` and `pi`.

## Common Development Commands

```bash
# Fast compile check
cargo check

# Run tests
cargo test

# Format check
cargo fmt --check

# Lint
cargo clippy
```

## Troubleshooting

`cargo: command not found`:

1. Reopen terminal after Rust install
2. Ensure Cargo bin path is in PATH

Windows linker error (`link.exe` not found):

1. Install Visual Studio Build Tools (C++)
2. Restart terminal and rebuild

No output files generated:

1. Check write permission for `output/`
2. Check console warnings for CSV/plot write failures

## Notes

- All core calculations use `f64`
- CSV rows are streamed and flushed during execution
- Plot generation failures are counted and reported at the end
