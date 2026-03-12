# RWA - Rust Rewrite

This is a Rust rewrite of the original Python project for Renewable Energy Wholesaler Arbitrage (RWA) simulation.

## Project Structure

- `src/main.rs`: Main entry point with async simulation logic
- `src/game.rs`: Game theory models for players (upstream, stage one, stage two)
- `src/ga.rs`: Genetic algorithm implementation for optimization
- `src/utils.rs`: Utility functions for statistical sampling and optimization

## Key Components

### Game Theory Models
- **UpstreamPlayer**: Represents the energy storage provider
- **StageOneGame**: Renewable energy generators
- **StageTwoGame**: Electricity offtakers

### Optimization
- Genetic Algorithm for parameter optimization
- Constraint-based best response functions
- Statistical sampling (Gamma, Lognormal distributions)

### Simulation
- Multi-threaded Monte Carlo simulations
- CSV output for results
- Directory structure for organized outputs

## Dependencies

- `rand` & `rand_distr`: Random number generation
- `nalgebra`: Linear algebra operations
- `argmin`: Optimization framework
- `tokio`: Async runtime
- `chrono`: Date/time handling
- `csv`: CSV file handling
- `serde`: Serialization

## Building and Running

```bash
cargo build --release
cargo run
```

## Differences from Python Version

1. **Performance**: Rust provides better performance and memory safety
2. **Concurrency**: Uses async/await for parallel simulations
3. **Type Safety**: Compile-time guarantees prevent runtime errors
4. **Simplified Sampling**: Current implementation uses independent sampling; full multivariate lognormal to be implemented

## TODO

- Implement full multivariate lognormal sampling
- Complete the genetic algorithm integration
- Add CSV output functionality
- Implement plotting with plotters crate
- Add comprehensive error handling

## Original Python Features

The original Python project included:
- Complex game theory simulations
- Genetic algorithm optimization
- Statistical analysis and visualization
- Multi-threaded execution

This Rust version aims to replicate and improve upon these features while leveraging Rust's strengths in performance and safety.

---

Original README:

A project for simulating multi-player in RWA market

测试结果：
1. 价格和需求波动大时，投资者倾向于减少投资
2. 在价格/需求波动小时，双方投资者都会投资，但是上层指定的r（分红）比例很高，同时p2也会接近于f（市场销售单价） ***
3. 