# RWA - Renewable Energy Wholesaler Arbitrage Simulation (Rust)

Rust 版本的 RWA 三阶段博弈仿真程序。当前实现以并行 Monte Carlo 为主流程，在每次迭代中完成采样、博弈求解、CSV 落盘和图像输出。

## Overview

三层参与者：

- 上游（Upstream）：储能提供方，优化 `q/r/p1/p2`
- 中层（Stage One）：可再生能源发电方
- 下游（Stage Two）：购电方

核心流程：

1. 采样需求、价格、价值、运行成本
2. 运行多阶段博弈与 GA
3. 生成一行结果并立即写入 CSV
4. 按迭代输出两张分布图

## Current Project Structure

```text
src/
	main.rs          # 入口，线程配置，进度显示，CSV 写入，汇总统计
	distribution.rs  # Gamma 与多元对数正态采样
	game.rs          # 玩家建模、约束函数、最优响应与外层 GA
	ga.rs            # 遗传算法实现
	plotting.rs      # 每次迭代输出 figure-1/figure-2
	utils.rs         # 均值、标准差、向量工具
```

## Build and Run

### Prerequisites

- Rust toolchain（建议 stable）
- Cargo

### Build

```bash
cargo build --release
```

### Run

```bash
# 默认线程数 = 逻辑 CPU 核数
cargo run --release

# 指定线程数
cargo run --release -- --workers 4
```

说明：当前迭代次数固定为 100。

## Output

每次运行会生成：

- `output/YYYY-MM-DD HH-MM-SS/result.csv`
- `output/YYYY-MM-DD HH-MM-SS/figure-1/*.png`
- `output/YYYY-MM-DD HH-MM-SS/figure-2/*.png`

并同步写入：

- `output/result.csv`

程序会在终端打印多线程进度，并在结束时输出各列的均值、标准差、最小值、最大值。

## CSV Schema (36 columns)

完整列顺序：

`T, c_t, k, f, E_D, E_P, E_V, E_DP, E_PV, E_cf, sigma_D, sigma_P, sigma_V, sigma_DP, sigma_PV, sigma_cf, m11, lambda1, theta1, m12, lambda2, theta2, m21, gamma1, mu1, m22, gamma2, mu2, q, r, p1, p2, cons_1, cons_2, cons_3, pi`

其中：

- `cons_1`, `cons_2`, `cons_3` 为 `UpstreamPlayer` 三个约束函数值
- 这三列位于 `p2` 和 `pi` 之间

## Dependencies (Current)

- `argmin`
- `chrono`
- `csv`
- `mimalloc`
- `plotters`
- `rand`
- `rand_chacha`
- `rand_distr`
- `rayon`

## Notes

- 使用 `f64` 进行计算
- CSV 采用流式写入，减少中间结果堆积
- 绘图失败与 CSV 写入失败会统计并在结束时给出 warning
