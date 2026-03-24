# RWA Rust 项目说明

该项目用于模拟 RWA（Renewable Energy Wholesaler Arbitrage）三阶段博弈，并通过并行 Monte Carlo 迭代输出结果与分布图。

## 当前实现状态

- 并行执行：基于 Rayon 线程池
- 统计采样：Gamma + 多元对数正态采样
- 博弈求解：上下游玩家 + 最优响应迭代 + GA 外层优化
- 输出方式：按迭代实时写入 CSV（时间戳目录 + 全局汇总）
- 绘图：每次迭代保存价格和成本分布图

## 项目结构

```text
src/
	main.rs          # 程序入口、并行调度、CSV 写入、进度显示、统计汇总
	game.rs          # 三阶段博弈与约束、GA 目标/惩罚函数
	distribution.rs  # Gamma 与多元对数正态采样
	plotting.rs      # figure-1 / figure-2 绘图输出
	ga.rs            # 遗传算法
	utils.rs         # 统计与向量工具
```

## 运行方式

### 1) 检查编译

```bash
cargo check
```

### 2) 运行（默认线程数 = 逻辑核数）

```bash
cargo run --release
```

### 3) 指定线程数运行

```bash
cargo run --release -- --workers 4
```

说明：程序当前固定执行 100 次迭代（`main.rs` 内 `total_iterations = 100`）。

## 输出目录

每次运行会创建：

- `output/YYYY-MM-DD HH-MM-SS/result.csv`
- `output/YYYY-MM-DD HH-MM-SS/figure-1/*.png`
- `output/YYYY-MM-DD HH-MM-SS/figure-2/*.png`

同时会更新全局汇总文件：

- `output/result.csv`

## CSV 列定义（当前 36 列）

列顺序如下：

1. `T`
2. `c_t`
3. `k`
4. `f`
5. `E_D`
6. `E_P`
7. `E_V`
8. `E_DP`
9. `E_PV`
10. `E_cf`
11. `sigma_D`
12. `sigma_P`
13. `sigma_V`
14. `sigma_DP`
15. `sigma_PV`
16. `sigma_cf`
17. `m11`
18. `lambda1`
19. `theta1`
20. `m12`
21. `lambda2`
22. `theta2`
23. `m21`
24. `gamma1`
25. `mu1`
26. `m22`
27. `gamma2`
28. `mu2`
29. `q`
30. `r`
31. `p1`
32. `p2`
33. `cons_1`
34. `cons_2`
35. `cons_3`
36. `pi`

其中 `cons_1/cons_2/cons_3` 为 `UpstreamPlayer` 的三个约束函数值，位置已插入在 `p2` 与 `pi` 之间。

## 依赖（按当前 Cargo.toml）

- `argmin`
- `chrono`
- `csv`
- `mimalloc`
- `plotters`
- `rand`
- `rand_chacha`
- `rand_distr`
- `rayon`