# N sweep 实验结果汇总

## 实验设置

- M = 4096
- K = 4096
- SplitK = 1
- N = 1, 8, 16, 32, 64, 128, 256, 512

## 结果表

| N | ZipServ Time/ms | cuBLAS TC Time/ms | cuBLAS non-TC Time/ms | Speedup vs cuBLAS TC | Compression Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0980 | 0.0760 | 0.0740 | 0.7755 | 1.42x |
| 8 | 0.0960 | 0.0770 | 0.4510 | 0.8021 | 1.42x |
| 16 | 0.0970 | 0.0770 | 0.4500 | 0.7938 | 1.42x |
| 32 | 0.0980 | 0.0770 | 0.4590 | 0.7857 | 1.42x |
| 64 | 0.1060 | 0.0860 | 0.4990 | 0.8113 | 1.42x |
| 128 | 0.1390 | 0.1000 | 0.5010 | 0.7194 | 1.42x |
| 256 | 0.1640 | 0.1190 | 0.5390 | 0.7256 | 1.42x |
| 512 | 0.3090 | 0.2350 | 1.0620 | 0.7605 | 1.42x |

## 说明

该实验是论文 Figure 15 的简化复现。当前硬件为 NVIDIA L20，和论文中的 RTX4090/L40S 不一致，因此主要用于观察趋势和形成可追溯复现实验流程。

生成图表：

- `figures/reproduced/fig15_n_sweep_latency.png`
- `figures/reproduced/fig15_n_sweep_latency.pdf`
- `figures/reproduced/fig15_n_sweep_speedup.png`
- `figures/reproduced/fig15_n_sweep_speedup.pdf`