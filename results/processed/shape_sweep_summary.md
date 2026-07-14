# Shape sweep 实验结果汇总

## 实验设置

- N = 32
- SplitK = 1
- 对比对象：BF16_triple_bitmap、cuBLAS_TC、cuBLAS_non-TC

## 结果表

| Shape | M | K | ZipServ Time/ms | cuBLAS TC Time/ms | Speedup vs cuBLAS TC | Compression Ratio |
|---|---:|---:|---:|---:|---:|---:|
| square_4096 | 4096 | 4096 | 0.0980 | 0.0770 | 0.7857 | 1.42x |
| llama_ffn_gateup_11008x4096 | 11008 | 4096 | 0.1410 | 0.1980 | 1.4043 | 1.42x |
| mistral_like_gateup_28672x4096 | 28672 | 4096 | 0.3910 | 0.4990 | 1.2762 | 1.42x |
| down_proj_4096x14336 | 4096 | 14336 | 0.2900 | 0.2440 | 0.8414 | 1.42x |

## 说明

该实验是论文 Figure 11 的简化复现，使用当前 NVIDIA L20 服务器对若干代表性 LLM linear layer shape 进行 kernel-level benchmark。由于硬件和 baseline 覆盖与原论文不同，该图主要用于展示复现流程和趋势分析。

生成图表：

- `figures/reproduced/fig11_shape_sweep_latency.png`
- `figures/reproduced/fig11_shape_sweep_latency.pdf`
- `figures/reproduced/fig11_shape_sweep_speedup.png`
- `figures/reproduced/fig11_shape_sweep_speedup.pdf`
