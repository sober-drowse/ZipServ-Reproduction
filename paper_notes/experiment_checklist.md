# ZipServ 实验梳理清单

## 1. 实验复现优先级

- P0：已经跑通或近期必须跑通，构成最小复现闭环。
- P1：论文核心 Evaluation，优先用于复刻主要图表。
- P2：需要额外 baseline、Nsight、模型权重或多卡资源，视时间推进。
- P3：讨论性实验或难以完全复现的补充实验。

## 2. 实验总表

| 图/表 | 实验名称 | 实验类型 | 需要代码 | 需要模型/数据 | 需要硬件 | 优先级 | 当前状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| Fig. 1 | lossless compression pipeline overhead | 动机实验 | 需要 decompression + GEMM 计时脚本 | GateUp_proj layer shape | L40S-like GPU | P2 | 未开始 | 用于说明 decoupled 解压开销可能大于 GEMM |
| Fig. 2 | exponent bit distribution | 数据分布/可视化 | 可自写 histogram 脚本 | LLM BF16 权重 | CPU/GPU 均可 | P2 | 未开始 | 需要下载模型权重，统计 BF16 exponent 分布 |
| Fig. 3 | Huffman-based BF16 lossless compression illustration | 方法/动机图 | 不需要运行 | 无 | 无 | P3 | 仅阅读 | 解释传统变长编码问题 |
| Fig. 4 | existing decoupled inference pipeline | 方法/动机图 | 不需要运行 | 无 | 无 | P3 | 仅阅读 | 解释额外 global memory traffic |
| Fig. 5 | roofline analysis | 理论/效率分析 | 可自写计算脚本 | 合成参数 | 无 | P2 | 未开始 | 可用公式复算 CI 趋势 |
| Fig. 6 | ZipServ overview | 架构图 | 不需要运行 | 无 | 无 | P3 | 已阅读 | Offline compressor + online inference engine |
| Algorithm 1 | TCA-TBE offline compressor | 方法验证 | 官方 compressor/benchmark | BF16 矩阵 | CPU/GPU | P1 | 部分完成 | `test_mm` 输出 compression statistics |
| Algorithm 2 | ZipGEMM thread-local decompression | 方法验证 | 官方 CUDA kernel | BF16 矩阵 tile | GPU | P1 | 部分完成 | 通过 `test_mm` 间接验证 |
| Fig. 10 | hierarchical software pipeline | 方法/分析 | 不单独运行 | 无 | GPU | P3 | 仅阅读 | 用于解释 ZipGEMM latency hiding |
| Fig. 11 | ZipGEMM kernel performance | 主对比实验 | `test_mm` | LLaMA/Qwen/Gemma/Mistral layer shape | RTX4090/L40S；当前 L20 替代 | P1 | 最小实验已跑通 | 当前已有 `4096 4096 128 1`，需扩展 shape |
| Fig. 12 | micro-level kernel performance analysis | 复杂度/效率实验 | Nsight Compute | 代表性 shape | RTX4090-like GPU | P2 | 未开始 | 需要 NCU 权限和 profile 脚本 |
| Fig. 13 | standalone decompression kernel comparison | 效率实验 | `test_decompress` + baseline | LLaMA/Mistral block shape | GPU | P1 | 已编译，未运行 | 先确认 `test_decompress` usage |
| Fig. 14 | cross-generation performance comparison | 硬件泛化实验 | `test_mm` | LLaMA3.1-8B/Mistral-24B GateUp shape | RTX5090/A100/H800 | P3 | 难以完整复现 | 当前 L20 只能做替代硬件分析 |
| Fig. 15 | performance under different N settings | 参数敏感性实验 | `test_mm` | 合成矩阵 shape | GPU | P1 | 未开始 | 下一步优先跑 N sweep |
| Fig. 16 | end-to-end inference performance | 端到端主实验 | vLLM integration | LLaMA3.1-8B/Mistral-24B/LLaMA3.1-70B | 1-4 GPU | P2 | 未开始 | 依赖模型权重和官方 vLLM 集成完整性 |
| Fig. 17 | latency and memory breakdown | 系统效率分析 | profiling + vLLM | LLaMA3.1-8B | GPU | P2 | 未开始 | 需要端到端跑通后分析 |
| Fig. 18 | training-oriented GPU performance | 硬件讨论实验 | `test_mm` | 代表性 GateUp shape | A100/H800 | P3 | 难以完整复现 | 当前无 A100/H800 |

## 3. 已完成实验记录

### 3.1 最小 kernel benchmark

命令：

```bash
./test_mm 4096 4096 128 1
```

日志：

```text
logs/experiments/test_mm_4096_4096_128_1.log
```

摘要：

| 方法 | Time/ms | TFLOPs | TotalError |
|---|---:|---:|---:|
| BF16_triple_bitmap | 0.139 | 30.88 | 38.99 |
| CuBLAS_TC | 0.099 | 43.22 | 0.00 |
| CuBLAS_non-TC | 0.492 | 8.74 | 38.98 |

说明：该结果只验证工程闭环，不能直接代表论文 Figure 11 结论。当前硬件为 L20，与论文主实验 RTX4090/L40S 不一致。

## 4. 下一步实验计划

### 4.1 `test_decompress` 参数确认

目标：确认 standalone decompression benchmark 的命令格式和输出字段。

计划：

```bash
cd code/ZipServ_ASPLOS26_patched/kernel_benchmark
./test_decompress
```

如果输出 usage，则按 usage 设计运行脚本。

### 4.2 Figure 15 初步复现

固定一个代表性 shape，扫描不同 N：

```text
M=4096, K=4096, SplitK=1
N in [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
```

输出：

- `results/raw/test_mm_n_sweep.csv`
- `results/processed/test_mm_n_sweep_summary.csv`
- `figures/reproduced/fig15_n_sweep_latency.png`
- `figures/reproduced/fig15_n_sweep_speedup.png`

### 4.3 Figure 11 初步复现

选择代表性 LLM layer shape，比较：

- BF16_triple_bitmap
- CuBLAS_TC
- CuBLAS_non-TC

输出：

- `results/processed/kernel_shape_summary.csv`
- `figures/reproduced/fig11_kernel_speedup.png`

## 5. 难点和风险

- 论文使用 RTX4090、L40S、RTX5090、A100、H800，当前只有 L20，性能趋势可能不同。
- DietGPU、nvCOMP、DFloat11 baseline 需要额外代码和适配，短期可能无法完整复现。
- End-to-end vLLM 实验需要模型权重、多卡、长输出长度和可能的 HuggingFace 权限。
- Nsight Compute 分析可能需要服务器权限和较长独占 GPU 时间。
