# ZipServ 论文复现报告

论文：ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression  
会议：ASPLOS 2026  
官方代码：https://github.com/HPMLL/ZipServ_ASPLOS26  
复现仓库：https://github.com/sober-drowse/ZipServ-Reproduction  

## 1. 复现目标与范围

本次复现目标是按照论文复现任务要求，完成 ZipServ 论文的阅读梳理、服务器环境搭建、官方代码适配、核心 kernel benchmark 运行、实验日志保存、结果表格与复刻图表整理，并对复现过程中的问题进行复盘。

由于本次实验使用的服务器硬件为 NVIDIA L20，而论文主实验涉及 RTX4090、L40S、RTX5090、A100、H800 等多类 GPU，并且端到端实验还依赖 vLLM 集成、大模型权重和多 GPU 配置，因此本次复现重点放在官方开源仓库中可直接运行和验证的 kernel-level 实验上。当前已完成：

- 官方代码获取与 CUDA 12.0 兼容性修复。
- ZipServ core library、`test_mm`、`test_decompress` 编译。
- 最小 `test_mm` 闭环实验。
- standalone decompression 正确性与性能实验。
- Figure 15 对应的 N sweep 简化复现。
- Figure 11 对应的 shape sweep 简化复现。
- 实验日志、结果 CSV、复刻 PNG/PDF 图表归档。

尚未完整复现的部分包括 DietGPU、nvCOMP、DFloat11 等外部 baseline，Nsight Compute 细粒度 profiling，以及 vLLM 端到端推理实验。这些内容在报告后文作为复现限制和后续工作说明。

## 2. 论文研读阶段

### 2.1 论文核心问题

大模型推理中的权重矩阵规模很大，推理时频繁读取权重会造成显存容量和显存带宽压力。传统有损压缩方法可以减少存储或计算开销，但会引入精度损失；传统无损压缩方法能够保持 bit-exact，但通常需要先把权重完整解压到显存，再调用 GEMM，导致额外的 global memory traffic 和解压开销。

ZipServ 试图解决的问题是：如何让无损压缩不仅减少模型权重存储，还能在 GPU 推理时带来实际加速。

### 2.2 模型整体架构

ZipServ 由两个核心部分组成：

1. Offline Compressor：离线压缩 BF16 权重。论文观察到 LLM 权重的 BF16 exponent 分布高度集中，因此提出 TCA-TBE 压缩格式，用 base exponent、3-bit offset、triple bitmap、sign/mantissa buffer 和 fallback buffer 表示权重。
2. Online Inference Engine：在线推理阶段根据 prefill 和 decode 的计算特性采用不同策略。Prefill 阶段更偏 compute-bound，适合 decoupled decompression + GEMM；decode 阶段更偏 memory-bound，使用 fused decompression-GEMM，即 ZipGEMM。

ZipGEMM 的关键思想是 load-compressed, compute-decompressed：从显存读取压缩权重，在寄存器中即时恢复 BF16 值，然后直接进入 Tensor Core MMA，避免把完整解压后的权重重新写回 global memory。

### 2.3 主要创新点

- 利用 BF16 exponent 分布集中这一结构特征，而不是使用通用 Huffman/ANS 变长熵编码。
- 设计 GPU 友好的 TCA-TBE 固定长度压缩格式，减少 SIMT 分支和串行依赖。
- 将解压逻辑与 GEMM kernel 融合，降低解压后权重落回 global memory 的额外开销。
- 区分 prefill 与 decode 阶段，采用 stage-aware inference 策略。

### 2.4 实验大类梳理

| 实验类别 | 论文对应内容 | 目的 | 本次复现状态 |
|---|---|---|---|
| 动机实验 | Figure 1, 3, 4, 5 | 说明传统无损压缩与 decoupled pipeline 的开销 | 阅读梳理，部分用 decompression 实验支撑 |
| 主对比实验 | Figure 11, 16, 17 | 对比 ZipServ 与 cuBLAS、vLLM、Transformers、DFloat11 等 | 完成 Figure 11 简化 kernel-level 复现 |
| 消融/效率实验 | Figure 12, 13 | 分析 kernel 内部效率与 standalone decompression | 完成 `test_decompress` 最小复现 |
| 参数敏感性实验 | Figure 15 | 分析不同 N 下性能变化 | 完成 N sweep 简化复现 |
| 硬件泛化实验 | Figure 14, 18 | 比较不同 GPU 代际 | 当前仅在 L20 上复现，无法覆盖原文全部 GPU |
| 可视化案例 | Figure 2 等 | 展示 BF16 exponent 分布和系统流程 | 阅读梳理，未下载完整模型权重统计 |

### 2.5 官方代码检索

官方开源代码位于：

```text
https://github.com/HPMLL/ZipServ_ASPLOS26
```

本复现仓库中保留了经过 CUDA 12.0 兼容性修复的完整工程代码：

```text
code/ZipServ_ASPLOS26_patched/
```

相对官方代码的兼容性修改以 patch 形式保存：

```text
patches/zipserv_cuda12_bf16_host_fix.patch
```

## 3. 环境搭建阶段

### 3.1 服务器环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA L20 x 2 |
| Driver | 595.71.05 |
| CUDA Toolkit | 12.0 |
| nvcc 路径 | `/usr/bin/nvcc` |
| gcc/g++ | 13.3.0 |
| make | 4.3 |
| Conda 环境 | `zipserv` |
| Python | 3.10 |
| 项目路径 | `~/ZipServ-Reproduction` |

完整环境文件：

```text
env/environment.yml
env/requirements.txt
env/environment.md
```

### 3.2 一键部署与运行脚本

一键编译脚本：

```bash
bash scripts/setup_zipserv.sh
```

最小实验运行脚本：

```bash
bash scripts/run_minimal_test.sh 0 4096 4096 128 1
```

N sweep 运行与解析：

```bash
bash scripts/run_n_sweep.sh 0
python scripts/parse_and_plot_n_sweep.py
```

Shape sweep 运行与解析：

```bash
bash scripts/run_shape_sweep.sh 0
python scripts/parse_and_plot_shape_sweep.py
```

### 3.3 踩坑记录

| 问题 | 现象 | 原因 | 解决方案 |
|---|---|---|---|
| GitHub HTTPS 克隆失败 | `Failed to connect to github.com port 443` | 服务器访问 GitHub HTTPS 不稳定 | 改用 GitHub SSH，并配置 SSH key |
| CUDA 路径不匹配 | `make: /usr/local/cuda/bin/nvcc: No such file or directory` | 官方 Makefile 默认寻找 `/usr/local/cuda`，服务器实际为 `/usr/bin/nvcc` | 编译时显式传入 `CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr` |
| CUDA BF16 host/device 编译错误 | host 函数调用 `__device__` BF16 转换函数报错 | CUDA 12.0 下部分 BF16 intrinsic 只能在 device 端调用 | 添加 host 端 `std::memcpy` bit 转换函数并替换 host 调用 |
| benchmark 编译 warning | `printf` 格式与 `uint64_t` 不匹配 | 调试输出格式问题 | 不影响 benchmark 正确编译和运行，暂未修改 |
| conda 初次使用 ToS 问题 | `Terms of Service have not been accepted` | Anaconda channel 新版条款需要确认 | 执行 `conda tos accept ...` 后创建环境 |

详细记录见：

```text
docs/pitfalls.md
```

## 4. 代码复现与实验运行

### 4.1 最小 kernel benchmark

命令：

```bash
./test_mm 4096 4096 128 1
```

日志：

```text
logs/experiments/test_mm_4096_4096_128_1.log
```

结果：

| 方法 | Time/ms | TFLOPs | TotalError |
|---|---:|---:|---:|
| BF16_triple_bitmap | 0.139 | 30.88 | 38.99 |
| cuBLAS_TC | 0.099 | 43.22 | 0.00 |
| cuBLAS_non-TC | 0.492 | 8.74 | 38.98 |

该实验说明官方 kernel benchmark 已能在当前 L20 服务器上完成编译和运行，是后续批量实验的最小闭环。

### 4.2 Standalone decompression 实验

命令：

```bash
./test_decompress 128 128 128
```

日志：

```text
logs/experiments/test_decompress_128_128_128.log
```

正确性结果：

| 指标 | 结果 |
|---|---:|
| Total elements | 16384 |
| Identical | 16384 (100.00%) |
| Different | 0 (0.00%) |
| Max absolute difference | 0.000000 |
| Max relative difference | 0.000000 |
| Total absolute error | 0.000000 |

性能结果：

| 项目 | 时间 |
|---|---:|
| Decompression time | 0.0060 ms |
| cuBLAS non-TC | 0.0208 ms |
| cuBLAS TC | 0.0069 ms |
| Decompression + cuBLAS non-TC | 0.0268 ms |
| Decompression + cuBLAS TC | 0.0129 ms |

该结果验证了解压流程的 lossless 正确性，也说明在小规模形状上 standalone decompression 相对 GEMM 本身仍有可见开销。

### 4.3 N sweep 参数敏感性实验

固定：

```text
M = 4096, K = 4096, SplitK = 1
N = 1, 8, 16, 32, 64, 128, 256, 512
```

结果表：

| N | ZipServ Time/ms | cuBLAS TC Time/ms | cuBLAS non-TC Time/ms | Speedup vs cuBLAS TC | Compression Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.098 | 0.076 | 0.074 | 0.776 | 1.42x |
| 8 | 0.096 | 0.077 | 0.451 | 0.802 | 1.42x |
| 16 | 0.097 | 0.077 | 0.450 | 0.794 | 1.42x |
| 32 | 0.098 | 0.077 | 0.459 | 0.786 | 1.42x |
| 64 | 0.106 | 0.086 | 0.499 | 0.811 | 1.42x |
| 128 | 0.139 | 0.100 | 0.501 | 0.719 | 1.42x |
| 256 | 0.164 | 0.119 | 0.539 | 0.726 | 1.42x |
| 512 | 0.309 | 0.235 | 1.062 | 0.761 | 1.42x |

生成图表：

```text
figures/reproduced/fig15_n_sweep_latency.png
figures/reproduced/fig15_n_sweep_latency.pdf
figures/reproduced/fig15_n_sweep_speedup.png
figures/reproduced/fig15_n_sweep_speedup.pdf
```

观察：在 L20 当前配置下，ZipServ 相对 cuBLAS_TC 的 speedup 小于 1，即没有超过 Tensor Core cuBLAS；但相对 cuBLAS_non-TC 有明显优势。该趋势与论文主张中的 memory-efficient kernel 思路相关，但由于硬件和 benchmark 覆盖不同，不能直接等同于原文曲线。

### 4.4 Shape sweep 主对比简化实验

固定：

```text
N = 32, SplitK = 1
```

结果表：

| Shape | M | K | ZipServ Time/ms | cuBLAS TC Time/ms | Speedup vs cuBLAS TC | Compression Ratio |
|---|---:|---:|---:|---:|---:|---:|
| square_4096 | 4096 | 4096 | 0.098 | 0.077 | 0.786 | 1.42x |
| llama_ffn_gateup_11008x4096 | 11008 | 4096 | 0.141 | 0.198 | 1.404 | 1.42x |
| mistral_like_gateup_28672x4096 | 28672 | 4096 | 0.391 | 0.499 | 1.276 | 1.42x |
| down_proj_4096x14336 | 4096 | 14336 | 0.290 | 0.244 | 0.841 | 1.42x |

生成图表：

```text
figures/reproduced/fig11_shape_sweep_latency.png
figures/reproduced/fig11_shape_sweep_latency.pdf
figures/reproduced/fig11_shape_sweep_speedup.png
figures/reproduced/fig11_shape_sweep_speedup.pdf
```

观察：ZipServ 在两个 FFN GateUp 类 shape 上快于 cuBLAS_TC，但在 square 和 down projection 类 shape 上慢于 cuBLAS_TC。这说明压缩 GEMM 的收益与矩阵形状、N、硬件带宽/计算比例密切相关，也与论文中强调的 stage-aware 和 hardware-aware 设计相一致。

## 5. 结果对比与误差分析

### 5.1 与原文实验的对应关系

| 原文实验 | 本次复现材料 | 说明 |
|---|---|---|
| Figure 11 kernel performance | `shape_sweep` CSV 与图表 | 只覆盖内置 cuBLAS_TC/cuBLAS_non-TC baseline，未覆盖 DietGPU/nvCOMP/DFloat11 |
| Figure 13 decompression | `test_decompress_128_128_128.log` | 完成最小正确性与性能验证 |
| Figure 15 N sensitivity | `n_sweep` CSV 与图表 | 完成 4096x4096 shape 下 N=1 到 512 的简化扫描 |
| 动机实验 | decompression 与阅读笔记 | 通过实验输出展示解压开销，但未完整复刻原文所有动机图 |
| End-to-end inference | 未完成 | 需要模型权重、vLLM 集成、多 GPU 与更长实验时间 |

### 5.2 误差来源

1. 硬件不一致：本次使用 NVIDIA L20，原文使用 RTX4090、L40S、RTX5090、A100、H800 等。
2. CUDA 与编译器不一致：本次为 CUDA Toolkit 12.0、gcc/g++ 13.3.0，原文环境不同。
3. baseline 覆盖不完整：当前主要比较官方 benchmark 内置的 cuBLAS_TC 和 cuBLAS_non-TC。
4. shape 覆盖不完整：本次选择若干代表性 shape，并非原文所有模型层。
5. kernel 参数未完全调优：SplitK、tile 配置和不同 GPU 的最优参数可能不同。
6. 端到端系统未复现：kernel-level speedup 不一定直接等价于完整推理系统 speedup。

### 5.3 图表复刻说明

本次复刻图表均由实验日志自动解析生成，包含 PNG 和 PDF 两种格式。由于缺少原论文完整原始数据，当前采用“原文图类型 + 本次复现实验数据”的方式重绘，不进行逐点完全对齐。

## 6. 问题复盘与可复现性评价

### 6.1 复现障碍

| 障碍 | 影响 | 解决情况 |
|---|---|---|
| 服务器 GitHub HTTPS 不稳定 | 无法直接克隆官方代码 | 改用 SSH |
| 官方 Makefile CUDA 路径假设不匹配 | 无法编译 | 脚本中显式传入 CUDA 路径 |
| CUDA 12.0 BF16 host/device 限制 | 多个 `.cu/.h` 文件编译失败 | 增加 host 端 BF16 bit 转换函数 |
| 原文硬件与当前硬件不同 | 性能数值不可直接对齐 | 在报告中说明硬件差异并做趋势分析 |
| 外部 baseline 缺失 | 无法完整复现主对比实验 | 当前先完成官方内置 benchmark，后续补充 |
| 端到端模型实验资源要求高 | 需要模型权重、多 GPU、vLLM 集成 | 暂作为后续工作 |

### 6.2 可复现性评价

代码完整性：官方仓库提供了 kernel benchmark 相关代码，能够支持核心 CUDA kernel 的编译和运行。但在当前 CUDA 12.0 环境中需要额外兼容性修改。

实验描述清晰度：论文对整体方法、kernel 思路、benchmark 指标描述较清楚，但完整复现 Figure 11、Figure 16、Figure 17 需要更多 baseline、模型权重和具体运行配置。

隐性技巧：性能结果对 GPU 架构、CUDA 版本、矩阵形状、N、SplitK 和编译参数较敏感。若只根据论文文字复现，很难保证与原文数值完全一致。

总体评价：ZipServ 的 kernel-level 部分具备较好的可复现基础；系统级端到端部分复现门槛较高，需要更多硬件资源、模型资源和工程集成细节。

## 7. 最终交付材料对应

| 交付要求 | 仓库位置 | 状态 |
|---|---|---|
| 复现阅读笔记 | `paper_notes/` | 已整理初版 |
| 实验梳理清单 | `paper_notes/experiment_checklist.md` | 已整理初版 |
| 论文核心实验逻辑 | `paper_notes/core_experiment_logic.md` | 已整理初版 |
| 完整工程代码 | `code/ZipServ_ASPLOS26_patched/` | 已提交 |
| 一键部署脚本 | `scripts/setup_zipserv.sh` | 已提交 |
| 一键运行脚本 | `scripts/run_minimal_test.sh`, `scripts/run_n_sweep.sh`, `scripts/run_shape_sweep.sh` | 已提交 |
| 环境配置文件 | `env/environment.yml`, `env/requirements.txt` | 已提交 |
| 环境说明文档 | `env/environment.md` | 已提交 |
| 数据集处理代码与说明 | 暂不涉及真实数据集，kernel benchmark 使用合成矩阵 | 已在报告说明 |
| 全部实验日志 | `logs/build/`, `logs/experiments/` | 已保存 |
| 实验结果汇总表 | `results/raw/`, `results/processed/` | 已生成 |
| 复刻图表 | `figures/reproduced/` | 已生成 PNG/PDF |
| 单篇完整复现报告 | `report/reproduction_report.md` | 已完成 |
| 个人实践总结 | `report/personal_summary.md` | 已完成 |
| PPT 汇报材料 | `presentation/ZipServ_reproduction_presentation.pptx` | 已完成 |

## 8. 后续工作建议

如果继续推进完整论文复现，建议按以下优先级进行：

1. 补充 DietGPU、nvCOMP、DFloat11 baseline，完善 Figure 11 主对比。
2. 使用 Nsight Compute 对 ZipGEMM 做更细粒度 profiling，补充 Figure 12。
3. 下载可访问模型权重，统计 BF16 exponent 分布，复刻 Figure 2。
4. 尝试 vLLM 集成与端到端 latency/throughput 测试，补充 Figure 16、Figure 17。
5. 在不同 GPU 上重复 shape sweep，分析硬件差异对 ZipServ 收益的影响。
