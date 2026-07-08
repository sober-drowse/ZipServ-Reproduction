# ZipServ 论文复现阅读笔记

## 1. 论文基本信息

- 论文标题：ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression
- 会议：ASPLOS 2026
- 研究方向：LLM inference、lossless model compression、GPU kernel、Tensor Core GEMM
- 官方代码：https://github.com/HPMLL/ZipServ_ASPLOS26
- 当前复现仓库代码位置：`code/ZipServ_ASPLOS26_patched/`
- 当前服务器：NVIDIA L20 x 2，CUDA Toolkit 12.0

## 2. 论文要解决的问题

大模型推理的主要瓶颈之一是权重读写带来的显存容量和带宽压力。传统量化、剪枝等有损压缩方法可以减少存储和计算开销，但会带来精度损失风险；而无损压缩可以保证 bit-exact，但已有方法通常只适合存储或离线传输，推理时需要先解压到全精度权重，再调用 GEMM，导致额外内存访问和解压开销。

ZipServ 的核心问题是：如何让无损压缩不仅节省存储，还能在 GPU 推理时带来实际加速。

## 3. 方法整体架构

ZipServ 由两个阶段组成。

### 3.1 Offline Compressor

离线阶段对 BF16 权重做 TCA-TBE 压缩。论文观察到 LLM BF16 权重的 exponent 分布高度集中，而且 top-7 高频 exponent 通常是连续区间。因此每个权重矩阵可用一个 base exponent 加 3-bit offset 表示大多数权重的 exponent。

TCA-TBE 对每个 `8 x 8` tile 存储：

- 三个 64-bit bitmap：编码高频 exponent 的 3-bit code。
- high-frequency buffer：存储高频元素的 sign 和 mantissa。
- fallback buffer：存储低频/outlier 元素的完整 BF16。
- base exponent：用于恢复高频 exponent。

### 3.2 Online Inference Engine

在线阶段根据推理阶段采用不同策略：

- Prefill 阶段：计算密集，使用 decoupled pipeline，先解压再调用高吞吐 GEMM。
- Decode 阶段：更容易 memory-bound，使用 fused decompression-GEMM，也就是 ZipGEMM。

ZipGEMM 的关键是 “load-compressed, compute-decompressed”：从显存读取压缩权重，在寄存器中即时解压，然后直接送入 Tensor Core MMA，不把完整解压权重落到 global memory。

## 4. 核心创新点

### 4.1 利用 BF16 exponent 分布结构

论文不是泛泛地使用 Huffman/ANS 这类变长熵编码，而是利用 LLM BF16 权重 exponent 分布的结构性特点：高频 exponent 集中且连续。这样可以用固定长度 bitmap 编码，避免 GPU 上变长码解码的分支和串行依赖。

### 4.2 TCA-TBE 固定长度压缩格式

TCA-TBE 使用 triple bitmap 表示 3-bit exponent code，能够让每个线程独立恢复自己负责的元素。相比 Huffman/ANS，TCA-TBE 更适合 SIMT 并行执行。

### 4.3 ZipGEMM 融合解压与矩阵乘法

ZipGEMM 把解压逻辑融合到 GEMM kernel 内部，直接在寄存器里恢复 BF16 权重，避免传统 decoupled pipeline 中“压缩权重读一次、解压权重写一次、GEMM 再读一次”的冗余数据移动。

### 4.4 Stage-aware 推理策略

论文没有强行所有阶段都用 ZipGEMM，而是区分 prefill 和 decode。prefill 用 decoupled 解压加 cuBLAS，decode 用 fused ZipGEMM。这体现了对推理阶段计算/访存特性的区分。

## 5. 实验大类

### 5.1 动机实验

对应论文 Figure 1、Figure 3、Figure 4、Figure 5。主要证明：

- 传统无损压缩的解压时间可能超过 GEMM 本身。
- 变长熵编码与 GPU SIMT 执行不匹配。
- decoupled pipeline 会导致额外 global memory traffic。
- fused decompression-GEMM 在 roofline 视角下能提高 compute intensity。

### 5.2 Kernel-level 主对比实验

对应 Figure 11。比较 ZipGEMM 与 cuBLAS_TC、DietGPU、nvCOMP、DFloat11 等 baseline。评价指标包括 latency、TFLOPs、speedup。

当前已完成最小闭环：

```bash
./test_mm 4096 4096 128 1
```

该实验验证了官方 benchmark 在 NVIDIA L20 上可以编译并运行，但不代表论文主结论。

### 5.3 Micro-level 效率分析

对应 Figure 12。使用 Nsight Compute 分析 ZipGEMM 的 ALU 指令、DRAM 读流量、Tensor Core 利用率、shared memory bank conflict 等。

### 5.4 Standalone decompression kernel 实验

对应 Figure 13。比较 ZipServ-Decomp 与 DietGPU、nvCOMP、DFloat11 的纯解压性能。

当前代码中已经编译出：

```text
kernel_benchmark/test_decompress
```

后续需要确认参数格式并跑 decompression benchmark。

### 5.5 不同 GPU 世代和层级实验

对应 Figure 14 和 Figure 18。论文比较 RTX4090、RTX5090、A100、H800 等硬件上的表现。当前服务器是 NVIDIA L20，因此只能做硬件替代复现，并在报告中说明硬件不一致带来的差异。

### 5.6 参数敏感性实验

对应 Figure 15。研究不同 `N = batch_size x seq_len` 设置下的性能变化，用于说明 decode 小 N 场景中 ZipGEMM 更有优势，而 prefill 大 N 场景中 decoupled pipeline 更合适。

### 5.7 End-to-end inference 实验

对应 Figure 16、Figure 17。比较 ZipServ、vLLM、Transformers、DFloat11 的端到端 latency、throughput、memory breakdown。涉及 LLaMA3.1-8B、Mistral-24B、LLaMA3.1-70B 等模型。

这部分需要更多模型权重、vLLM 集成、多 GPU 资源和更复杂的脚本。当前阶段先完成 kernel benchmark 复现。

### 5.8 Limitation 和讨论实验

论文讨论了训练型 GPU/HBM GPU 上收益可能不如消费级或推理优化 GPU 明显，原因是 HBM 带宽更高、内存瓶颈被缓解。当前 L20 结果需要结合这一点分析。

## 6. 当前复现状态

已完成：

- 官方代码检索并下载。
- 在服务器上配置 conda 环境 `zipserv`。
- 修复 GitHub HTTPS 克隆失败，改用 SSH。
- 修复 CUDA 路径不匹配。
- 修复 CUDA 12.0 下 BF16 host/device 编译兼容问题。
- 编译 `libL_API.so`、`test_mm`、`test_decompress`。
- 跑通最小 `test_mm` 实验。
- 已将完整 patched 源码、一键脚本、环境文件、日志和 patch 上传 GitHub。

待完成：

- 批量运行 Fig. 15 相关不同 N 实验。
- 整理 Fig. 11 中代表性模型 layer shape。
- 编写日志解析脚本，生成 `results/summary.csv`。
- 复刻图表。
- 根据资源情况决定是否做端到端 vLLM 复现。

