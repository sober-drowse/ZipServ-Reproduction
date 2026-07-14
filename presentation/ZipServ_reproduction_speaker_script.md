# ZipServ 论文复现汇报讲稿

适用对象：计算机软件工程本科生课程/助研汇报  
建议时长：12-18 分钟  
使用方式：每页先讲“这一页要说明什么”，再按下面讲稿展开。讲的时候不用逐字背，可以把每页最后一两句话作为重点记住。

## 开场白

各位老师好，我汇报的论文是 ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression。这篇论文研究的是大语言模型推理阶段的显存和带宽瓶颈。

我这次复现的重点不是完整跑通论文所有端到端系统实验，而是先完成官方开源代码中最核心、最容易落地的 CUDA kernel benchmark。具体包括：服务器环境搭建、CUDA 编译问题修复、`test_mm`、`test_decompress`、N sweep、shape sweep、日志保存、CSV 汇总、图表生成和结果分析。

需要提前说明的是，我使用的是学校服务器上的 NVIDIA L20，而原论文实验还涉及 RTX4090、L40S、RTX5090、A100、H800 等 GPU。所以我的结果属于 kernel-level 简化复现，不能直接等同于原论文全部端到端结论。

## 第 1 页：标题与汇报目标

这一页主要说明我这次汇报有三个目标。

第一，讲清楚原论文想解决什么问题。ZipServ 不是普通的模型压缩论文，它想证明无损压缩不只是节省模型存储，还可以参与 GPU 推理加速。

第二，讲清楚论文怎么做。它提出了 TCA-TBE 压缩格式和 ZipGEMM，也就是把解压和矩阵乘法融合起来的 CUDA kernel。

第三，讲清楚我复现了什么。我主要复现官方 kernel benchmark，分析了不同 N 和不同矩阵形状下 ZipServ 与 cuBLAS 的性能差异。

## 第 2 页：为什么 LLM 推理会被显存和带宽卡住

LLM 是 Large Language Model，也就是大语言模型。训练好的模型用来回答问题、生成文本，这个过程叫 inference，也就是推理。

推理里最核心的计算之一是 GEMM，General Matrix Multiplication，也就是通用矩阵乘法。Transformer 里的 QKV projection、O projection、GateUp projection、Down projection 等线性层，本质上都可以写成：

```text
Y = W x X
```

这里 `W` 是模型权重矩阵，`X` 是输入激活，`Y` 是输出。

大模型的问题是 `W` 非常大。GPU 不仅要算得快，还要不断从显存里读这些权重。如果显存读得慢，Tensor Core 这些计算单元就会等数据。这就是论文说的 memory bandwidth bottleneck。

ZipServ 的目标可以用一句话概括：

> 用无损压缩减少权重读取量，同时避免解压开销拖慢推理。

## 第 3 页：四个基础概念

这一页是为了让本科生更容易理解论文。

Token 可以理解为模型处理文本的基本单位。模型不是按完整句子一次性计算，而是把文本切成 token。实验里的 `N` 可以近似理解为一次矩阵乘法里处理的 token/batch 规模。

BF16 是一种 16 位浮点格式，由三部分组成：

```text
sign: 1 bit，表示正负
exponent: 8 bits，表示数量级
mantissa: 7 bits，表示有效数字
```

ZipServ 主要利用的是 BF16 权重中 exponent，也就是指数位的冗余。论文发现，大模型权重的 exponent 往往集中在少数几个相近值附近，所以不用每个数都完整存 8 位 exponent。

Tensor Core 是 NVIDIA GPU 上专门做矩阵乘法的硬件单元。cuBLAS_TC 和 ZipServ 都会用到 Tensor Core。

SIMT 是 GPU 的执行方式，可以理解成一个 warp 里的很多线程最好做同样的事。如果压缩格式是变长编码，每个线程解码位置不同，就容易分支和等待；如果是固定 bitmap，线程路径更一致，GPU 更喜欢。

## 第 4 页：prefill 和 decode 是什么

LLM 推理可以分成两个阶段：prefill 和 decode。

Prefill 可以理解成“读题阶段”。用户输入一段 prompt，模型先把输入的 token 一起处理，建立上下文。这个阶段一次处理的 token 多，矩阵乘法规模大，计算比较充分，更容易是 compute-bound，也就是算力瓶颈。

Decode 可以理解成“答题阶段”。模型开始一个 token 一个 token 地生成输出，比如先生成“ZipServ”，再生成“是”，再生成“一种”。每生成一个 token，都要读取模型权重并做计算。这个阶段每次新 token 少，但权重要反复读，所以更容易是 memory-bound，也就是显存带宽瓶颈。

ZipServ 更关注 decode 阶段，因为 decode 更容易被显存读权重卡住。压缩权重如果能少读显存，就更可能带来收益。

## 第 5 页：原论文动机

传统无损压缩方法，比如 Huffman 或 ANS，适合存储，但不一定适合 GPU 推理。

第一个问题是变长编码。每个符号长度不一样，GPU 线程要找自己的数据位置会变复杂，容易产生跨字节读取、分支和串行依赖。

第二个问题是传统解压流程通常是 decoupled pipeline。它会先把压缩权重解压成完整 BF16 权重，写回 global memory，也就是 GPU 显存，然后 GEMM 再从显存读一次完整权重。

这个流程的问题是：虽然存储时省了空间，但推理时多了一次完整权重的写回和读取，显存流量反而增加。

论文 Figure 1 提到，单独解压步骤可能达到核心计算时间的 1.56 到 3.44 倍。也就是说，不改计算流程，直接接传统无损压缩，可能越压越慢。

## 第 6 页：ZipServ 的核心方案

ZipServ 的核心方案是 TCA-TBE 加 ZipGEMM。

TCA-TBE 是一种 GPU 友好的压缩格式。它重点压缩 BF16 权重的 exponent 位。直观理解是：如果一个 tile 里大多数 exponent 都集中在几个连续值上，就保存一个 base exponent，然后每个元素只存 3-bit offset。

例如 exponent 大多在：

```text
112, 113, 114, 115, 116, 117, 118
```

就可以保存一个 base，再用 3 bit 表示偏移，而不是每个元素都完整存 8 bit exponent。

ZipGEMM 是把解压和 GEMM 融合在一个 kernel 里。它不把完整解压权重写回显存，而是在 shared memory / register 附近恢复 BF16，然后直接送进 Tensor Core 计算。

论文里最关键的一句话是：

```text
load-compressed, compute-decompressed
```

也就是：从显存读的是压缩权重，参与计算的是即时恢复后的 BF16 权重。

## 第 7 页：为什么要 triple bitmap

TCA-TBE 不是简单地把每个 3-bit offset 连续挨着存，而是把它拆成三张 bitmap，也就是 triple bitmap。

原因是 GPU 不喜欢不对齐的 3-bit 连续编码。假如直接把很多 3-bit code 拼成一串，某个元素的 code 可能跨 byte 或跨 word 边界。线程要取自己的 code 时，需要复杂的偏移、移位、拼接，效率不高。

ZipServ 的做法是：一个 8 x 8 tile 有 64 个元素，每个元素 offset 有 3 bit。它把所有元素的第 1 位放进第一张 64-bit bitmap，所有元素的第 2 位放进第二张 bitmap，所有元素的第 3 位放进第三张 bitmap。

这样第 `i` 个元素只需要在三张 bitmap 中取第 `i` 位，就能恢复自己的 3-bit offset。

好处是：

- 每张 bitmap 正好 64 bit，内存对齐好。
- 每个线程位置好找。
- 所有线程执行相同流程，适合 SIMT。
- 每个元素可以独立解码，不依赖前一个元素。

所以 triple bitmap 的目的不是让格式看起来简单，而是让 GPU 解码更规则、更并行。

## 第 8 页：ZipGEMM 和传统解压流程的区别

这里要解释 shared memory、register 和 Tensor Core 在哪里。

可以把 GPU 存储距离想成这样：

```text
global memory: GPU 显存，容量大，但离计算单元远，访问慢
shared memory: GPU SM 内部的小型高速共享存储
register: 每个线程自己的寄存器，容量最小，但最快
Tensor Core: 专门做矩阵乘法的计算单元
```

传统流程是：

```text
压缩权重在 global memory
解压成完整 BF16 权重
写回 global memory
GEMM 再读完整 BF16 权重
送入 Tensor Core
```

ZipServ 希望变成：

```text
压缩权重在 global memory
在 shared memory / register 中恢复 BF16
直接送入 Tensor Core
```

区别就是：ZipServ 尽量避免“解压后的完整权重再次落回 global memory”。这样可以减少显存读写。

## 第 9 页：原论文主要成果

这一页讲的是原论文结果，不是我的复现结果。

论文报告 ZipServ 最多可以减少 30% 模型大小，ZipGEMM kernel-level speedup 最高达到 2.21x over cuBLAS，最高达到 5.53x over DFloat11。端到端推理方面，平均比 vLLM 快 1.22x。

论文还说，在 LLaMA3.1-8B、Mistral-24B、LLaMA3.1-70B 上，权重 footprint 大约降到原来的 71%-72%。省下来的显存可以给 KV cache 使用，从而支持更大 batch 或更长上下文。

但论文也承认，在 A100、H800 这类 HBM 带宽更强的训练型 GPU 上，ZipGEMM 不一定总能超过 cuBLAS。这说明 ZipServ 的收益和硬件环境有关。

## 第 10 页：我的复现范围

我的复现重点是官方仓库里的 kernel benchmark。

这里先解释 `test_mm` 是什么。`mm` 可以理解成 matrix multiplication，也就是矩阵乘法测试。它测试的是：

```text
Y = W x X
```

其中：

```text
W: M x K，权重矩阵
X: K x N，输入激活
Y: M x N，输出
```

`test_mm` 会在相同的 M、K、N、SplitK 下比较三条路径：

- `BF16_triple_bitmap`：ZipServ 路径，读取压缩权重，在 kernel 内即时解压，再用 Tensor Core 做 GEMM。
- `cuBLAS_TC`：NVIDIA 官方 cuBLAS Tensor Core 路径，不压缩权重，直接读取完整 BF16 权重。
- `cuBLAS_non-TC`：不使用或较少使用 Tensor Core 的 cuBLAS 路径，也不压缩权重。

所以 `test_mm` 和推理加速的关系是：它不是完整聊天模型推理，但它测试的是 LLM 推理中最核心的线性层矩阵乘法。如果这个核心算子更快，完整推理才有可能更快。

这一页底部的“汇报口径”说明官方代码支持范围：官方仓库支持 ZipServ kernel benchmark 和 cuBLAS 对比；但 DietGPU、nvCOMP、DFloat11、vLLM 等外部 baseline 不是一键集成好的，需要额外获取、编译和适配。所以我优先完成官方代码支持的 kernel-level 实验。

## 第 11 页：服务器环境与兼容性修改

服务器 GPU 是 NVIDIA L20 x2，CUDA Toolkit 是 12.0，`nvcc` 在 `/usr/bin/nvcc`，Python 环境是 conda 的 `zipserv`。

第一个问题是官方 Makefile 默认找：

```text
/usr/local/cuda/bin/nvcc
```

但服务器实际是：

```text
/usr/bin/nvcc
```

所以我在编译脚本里显式传入：

```bash
CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr
```

第二个问题是 CUDA 12.0 下 BF16 intrinsic 的 host/device 调用不兼容。官方代码里一些 CPU host 函数调用了只能在 GPU device 端使用的 BF16 bit 转换函数，会导致编译失败。

我的解决方式是增加 host 端可用的 BF16 bit 转换函数，用 `std::memcpy` 原样复制 16 个 bit。这个修改只解决编译兼容性，不改变 ZipServ 算法逻辑。

## 第 12 页：最小实验验证

我先运行了：

```bash
test_mm 4096 4096 128 1
```

这个实验测试的是：

```text
W: 4096 x 4096
X: 4096 x 128
Y: 4096 x 128
```

结果中 ZipServ 的 `BF16_triple_bitmap` 时间是 0.139 ms，性能是 30.88 TFLOPs。它说明官方 benchmark 已经可以在 L20 上编译运行并输出性能指标。

然后我运行了：

```bash
test_decompress 128 128 128
```

这个实验单独验证解压是否正确。结果显示原始矩阵和解压矩阵 16384 个元素 100% identical，absolute error 为 0。

这说明：至少在这个最小实验中，ZipServ 的解压过程是 lossless 的。

## 第 13 页：N sweep 是什么

N sweep 就是固定 M 和 K，只改变 N。

我固定：

```text
M = 4096
K = 4096
SplitK = 1
```

改变：

```text
N = 1, 8, 16, 32, 64, 128, 256, 512
```

N 的含义可以理解为一次 GEMM 中处理的 token/batch 规模。N 小的时候更接近 decode 场景，一次处理 token 少，更容易受显存读权重影响；N 大的时候更接近 prefill 场景，矩阵乘法规模更大，计算更充分。

所以 N sweep 的目的就是：

> 看 token/batch 规模变化时，ZipServ 和 cuBLAS 的性能关系怎么变。

我的结果里，ZipServ 在这个 4096 x 4096 shape 上没有超过 cuBLAS_TC，但明显快于 cuBLAS_non-TC。

原因是 cuBLAS_TC 是 NVIDIA 高度优化的 Tensor Core GEMM，非常强。ZipServ 虽然少读了压缩权重，但它还要做 bitmap 解码、exponent 恢复、BF16 拼接等额外工作。只有当省下的显存读取时间大于这些解压开销时，ZipServ 才会超过 cuBLAS_TC。

## 第 14 页：shape sweep 是什么

Shape sweep 是固定 N，只改变 M 和 K，也就是改变权重矩阵形状。

我固定：

```text
N = 32
SplitK = 1
```

改变不同 shape：

```text
4096 x 4096
11008 x 4096
28672 x 4096
4096 x 14336
```

这里 M 和 K 决定权重矩阵 `W` 的形状：

```text
W: M x K
```

M 可以理解为输出维度，K 可以理解为输入 hidden dimension。不同 LLM layer，比如 GateUp projection、Down projection，矩阵形状不同，memory-bound 和 compute-bound 程度也不同。

Shape sweep 的目的就是：

> 看不同模型层矩阵形状下，ZipServ 是否更容易发挥压缩带来的优势。

我的结果里，ZipServ 在两个 FFN GateUp 类 shape 上超过 cuBLAS_TC，speedup 大约是 1.40 和 1.28。但在 square 和 down projection 类 shape 上没有超过。

这说明 ZipServ 不是所有情况下都必然更快，它的收益和 shape、N、GPU 架构、kernel 参数有关。

## 第 15 页：为什么复现结果和论文不完全一致

原论文实验使用 RTX4090、L40S、RTX5090、A100、H800 等 GPU，还包含 DietGPU、nvCOMP、DFloat11、vLLM、Transformers 等 baseline，并覆盖更多模型和 layer shape。

我的复现条件是 NVIDIA L20，主要使用官方 benchmark 内置的 cuBLAS_TC 和 cuBLAS_non-TC 对比，重点是 kernel-level 实验。

所以这次结果应该表述为：

> 我完成了可追溯的 kernel-level 简化复现，并在部分 shape 上观察到与论文思想一致的趋势，而不是完整复刻论文所有数值。

## 第 16 页：可复现性评价

我认为 ZipServ 的 kernel 部分可复现性较好，因为官方仓库提供了 benchmark 代码。但系统级实验门槛较高，因为端到端实验需要模型权重、vLLM 集成、多 GPU、外部 baseline 和更长时间的实验。

隐藏成本主要包括：

- CUDA 版本差异。
- BF16 编译兼容问题。
- GPU 架构差异。
- SplitK 和 tile 参数。
- 不同矩阵 shape。
- 外部 baseline 的安装和适配。

我的处理方式是保留完整日志、CSV、图表、patch 和踩坑记录，让复现过程可追溯。

## 第 17 页：后续工作

如果继续做，我建议补三类实验。

第一是 baseline，补 DietGPU、nvCOMP、DFloat11。这里的外部 baseline 适配不是简单 clone 代码，而是要找代码、配环境、编译、统一输入 shape、统一计时方式，再接入同一套 CSV 和图表。难度中高。

第二是 profiling，使用 Nsight Compute 分析显存流量、Tensor Core 利用率、ALU 指令开销。这取决于服务器是否安装 `ncu`，以及普通用户是否有 GPU profiling 权限。

第三是 end-to-end，下载模型权重，尝试 vLLM 集成，测试吞吐和延迟。这个最重，需要模型权重、vLLM 环境、ZipServ 集成和较长 GPU 独占时间。

所以我把这些作为后续工作，而不是当前已经完成的内容。

## 第 18 页：总结

从论文角度看，ZipServ 的价值在于把无损压缩从“只省存储”推进到“可以加速推理”。它的核心是硬件感知压缩格式 TCA-TBE 和 fused decompression-GEMM kernel。

从复现角度看，我完成了官方 kernel benchmark 的编译、CUDA 兼容性修复、最小实验、解压实验、N sweep、shape sweep、日志保存、结果汇总和图表复刻。

本次复现的局限是没有完整覆盖论文所有 baseline 和端到端系统实验。但从课程复现任务角度，已经形成了一个从论文阅读、服务器环境搭建、代码修复、实验运行、结果分析到报告和 PPT 的完整闭环。

我的汇报到这里，谢谢老师。

## 可能被问到的问题

### Q1：为什么 N sweep 里 ZipServ 没有超过 cuBLAS_TC？

cuBLAS_TC 是 NVIDIA 高度优化的 Tensor Core GEMM，本身非常强。ZipServ 虽然减少了权重读取量，但也增加了 bitmap 解码和 BF16 恢复开销。只有当节省的显存读取时间大于额外解压开销时，ZipServ 才会超过 cuBLAS_TC。在我的 L20 和 4096 x 4096 shape 设置下，这个收益没有完全体现出来。

### Q2：既然 ZipServ 也要解压，为什么可能更快？

关键不是解压免费，而是减少了 global memory 读写。传统方法把完整解压权重写回显存，GEMM 再读一次。ZipServ 把解压放在 shared memory / register 附近，并直接送入 Tensor Core，避免完整解压权重再次落回显存。如果显存带宽是瓶颈，少读数据的收益可能超过解压开销。

### Q3：cuBLAS_TC、cuBLAS_non-TC 和 ZipServ 有什么区别？

cuBLAS_TC 是不压缩权重、直接用 NVIDIA Tensor Core 的官方强 baseline。cuBLAS_non-TC 是不压缩权重、不走或较少走 Tensor Core 的普通 GEMM baseline。ZipServ 是读取压缩权重，在 GEMM kernel 内即时解压，再送入 Tensor Core 计算。

### Q4：N sweep 和 shape sweep 区别是什么？

N sweep 是固定 M 和 K，只改变 N，看 token/batch 规模变化对性能的影响。Shape sweep 是固定 N，只改变 M 和 K，看不同 LLM layer 权重矩阵形状对性能的影响。

### Q5：外部 baseline 为什么没做完？

论文里的 DietGPU、nvCOMP、DFloat11 不是官方 ZipServ 仓库中一键集成好的实验。要补它们，需要额外获取代码、编译依赖、统一输入 shape、统一计时方式，并把结果接入同一套图表。考虑到当前时间和服务器条件，我先完成官方代码支持的 kernel-level 实验，把外部 baseline 作为后续增强方向。
