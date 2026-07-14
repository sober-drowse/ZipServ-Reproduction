# ZipServ 论文复现汇报演讲稿

适用对象：计算机软件工程本科生课程/助研汇报  
建议时长：12-18 分钟  
使用方式：PPT 每页先讲“这一页想说明什么”，再按讲稿补充解释。遇到老师追问时，可以重点回答“硬件差异、baseline 缺失、端到端实验未完整复现”的限制。

## 开场白

各位老师好，我汇报的论文是 ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression。这篇论文关注的是大语言模型推理阶段的显存和带宽瓶颈。我的复现重点是官方开源代码中的 CUDA kernel benchmark，包括代码编译、兼容性修复、最小实验、N sweep、shape sweep、日志保存、图表复刻和误差分析。

需要提前说明的是，本次复现使用的是学校服务器上的 NVIDIA L20，而原论文实验使用了 RTX4090、L40S、RTX5090、A100、H800 等多类 GPU。因此我的结果是 kernel-level 简化复现，不能直接等同于原论文全部端到端结论。

## 第 1 页：标题与汇报目标

这一页主要说明我这次汇报有三个目标。

第一，讲清楚原论文为什么要做 ZipServ。它不是普通的模型压缩论文，而是想把无损压缩真正用到 GPU 推理加速里。

第二，讲清楚论文的方法和成果。论文提出了 TCA-TBE 压缩格式和 ZipGEMM fused decompression-GEMM kernel。

第三，讲清楚我的复现工作，包括服务器环境、代码修复、运行了哪些实验、结果和原论文有什么差异。

## 第 2 页：问题背景

这一页先解释论文的背景。

LLM 是 Large Language Model，也就是大语言模型，比如 GPT、LLaMA、Qwen 这类模型。模型训练完成以后，我们把它用于回答问题、生成文本，这个过程叫 inference，也就是推理。

推理时最核心的计算之一是 GEMM，也就是 General Matrix Multiplication，通俗说就是大矩阵乘法。Transformer 里面很多线性层都可以看成矩阵乘法。

问题在于，大模型的权重矩阵非常大。GPU 不仅要算得快，还要不断从显存里读取权重。如果显存带宽跟不上，计算单元可能就在等数据。这就是论文说的 memory bandwidth bottleneck。

ZipServ 的目标是：通过无损压缩减少权重读取量，同时不要因为解压而拖慢速度。

## 第 3 页：四个基础概念

这一页是给本科生理解论文用的基础概念。

Token 可以理解为模型处理文本的基本单位。比如一句话会被切成很多 token。实验中的 N 维度可以近似理解为一次矩阵乘法中 token 或 batch 的规模。

BF16 是一种 16 位浮点格式。它包含 sign、exponent 和 mantissa。sign 表示正负，exponent 表示数量级，mantissa 表示有效数字。ZipServ 主要利用的是 BF16 权重里 exponent 分布比较集中的特点。

Tensor Core 是 NVIDIA GPU 里专门做矩阵乘法的硬件单元。普通 CUDA core 可以做很多计算，但 Tensor Core 对矩阵乘法特别快。ZipServ 的 ZipGEMM 目标就是让解压后的权重直接进入 Tensor Core 计算路径。

SIMT 是 GPU 执行模型，意思是多个线程执行同一条指令。如果压缩格式是变长的，每个线程解码进度不同，就会产生分支和等待；如果格式是固定长度 bitmap，GPU 并行起来就更友好。

## 第 4 页：prefill 和 decode

大语言模型推理通常分成两个阶段：prefill 和 decode。

Prefill 阶段处理用户输入的 prompt。这个阶段可以并行处理很多 token，所以矩阵乘法规模较大，计算密度高，更偏 compute-bound，也就是算力瓶颈。

Decode 阶段是一边生成一边把新 token 接到上下文里，通常逐 token 生成。这个阶段每一步矩阵规模较小，但权重还是要反复读取，所以更容易 memory-bound，也就是显存带宽瓶颈。

ZipServ 很重要的一点是 stage-aware。它不是强行所有阶段都用同一种方式，而是 prefill 用 decoupled decompression + GEMM，decode 用 fused ZipGEMM。

## 第 5 页：原论文动机

这一页解释为什么传统无损压缩不能直接拿来加速 LLM 推理。

传统无损压缩常用 Huffman 或 ANS 这类变长编码。它们适合存储，但在 GPU 上解码不友好。因为每个符号长度不同，线程很难完全独立推进，warp 内部会产生分支和串行依赖。

另一个问题是系统 pipeline。很多方案会先把压缩权重完整解压到 global memory，再调用 GEMM。这样虽然权重存储变小了，但推理时多了一次完整权重的写回和读取，反而增加显存流量。

论文的 Figure 1 提到，decoupled decompression step alone takes 1.56-3.44x the time of the core inference computation。也就是说，单独解压的开销可能比核心计算还大。

所以论文的判断是：问题不是无损压缩本身没用，而是传统压缩算法和 GPU 推理架构不匹配。

## 第 6 页：ZipServ 的核心方案

ZipServ 的核心答案可以概括为 TCA-TBE + ZipGEMM。

TCA-TBE 是 Tensor-Core-Aware Triple Bitmap Encoding。它是一种固定长度、bitmap-based 的权重压缩格式。它的目的不是只提高压缩率，而是让 GPU 能并行、常数时间地解码。

ZipGEMM 是 fused decompression-GEMM kernel。它不把权重先完整解压到 global memory，而是在读取压缩权重后，在 shared memory 或 register 附近即时恢复 BF16 值，然后直接送入 Tensor Core。

论文里有一句非常关键的话：load-compressed, compute-decompressed。我的理解是：从显存读的是压缩表示，但参与计算的是即时恢复后的原始 BF16 数值。

## 第 7 页：TCA-TBE 的统计基础

这一页解释为什么 ZipServ 可以压缩 BF16 权重。

BF16 的 exponent 有 8 位，理论上可以表示很多数量级。但论文观察到，LLM 权重的 exponent 分布不是均匀的，而是集中在少数值附近。论文报告 exponent field 的信息熵大约是 2.57 到 2.74 bits，远低于 8-bit 分配。

因此，论文估计 BF16 权重存在大约 1.51x 的理论无损压缩空间。

TCA-TBE 的设计思路是：对于一个 8 x 8 tile，用 base exponent 加 3-bit offset 表示大部分高频 exponent。3-bit code 又被拆成三张 64-bit bitmap，也就是 triple bitmap。这样做的好处是内存对齐、解码规则固定、线程路径一致。

## 第 8 页：ZipGEMM 为什么能减少显存流量

这一页对比传统 decoupled pipeline 和 ZipGEMM fused pipeline。

传统做法是：读压缩权重，解压成完整权重，写回 global memory，然后 GEMM 再读一次完整权重。这就多了一次很大的显存写入和读取。

ZipGEMM 的做法是：读压缩权重，在 shared memory 或 register 路径中恢复 BF16，然后直接交给 Tensor Core 做矩阵乘法。它避免了把完整解压权重再次写回 global memory。

所以 ZipGEMM 的优势不是“解压本身一定非常快”，而是把解压隐藏在矩阵乘法的数据路径里，同时减少显存流量。

## 第 9 页：原论文主要成果

这一页是原论文的 paper result，不是我的复现结果。

论文摘要和实验部分报告，ZipServ 可以把模型大小最多减少 30%，ZipGEMM kernel-level speedup 最高达到 2.21x over NVIDIA cuBLAS，最高达到 5.53x over DFloat11。端到端推理方面，ZipServ 平均比 vLLM 快 1.22x。

论文还提到，在 LLaMA3.1-8B、Mistral-24B、LLaMA3.1-70B 上，权重 footprint 大约降到原来的 71%-72%。省下来的显存可以给 KV cache 使用，从而支持更大的 batch 或更长上下文。

但论文也讨论了限制：在 A100、H800 这类 HBM 带宽更强的训练型 GPU 上，ZipGEMM 不一定总能超过高度优化的 cuBLAS。这说明 ZipServ 更适合显存带宽更紧张的消费级或推理优化 GPU。

## 第 10 页：我的复现范围

这一页开始讲我的复现工作。

本次复现优先完成官方仓库中的 kernel benchmark，而不是一开始就做端到端 vLLM。原因是 kernel benchmark 不需要下载完整大模型权重，也不需要复杂系统集成，更适合先形成可验证闭环。

我完成了五类材料：

第一，完整工程代码放在 `code/ZipServ_ASPLOS26_patched/`。

第二，一键编译脚本 `scripts/setup_zipserv.sh`。

第三，最小 `test_mm` 实验日志。

第四，`test_decompress` 解压实验日志。

第五，N sweep 和 shape sweep 的批量实验、CSV 汇总、Markdown 摘要和 PNG/PDF 图表。

这一页底部的“汇报口径”是为了说明官方代码的支持范围。可以补充说：官方仓库支持 ZipServ kernel benchmark 和 cuBLAS 对比，但 DietGPU、nvCOMP、DFloat11、vLLM 等外部 baseline 不是官方仓库里已经一键集成好的完整脚本，需要额外获取、编译和适配。所以本次复现优先完成官方代码支持的 kernel-level 实验。

## 第 11 页：服务器环境与兼容性修改

这一页说明我复现过程中遇到的工程问题。

服务器 GPU 是 NVIDIA L20 x2，CUDA Toolkit 是 12.0，nvcc 位于 `/usr/bin/nvcc`。Python 环境使用 conda 创建，环境名是 zipserv。

第一个问题是官方 Makefile 默认找 `/usr/local/cuda/bin/nvcc`，但服务器实际是 `/usr/bin/nvcc`。我在脚本里显式传入 `CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr`。

第二个问题是 CUDA 12.0 下 BF16 intrinsic 的 host/device 调用不兼容。官方代码在 host 函数里调用了部分 device 函数，会编译失败。我的解决办法是增加 host 端可用的 BF16 bit 转换函数，用 `std::memcpy` 保留 bit 表示。

## 第 12 页：最小实验验证

这一页说明代码已经跑通。

我先运行了 `test_mm 4096 4096 128 1`。结果中 BF16_triple_bitmap 时间是 0.139 ms，吞吐是 30.88 TFLOPs。这个实验验证了官方 benchmark 可以在 L20 上编译运行。

然后运行了 `test_decompress 128 128 128`。解压结果和原始矩阵完全一致，16384 个元素 100% identical，absolute error 为 0。这说明 standalone decompression 在这个最小设置下满足 lossless 正确性。

这里要强调，这两个实验是最小闭环验证，不代表论文主结论完全复现。

## 第 13 页：N sweep 结果

这一页展示参数敏感性实验，也就是对应论文 Figure 15 的简化复现。

我固定 M=4096，K=4096，SplitK=1，扫描 N 从 1 到 512。N 可以理解为 token/batch 维度。左图是 latency，右图是相对 cuBLAS_TC 的 speedup。

在我的 L20 结果中，ZipServ 相对 cuBLAS_TC 的 speedup 都小于 1，也就是没有超过 cuBLAS_TC。但它相对 cuBLAS_non-TC 明显更快。

这个结果说明，在当前硬件和 shape 设置下，cuBLAS_TC 仍然非常强；ZipServ 的收益与硬件、矩阵形状和 N 都有关系。

## 第 14 页：Shape sweep 结果

这一页对应论文 Figure 11 的简化复现。

我固定 N=32，SplitK=1，选择了几个代表性 shape，包括 square_4096、LLaMA FFN GateUp 类 shape、Mistral-like GateUp shape 和 down projection shape。

结果显示，ZipServ 在两个 FFN GateUp 类 shape 上超过了 cuBLAS_TC。其中 11008x4096 的 speedup 大约是 1.40，28672x4096 的 speedup 大约是 1.28。但在 square 和 down projection 类 shape 上没有超过 cuBLAS_TC。

这说明 ZipServ 的收益不是对所有矩阵形状都一样，而是和 shape、N、硬件带宽以及 kernel 参数有关。

## 第 15 页：为什么复现结果和论文不完全一致

这一页解释误差来源。

原论文的实验条件包括 RTX4090、L40S、RTX5090、A100、H800 等 GPU，并且包含 DietGPU、nvCOMP、DFloat11、vLLM、Transformers 等 baseline，还覆盖更多模型和 layer shape。

我的复现条件是 NVIDIA L20，主要使用官方 benchmark 内置的 cuBLAS_TC 和 cuBLAS_non-TC 对比，重点是 kernel-level 实验。

所以本次结果应该表述为：我完成了可追溯的 kernel-level 简化复现，并在部分 shape 上观察到了与论文思想一致的趋势，而不是完整复刻论文所有数值。

## 第 16 页：可复现性评价

这一页是复盘。

我认为 ZipServ 的 kernel 部分可复现性较好，因为官方仓库提供了 benchmark 代码。但系统级实验门槛较高，因为要复现端到端结果，需要模型权重、vLLM 集成、多 GPU、外部 baseline 和更长时间的实验。

隐藏成本主要包括：CUDA 版本差异、BF16 编译兼容问题、GPU 架构差异、SplitK 参数、矩阵 shape 选择，以及 baseline 的安装和适配。

我的处理方式是保留完整日志、结果 CSV、图表、patch 和踩坑记录，保证复现过程可追溯。

## 第 17 页：后续工作

如果继续做，我建议优先补三类实验。

第一是 baseline，补 DietGPU、nvCOMP、DFloat11，这样 Figure 11 的主对比会更完整。

第二是 profiling，使用 Nsight Compute 分析显存流量、Tensor Core 利用率、ALU 指令开销，这对应论文 Figure 12。

第三是 end-to-end，下载模型权重，尝试 vLLM 集成，测试吞吐和延迟，对应 Figure 16 和 Figure 17。

我的建议是先补 baseline，再做 profiling，最后做端到端，因为端到端最重、依赖最多。

这一页底部的补充说明是为了避免把后续工作说得过满。可以这样讲：在当前 L20 服务器条件下，baseline 理论上可以继续补，但需要额外适配；profiling 取决于服务器是否安装 `ncu` 以及是否开放 GPU profiling 权限；end-to-end 需要模型权重、vLLM 集成和较长 GPU 独占时间，所以我把它作为后续工作，而不是当前已经完成的内容。

## 第 18 页：总结

最后总结一下。

从论文角度看，ZipServ 的价值在于把无损压缩从“只省存储”推进到“可以加速推理”。它的核心是硬件感知压缩格式 TCA-TBE 和 fused decompression-GEMM kernel。

从复现角度看，我完成了官方 kernel benchmark 的代码编译、CUDA 兼容性修复、最小实验、解压实验、N sweep、shape sweep、日志保存、结果汇总和图表复刻。

本次复现的局限是没有完整覆盖原论文所有 baseline 和端到端系统实验。但从课程复现任务角度，已经形成了一个从论文阅读到实验运行、结果分析、报告和 PPT 的完整闭环。

我的汇报到这里，谢谢老师。

## 可能被问到的问题

### Q1：为什么你的 N sweep 里 ZipServ 没有超过 cuBLAS_TC？

可以回答：cuBLAS_TC 是 NVIDIA 高度优化的 Tensor Core GEMM，本身非常强。本次硬件是 L20，不是原论文主要测试的 RTX4090/L40S；同时我只扫描了一个 4096x4096 shape。ZipServ 的优势更依赖 memory-bound 场景和特定 shape，所以这个结果说明硬件和 shape 对收益影响很大。

### Q2：既然无损压缩还要解压，为什么可能更快？

可以回答：关键不是解压免费，而是 ZipServ 减少了显存读取量，并把解压放在 register/Tensor Core 数据路径附近，避免把完整解压权重写回 global memory。如果显存带宽是瓶颈，少读数据带来的收益可能超过解压开销。

### Q3：你的复现和论文主实验差在哪里？

可以回答：论文主实验包括更多 GPU、更多模型、更多 baseline 和端到端 vLLM 推理。我当前完成的是官方 kernel benchmark 的简化复现，重点验证代码可运行、解压正确、N 和 shape 对性能的影响。

### Q4：这篇论文对软件工程学生有什么启发？

可以回答：它说明高性能 AI 系统不是只写算法，还要理解硬件、内存层次、编译环境和系统 pipeline。论文复现也不只是跑代码，还要记录环境、日志、误差来源和可复现性限制。
