# ZipServ 论文核心实验逻辑

## 1. 总体实验目标

ZipServ 的实验要证明三件事：

1. LLM BF16 权重存在适合无损压缩的 exponent 冗余。
2. 固定长度、GPU 友好的 TCA-TBE 格式比传统变长熵编码更适合推理时解压。
3. fused decompression-GEMM 能把无损压缩从“只省存储”变成“同时省存储并加速推理”。

因此实验逻辑不是单一 benchmark，而是从动机、kernel、decompression、硬件适应性、端到端系统逐层展开。

## 2. 动机链条

### 2.1 权重可以无损压缩

论文先分析 BF16 权重的 exponent 分布，指出大多数权重 exponent 集中在少数连续值上。这个观察支撑 TCA-TBE 的设计：大部分 exponent 可以用 base exponent + 3-bit offset 恢复。

复现时需要关注：

- 是否能统计不同模型/矩阵的 exponent histogram。
- 高频 exponent 覆盖率是否接近论文中的约 97%。
- 压缩率是否接近论文中的约 1.4x 或模型体积减少约 30%。

### 2.2 传统无损解压不适合 GPU 推理

论文指出 Huffman/ANS 等变长编码在 GPU 上存在：

- 符号长度不定。
- bit pointer 依赖前面符号解码结果。
- warp 内线程分支和进度不一致。
- 解压与 GEMM 分离造成额外内存读写。

因此，单纯把传统无损压缩接入推理 pipeline，可能会让解压开销超过 GEMM 本身。

### 2.3 ZipServ 的核心假设

如果压缩格式足够规则，并且解压结果直接进入 Tensor Core 计算路径，那么无损压缩带来的显存带宽节省可以转化为实际速度收益。

## 3. Kernel benchmark 逻辑

Kernel benchmark 是当前最优先复现部分，因为它不需要完整模型权重，也不需要 vLLM 集成。

核心命令形式：

```bash
./test_mm M K N SplitK
```

其中：

- `M`：输出维度或权重矩阵行数。
- `K`：输入维度或权重矩阵列数。
- `N`：token/batch 维度，对 decode/prefill 特性很关键。
- `SplitK`：kernel 参数，用于部分形状的并行策略。

输出关注：

- `BF16_triple_bitmap` latency 和 TFLOPs。
- `CuBLAS_TC` latency 和 TFLOPs。
- `CuBLAS_non-TC` latency 和 TFLOPs。
- verification error。
- compression statistics。

最小实验已经跑通：

```bash
./test_mm 4096 4096 128 1
```

当前结果：

| Method | Time/ms | TFLOPs |
|---|---:|---:|
| BF16_triple_bitmap | 0.139 | 30.88 |
| CuBLAS_TC | 0.099 | 43.22 |
| CuBLAS_non-TC | 0.492 | 8.74 |

该结果用于验证工程闭环，不直接作为论文主结论。

## 4. 参数敏感性实验逻辑

Figure 15 的核心变量是 `N`。论文认为：

- 小 N 更接近 decode 阶段，memory-bound 更明显，ZipGEMM 更可能受益。
- 大 N 更接近 prefill 阶段，compute-bound 更明显，fused 解压开销可能抵消收益。

复现设计：

```text
固定 M, K, SplitK
变化 N = 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
记录 BF16_triple_bitmap 与 CuBLAS_TC latency
计算 speedup = CuBLAS_TC_time / ZipServ_time
```

预期分析方式：

- 如果 speedup > 1，说明 ZipServ 快于 cuBLAS_TC。
- 如果 speedup < 1，说明该形状或该 N 下 ZipServ 没有优势。
- 当前硬件是 L20，不能直接要求复现 RTX4090/L40S 论文曲线，只能做趋势对比和误差分析。

## 5. 主对比实验逻辑

Figure 11 的核心是不同模型 layer shape 上的 kernel-level speedup。

论文涉及模型族：

- LLaMA3.1：8B、70B、405B
- Qwen2.5：7B、14B、32B、72B
- Gemma3：12B、27B
- Mistral：24B、123B

涉及 layer 类型：

- QKV projection
- O projection
- GateUp projection
- Down projection
- LM head

当前复现可先选代表性 shape，而不是一开始覆盖所有模型：

1. LLaMA3.1-8B 类似 shape。
2. 较大的 FFN GateUp/Down shape。
3. 小层 O_proj shape。

比较对象先用官方 `test_mm` 内置的：

- BF16_triple_bitmap
- CuBLAS_TC
- CuBLAS_non-TC

DietGPU、nvCOMP、DFloat11 等 baseline 后续视时间和代码可用性决定是否加入。

## 6. Standalone decompression 实验逻辑

`test_decompress` 用于验证独立解压 kernel。该实验对应论文 Figure 13。

实验目标：

- 评估 TCA-TBE 格式即使不融合 GEMM，也是否能高效解压。
- 与传统 entropy codec baseline 对比。

当前状态：

- `test_decompress` 已编译成功。
- 尚未确认运行参数和输出格式。

下一步需要：

```bash
./test_decompress
```

查看 usage 或默认输出，再设计批量运行脚本。

## 7. End-to-end 实验逻辑

End-to-end 实验对应 Figure 16、Figure 17，目标是证明 kernel-level speedup 能转化为真实推理系统的 latency/throughput 改善。

论文配置：

- LLaMA3.1-8B on 1 x RTX4090
- Mistral-24B on 2 x L40S
- LLaMA3.1-70B on 4 x L40S
- batch size：8、32
- output length：128、256、512、1024、2048
- baseline：vLLM、Transformers、DFloat11

当前限制：

- 服务器 GPU 是 L20，不是论文中的 RTX4090/L40S。
- 需要模型权重和可能的 HuggingFace 权限。
- 需要确认官方仓库是否完整提供 vLLM 集成。

因此复现优先级低于 kernel benchmark。

## 8. 误差分析逻辑

复现实验结果与论文不一致时，优先从以下维度解释：

- GPU 不一致：NVIDIA L20 vs RTX4090/L40S。
- CUDA 版本不一致：当前 CUDA 12.0，论文使用 NVCC 12.4，RTX5090 使用 12.8。
- 编译器不一致：当前 gcc/g++ 13.3，论文使用 GCC 11.3。
- kernel 参数未调优：SplitK、tile 配置可能影响小 shape。
- baseline 缺失：当前最小复现只比较内置 cuBLAS，不含 DietGPU/nvCOMP/DFloat11。
- 输入 shape 覆盖不足：需要按论文 layer shape 扩展。

## 9. 当前下一步实验路线

1. 跑 `test_decompress`，确认参数和日志格式。
2. 批量跑不同 N 的 `test_mm`，生成参数敏感性数据。
3. 编写 `scripts/parse_test_mm_logs.py`，提取 Time/ms、TFLOPs、compression ratio。
4. 生成 `results/summary.csv`。
5. 绘制第一版复刻图：ZipServ vs cuBLAS_TC latency/speedup。

