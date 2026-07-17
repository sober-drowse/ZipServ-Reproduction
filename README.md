# ZipServ Reproduction

本仓库用于复现论文：

**ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression**

官方仓库：

```text
https://github.com/HPMLL/ZipServ_ASPLOS26
```

## 仓库结构

```text
.
├── code/ZipServ_ASPLOS26_patched/   # 已做 CUDA 12.0 兼容性修复的完整工程代码
├── patches/                         # 相对官方代码的修改 patch
├── scripts/                         # 一键编译与实验脚本
├── env/                             # 环境配置与说明
├── docs/                            # 踩坑记录、进度记录、交付材料映射
├── logs/                            # 编译和实验日志
├── results/                         # 实验结果摘要与汇总
├── figures/                         # 复刻图表
├── paper_notes/                     # 论文阅读笔记和实验清单
├── report/                          # 最终复现报告
└── presentation/                    # 最终汇报 PPT
```

## 当前复现状态

已完成：

- 官方代码获取
- CUDA 路径不匹配修复
- CUDA BF16 host/device 兼容性修复
- ZipServ core library 编译
- kernel benchmark 编译
- 最小 `test_mm` 实验运行
- `test_decompress` 正确性验证
- N sweep 与 shape sweep 简化复现实验
- 实验日志、结果表格、复刻图表、复现报告和汇报 PPT 整理

最小实验命令：

```bash
./test_mm 4096 4096 128 1
```

结果摘要见：

```text
results/processed/minimal_test_mm_4096_4096_128_1.md
```

## 环境配置

创建并激活 conda 环境：

```bash
conda create -n zipserv python=3.10 -y
conda activate zipserv
pip install -r env/requirements.txt
```

完整环境说明见：

```text
env/environment.md
```

## 一键编译

```bash
bash scripts/setup_zipserv.sh
```

该脚本会编译：

```text
code/ZipServ_ASPLOS26_patched/build/libL_API.so
code/ZipServ_ASPLOS26_patched/kernel_benchmark/test_mm
code/ZipServ_ASPLOS26_patched/kernel_benchmark/test_decompress
```

## 一键运行最小实验

默认使用 GPU 0，运行 `M=4096, K=4096, N=128, SplitK=1`：

```bash
bash scripts/run_minimal_test.sh
```

指定 GPU 和 shape：

```bash
bash scripts/run_minimal_test.sh 0 4096 4096 128 1
```

日志会保存到：

```text
logs/experiments/
```

## 兼容性修改说明

本仓库中的完整工程代码位于：

```text
code/ZipServ_ASPLOS26_patched/
```

相对官方代码做了 CUDA 12.0 兼容性修改，主要包括：

- 修复 `/usr/local/cuda` 路径假设导致的编译问题
- 修复 CUDA BF16 host/device 函数调用不兼容问题

详细记录见：

```text
docs/pitfalls.md
patches/zipserv_cuda12_bf16_host_fix.patch
```

该修改不改变 ZipServ 的压缩格式、核心算法或 benchmark 逻辑，仅用于保证当前服务器环境下可编译运行。

## 最终材料

```text
paper_notes/                                      # 阅读笔记、实验清单、核心实验逻辑
code/ZipServ_ASPLOS26_patched/                   # 完整工程代码
scripts/                                         # 一键部署、运行、解析脚本
env/                                             # 环境配置与说明
logs/                                            # 编译与实验日志
results/                                         # 原始结果与汇总表
figures/reproduced/                              # 复刻图表 png/pdf
report/reproduction_report.md                    # Markdown 版实验复现报告
report/实验复现报告.docx                         # Word 版实验复现报告
presentation/ZipServ_reproduction_final.pptx     # 最终汇报 PPT
```
