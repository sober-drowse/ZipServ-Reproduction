#!/usr/bin/env bash
set -e

cd ~/ZipServ-Reproduction
mkdir -p docs env results/processed patches scripts logs/build logs/experiments paper_notes

cat > docs/pitfalls.md <<'EOF'
# 踩坑记录与解决方案

## 1. GitHub HTTPS 克隆失败

### 现象

使用 HTTPS 克隆官方仓库失败：

```text
fatal: unable to access 'https://github.com/HPMLL/ZipServ_ASPLOS26.git/':
Failed to connect to github.com port 443
```

### 原因

服务器访问 GitHub HTTPS 不稳定或受网络环境限制。

### 解决方案

改用 GitHub SSH 方式：

```bash
ssh -T git@github.com
git clone git@github.com:HPMLL/ZipServ_ASPLOS26.git
```

并在 GitHub 账号中添加服务器用户 `zzk` 的 SSH 公钥。

---

## 2. CUDA 路径不匹配

### 现象

官方 Makefile 默认查找：

```text
/usr/local/cuda/bin/nvcc
```

但服务器实际 `nvcc` 路径为：

```text
/usr/bin/nvcc
```

导致报错：

```text
make: /usr/local/cuda/bin/nvcc: No such file or directory
```

### 原因

服务器 CUDA Toolkit 未安装在官方代码默认假设的 `/usr/local/cuda` 路径下。

### 解决方案

编译时显式指定 CUDA 路径：

```bash
make CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr
```

---

## 3. CUDA BF16 host/device 兼容问题

### 现象

编译时出现如下错误：

```text
calling a __device__ function("__bfloat16_as_ushort") from a __host__ function is not allowed
calling a __device__ function("__ushort_as_bfloat16") from a __host__ function is not allowed
```

涉及文件包括：

```text
csrc/L_API.cu
kernel_benchmark/utils.h
kernel_benchmark/test_mm.cu
kernel_benchmark/test_decompress.cu
```

### 原因

当前服务器 CUDA 12.0 环境下，部分 BF16 转换函数只能在 device 端调用，而官方代码在 host 函数中调用了这些函数。

### 解决方案

添加 host 端可用的 BF16 bit 转换函数，使用 `std::memcpy` 保留原始 bit 表示：

```cpp
static inline uint16_t host_bfloat16_as_ushort(__nv_bfloat16 val) {
    uint16_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    return bits;
}

static inline __nv_bfloat16 host_ushort_as_bfloat16(uint16_t bits) {
    __nv_bfloat16 val;
    std::memcpy(&val, &bits, sizeof(val));
    return val;
}
```

然后将 host 端调用替换为：

```cpp
host_bfloat16_as_ushort(...)
host_ushort_as_bfloat16(...)
```

### 影响范围

该修改只用于解决编译兼容性问题，不改变 ZipServ 的压缩格式、核心算法或 benchmark 逻辑。

---

## 4. benchmark 编译警告

### 现象

编译 `test_mm.cu` 时出现格式化输出警告：

```text
warning: format '%d' expects argument of type 'int', but argument has type 'uint64_t'
```

### 处理

该警告只影响调试打印格式，不影响 benchmark 编译和运行。暂未修改。
EOF

cat > docs/progress.md <<'EOF'
# 复现进度记录

## 已完成

### 1. 服务器环境初步确认

- GPU: NVIDIA L20 x 2
- Driver: 595.71.05
- CUDA Toolkit: 12.0
- Conda 环境: zipserv
- Python: 3.10

### 2. 官方代码获取

官方仓库：

```text
https://github.com/HPMLL/ZipServ_ASPLOS26
```

服务器路径：

```text
~/ZipServ-Reproduction/third_party/ZipServ_ASPLOS26
```

### 3. 编译问题修复

已解决：

- GitHub HTTPS 克隆失败
- CUDA 默认路径不匹配
- CUDA BF16 host/device 编译错误

修复记录见：

```text
docs/pitfalls.md
patches/zipserv_cuda12_bf16_host_fix.patch
```

### 4. 编译完成

已生成：

```text
third_party/ZipServ_ASPLOS26/build/libL_API.so
third_party/ZipServ_ASPLOS26/kernel_benchmark/test_mm
third_party/ZipServ_ASPLOS26/kernel_benchmark/test_decompress
```

### 5. 最小 benchmark 已跑通

命令：

```bash
./test_mm 4096 4096 128 1
```

日志：

```text
logs/experiments/test_mm_4096_4096_128_1.log
```

## 下一步

1. 批量运行不同 N 的 `test_mm`，复现参数敏感性实验。
2. 整理论文 Fig. 11 / Fig. 15 对应实验 shape。
3. 编写日志解析脚本，生成结果 CSV。
4. 绘制复刻图表。
EOF

cat > results/processed/minimal_test_mm_4096_4096_128_1.md <<'EOF'
# 最小 test_mm 实验结果

## 命令

```bash
./test_mm 4096 4096 128 1
```

## 实验形状

- M = 4096
- K = 4096
- N = 128
- SplitK = 1

## 结果

| 方法 | Time/ms | TFLOPs | TotalError |
|---|---:|---:|---:|
| BF16_triple_bitmap | 0.139 | 30.88 | 38.99 |
| CuBLAS_TC | 0.099 | 43.22 | 0.00 |
| CuBLAS_non-TC | 0.492 | 8.74 | 38.98 |

## 说明

该实验已成功完成，说明 ZipServ 官方 kernel benchmark 可以在当前服务器 NVIDIA L20 上完成编译和运行。

该结果仅作为最小闭环验证，不代表论文主实验结论。后续需要继续运行更多论文对应 shape 和不同 N 设置。
EOF

cat > docs/delivery_map.md <<'EOF'
# 最终交付材料对应位置

| 要求 | 当前存放位置 | 当前状态 |
|---|---|---|
| 复现阅读笔记 | `paper_notes/` | 待完善 |
| 实验梳理清单 | `paper_notes/experiment_checklist.md` | 已建初版/待补充 |
| 完整工程代码 | `third_party/ZipServ_ASPLOS26/` + `patches/` | 官方代码已下载，兼容 patch 已保存 |
| 部署说明 README | `README.md` | 已建/待完善 |
| 环境配置文件 | `env/` | 已建/待完善 |
| 环境说明文档 | `docs/progress.md`, `logs/build/` | 部分完成 |
| 数据集处理代码 | `scripts/` | 该阶段暂不涉及模型数据 |
| 实验日志 | `logs/experiments/` | 已保存最小实验日志 |
| 编译日志 | `logs/build/` | 已保存 |
| 权重存储目录 | 待定 | 后续端到端实验再建立 |
| 实验结果汇总表 | `results/processed/` | 已有最小实验摘要 |
| 复刻图表 | `figures/` | 待生成 |
| 完整复现报告 | `report/` | 待撰写 |
| 个人实践总结 | `report/` 或 `docs/` | 待撰写 |
| 踩坑记录 | `docs/pitfalls.md` | 已记录当前问题 |
EOF

cd ~/ZipServ-Reproduction/third_party/ZipServ_ASPLOS26
git diff > ../../patches/zipserv_cuda12_bf16_host_fix.patch
cd ~/ZipServ-Reproduction

if [ ! -f README.md ]; then
cat > README.md <<'EOF'
# ZipServ Reproduction

This repository contains reproduction materials for:

**ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression**

## Repository Structure

- `third_party/ZipServ_ASPLOS26`: official ZipServ implementation
- `env/`: environment records and setup scripts
- `paper_notes/`: paper reading notes and experiment checklist
- `scripts/`: reproduction scripts
- `logs/`: build and experiment logs
- `results/`: raw and processed experiment results
- `figures/`: reproduced and comparison figures
- `report/`: final reproduction report
- `docs/`: pitfalls and reproducibility review
EOF
fi

git remote set-url origin git@github.com:sober-drowse/ZipServ-Reproduction.git

git status
git add README.md docs env paper_notes scripts results logs patches
git commit -m "Record build fixes and minimal ZipServ benchmark"
git push -u origin main
