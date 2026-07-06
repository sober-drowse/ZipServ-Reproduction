# 复现进度记录

## 已完成

### 1. 服务器环境初步确认

- GPU: NVIDIA L20 x 2
- Driver: 595.71.05
- CUDA Toolkit: 12.0
- Conda 环境: zipserv
- Python: 3.10

### 2. 官方代码获取与完整工程代码整理

官方仓库：

```text
https://github.com/HPMLL/ZipServ_ASPLOS26
```

服务器原始副本：

```text
~/ZipServ-Reproduction/third_party/ZipServ_ASPLOS26
```

复现仓库中的完整工程代码：

```text
code/ZipServ_ASPLOS26_patched/
```

该目录包含已修复 CUDA 12.0 兼容性问题的 ZipServ 源码，可用于一键编译和最小 benchmark 运行。

### 3. 编译问题修复

已解决：

- GitHub HTTPS 克隆失败，改用 SSH
- CUDA 默认路径 `/usr/local/cuda` 不匹配，编译时显式指定 `CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr`
- CUDA BF16 host/device 编译错误，添加 host-side bit conversion helper

修复记录见：

```text
docs/pitfalls.md
patches/zipserv_cuda12_bf16_host_fix.patch
```

### 4. 一键脚本

已新增：

```text
scripts/setup_zipserv.sh
scripts/run_minimal_test.sh
```

其中：

- `setup_zipserv.sh`：编译 ZipServ core library 和 kernel benchmark
- `run_minimal_test.sh`：运行最小 `test_mm` benchmark 并保存日志

### 5. 环境配置与说明

已新增：

```text
env/environment.yml
env/requirements.txt
env/environment.md
```

### 6. 最小 benchmark 已跑通

命令：

```bash
./test_mm 4096 4096 128 1
```

日志：

```text
logs/experiments/test_mm_4096_4096_128_1.log
```

结果摘要：

```text
results/processed/minimal_test_mm_4096_4096_128_1.md
```

## 下一步

1. 完善论文研读笔记：`paper_notes/reading_notes.md`
2. 完善实验清单：`paper_notes/experiment_checklist.md`
3. 梳理论文核心实验逻辑：`paper_notes/core_experiment_logic.md`
4. 批量运行不同 N 的 `test_mm`，复现参数敏感性实验。
5. 整理论文 Fig. 11 / Fig. 15 对应实验 shape。
6. 编写日志解析脚本，生成结果 CSV。
7. 绘制复刻图表。
