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
