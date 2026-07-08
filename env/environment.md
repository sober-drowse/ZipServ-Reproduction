# 环境说明文档

## 服务器信息

- 用户：zzk
- 项目路径：`~/ZipServ-Reproduction`
- GPU：NVIDIA L20 x 2
- Driver：595.71.05
- CUDA Toolkit：12.0
- nvcc 路径：`/usr/bin/nvcc`

## Python / Conda 环境

- Conda 环境名：`zipserv`
- Python 版本：3.10
- 环境配置：
  - `env/environment.yml`
  - `env/requirements.txt`

## 编译环境

- gcc/g++：13.3.0
- make：4.3
- CUDA 编译参数：
  - `CUDA_INSTALL_PATH=/usr`
  - `CUDA_HOME=/usr`

## 注意事项

官方代码默认查找 `/usr/local/cuda/bin/nvcc`，但本服务器实际 `nvcc` 位于 `/usr/bin/nvcc`。因此本复现工程的一键编译脚本显式传入：

```bash
make CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr
```

此外，官方代码在 CUDA 12.0 环境下存在 BF16 host/device 编译兼容问题。本仓库 `code/ZipServ_ASPLOS26_patched/` 中已做兼容性修改，详细说明见：

```text
docs/pitfalls.md
patches/zipserv_cuda12_bf16_host_fix.patch
```
