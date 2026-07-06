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
