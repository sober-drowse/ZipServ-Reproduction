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
