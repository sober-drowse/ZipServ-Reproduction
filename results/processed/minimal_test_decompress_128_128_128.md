# 最小 test_decompress 实验结果

## 命令

```bash
./test_decompress 128 128 128
```

## 正确性验证

| 指标 | 结果 |
|---|---:|
| Total elements | 16384 |
| Identical | 16384 (100.00%) |
| Different | 0 (0.00%) |
| Max absolute difference | 0.000000 |
| Max relative difference | 0.000000 |
| Average absolute difference | 0.000000 |
| Total absolute error | 0.000000 |

结论：解压结果与原始矩阵完全一致，满足 lossless / bit-exact 要求。

## 性能结果

| 项目 | 时间 |
|---|---:|
| Decompression time | 0.0060 ms |
| CuBLAS non-TC | 0.0208 ms |
| CuBLAS TC | 0.0069 ms |
| Decompression + CuBLAS non-TC | 0.0268 ms |
| Decompression + CuBLAS TC | 0.0129 ms |

## 时间占比

| 对比对象 | 解压占比 |
|---|---:|
| vs CuBLAS non-TC | 22.5% |
| vs CuBLAS TC | 46.7% |

## 日志位置

```text
logs/experiments/test_decompress_128_128_128.log
```

## 说明

该实验验证了 standalone decompression 的正确性，并展示了 decoupled decompression 相对于 GEMM 的额外开销。该结果可作为论文动机实验和 Fig. 13 解压实验的最小复现记录。
