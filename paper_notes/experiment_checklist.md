# Experiment Checklist

| Figure/Table | Experiment | Type | Required Code | Required Model/Data | Required Hardware | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| Fig. 1 | Motivation: lossless compression pipeline overhead | Motivation | TBD | GateUp layers | L40S-like GPU | High | TODO | Need inspect code support |
| Fig. 11 | ZipGEMM kernel performance | Main comparison | Official benchmark | LLM linear layer shapes | RTX4090/L40S/L20 | Highest | TODO | Start here |
| Fig. 13 | Standalone decompression kernel comparison | Efficiency | Official + baselines | LLaMA/Mistral block | GPU | Medium | TODO | Baselines may be hard |
| Fig. 15 | Performance under different N | Sensitivity | Official benchmark | Synthetic shapes | GPU | High | TODO | Good first target |
| Fig. 16 | End-to-end inference | E2E | vLLM integration | LLaMA/Mistral | Multi-GPU | Medium | TODO | Check repo support |
| Fig. 17 | Latency/memory breakdown | Efficiency | Profiling scripts | LLaMA3.1-8B | GPU | Medium | TODO | Need profiling |
