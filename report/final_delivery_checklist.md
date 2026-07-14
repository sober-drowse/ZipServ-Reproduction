# 最终交付材料清单

| 要求 | 对应文件/目录 | 状态 |
|---|---|---|
| 复现阅读笔记 | `paper_notes/reading_notes.md` | 已整理初版 |
| 实验梳理清单 | `paper_notes/experiment_checklist.md` | 已整理初版 |
| 论文核心实验逻辑 | `paper_notes/core_experiment_logic.md` | 已整理初版 |
| 完整工程代码 | `code/ZipServ_ASPLOS26_patched/` | 已提交，包含 CUDA 12.0 兼容性修改 |
| 一键部署说明 README | `README.md`, `scripts/setup_zipserv.sh` | 已完成 |
| 环境配置文件 | `env/environment.yml`, `env/requirements.txt` | 已完成 |
| 环境说明文档 | `env/environment.md` | 已完成 |
| 数据集处理代码与说明 | 本阶段 kernel benchmark 使用合成矩阵，不涉及真实数据集 | 已在报告中说明 |
| 全部实验日志 | `logs/build/`, `logs/experiments/` | 已保存 |
| 训练输出/benchmark 输出 | `logs/experiments/` | 已保存 benchmark 输出 |
| 权重存储目录 | 当前未运行端到端模型权重实验 | 已在报告中说明限制 |
| 实验结果汇总表 | `results/raw/`, `results/processed/` | 已生成 CSV 与 Markdown 摘要 |
| 原文指标 vs 复现指标 | `report/reproduction_report.md` 第 5 节 | 已以实验对应关系和误差分析形式整理 |
| 复刻图表 png/pdf | `figures/reproduced/` | 已生成 |
| 单篇完整复现报告 Markdown | `report/reproduction_report.md` | 已完成 |
| 单篇完整复现报告 Word | `report/reproduction_report.docx` | 已生成 |
| 个人实践总结 | `report/personal_summary.md` | 已完成 |
| PPT 汇报材料 | `presentation/ZipServ_reproduction_presentation.pptx` | 已完成 |

## 说明

本次复现已形成从服务器环境、代码编译、实验运行、日志保存、结果解析、图表生成到报告/PPT 的完整闭环。当前未完整覆盖原文所有端到端实验，主要原因是缺少原文同款 GPU、外部 baseline 适配、vLLM 集成和模型权重资源。报告中已单独列出这些限制和后续工作。
