#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "experiments" / "shape_sweep"
RAW_DIR = ROOT / "results" / "raw"
PROCESSED_DIR = ROOT / "results" / "processed"
FIG_DIR = ROOT / "figures" / "reproduced"


def extract_float(pattern: str, text: str):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    match = re.search(r"test_mm_(.+)_(\d+)_(\d+)_(\d+)_(\d+)\.log", path.name)
    if not match:
        return None

    label, m, k, n, splitk = match.groups()
    row = {
        "label": label,
        "M": int(m),
        "K": int(k),
        "N": int(n),
        "SplitK": int(splitk),
        "log_file": str(path.relative_to(ROOT)),
        "compression_ratio": extract_float(r"Compression ratio:\s+([0-9.]+)x", text),
        "zipserv_time_ms": extract_float(r"BF16_triple_bitmap\s*->\s*Time/ms:\s*([0-9.]+)", text),
        "zipserv_tflops": extract_float(
            r"BF16_triple_bitmap\s*->\s*Time/ms:\s*[0-9.]+\s*Performance/TFLOPs:\s*([0-9.]+)",
            text,
        ),
        "cublas_tc_time_ms": extract_float(r"CuBLAS_TC\s*->\s*Time/ms:\s*([0-9.]+)", text),
        "cublas_tc_tflops": extract_float(
            r"CuBLAS_TC\s*->\s*Time/ms:\s*[0-9.]+\s*Performance/TFLOPs:\s*([0-9.]+)",
            text,
        ),
        "cublas_nontc_time_ms": extract_float(r"CuBLAS_non-TC\s*->\s*Time/ms:\s*([0-9.]+)", text),
        "cublas_nontc_tflops": extract_float(
            r"CuBLAS_non-TC\s*->\s*Time/ms:\s*[0-9.]+\s*Performance/TFLOPs:\s*([0-9.]+)",
            text,
        ),
    }

    if row["zipserv_time_ms"] and row["cublas_tc_time_ms"]:
        row["speedup_vs_cublas_tc"] = row["cublas_tc_time_ms"] / row["zipserv_time_ms"]
    else:
        row["speedup_vs_cublas_tc"] = None

    if row["zipserv_time_ms"] and row["cublas_nontc_time_ms"]:
        row["speedup_vs_cublas_nontc"] = row["cublas_nontc_time_ms"] / row["zipserv_time_ms"]
    else:
        row["speedup_vs_cublas_nontc"] = None

    return row


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "M",
        "K",
        "N",
        "SplitK",
        "compression_ratio",
        "zipserv_time_ms",
        "cublas_tc_time_ms",
        "cublas_nontc_time_ms",
        "zipserv_tflops",
        "cublas_tc_tflops",
        "cublas_nontc_tflops",
        "speedup_vs_cublas_tc",
        "speedup_vs_cublas_nontc",
        "log_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_speedup(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["label"].replace("_", "\n") for row in rows]
    speedups = [row["speedup_vs_cublas_tc"] for row in rows]

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(labels, speedups)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("Speedup vs cuBLAS TC")
    plt.title("Shape sweep speedup")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_shape_sweep_speedup.png", dpi=300)
    plt.savefig(FIG_DIR / "fig11_shape_sweep_speedup.pdf")
    plt.close()


def plot_latency(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [row["label"].replace("_", "\n") for row in rows]
    x = range(len(rows))
    width = 0.25

    plt.figure(figsize=(9, 4.8))
    plt.bar([i - width for i in x], [row["zipserv_time_ms"] for row in rows], width=width, label="BF16 triple bitmap")
    plt.bar(list(x), [row["cublas_tc_time_ms"] for row in rows], width=width, label="cuBLAS TC")
    plt.bar([i + width for i in x], [row["cublas_nontc_time_ms"] for row in rows], width=width, label="cuBLAS non-TC")
    plt.xticks(list(x), labels)
    plt.ylabel("Time (ms)")
    plt.title("Shape sweep latency")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_shape_sweep_latency.png", dpi=300)
    plt.savefig(FIG_DIR / "fig11_shape_sweep_latency.pdf")
    plt.close()


def write_markdown(rows):
    path = PROCESSED_DIR / "shape_sweep_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    n_values = sorted({row["N"] for row in rows})
    splitk_values = sorted({row["SplitK"] for row in rows})

    lines = [
        "# Shape sweep 实验结果汇总",
        "",
        "## 实验设置",
        "",
        f"- N = {', '.join(str(value) for value in n_values)}",
        f"- SplitK = {', '.join(str(value) for value in splitk_values)}",
        "- 对比对象：BF16_triple_bitmap、cuBLAS_TC、cuBLAS_non-TC",
        "",
        "## 结果表",
        "",
        "| Shape | M | K | ZipServ Time/ms | cuBLAS TC Time/ms | Speedup vs cuBLAS TC | Compression Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['label']} | {row['M']} | {row['K']} | {row['zipserv_time_ms']:.4f} | "
            f"{row['cublas_tc_time_ms']:.4f} | {row['speedup_vs_cublas_tc']:.4f} | "
            f"{row['compression_ratio']:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "该实验是论文 Figure 11 的简化复现，使用当前 NVIDIA L20 服务器对若干代表性 LLM linear layer shape 进行 kernel-level benchmark。由于硬件和 baseline 覆盖与原论文不同，该图主要用于展示复现流程和趋势分析。",
            "",
            "生成图表：",
            "",
            "- `figures/reproduced/fig11_shape_sweep_latency.png`",
            "- `figures/reproduced/fig11_shape_sweep_latency.pdf`",
            "- `figures/reproduced/fig11_shape_sweep_speedup.png`",
            "- `figures/reproduced/fig11_shape_sweep_speedup.pdf`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = []
    for path in sorted(LOG_DIR.glob("test_mm_*.log")):
        row = parse_log(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No shape sweep logs found in {LOG_DIR}")

    order = {
        "square_4096": 0,
        "llama_ffn_gateup_11008x4096": 1,
        "mistral_like_gateup_28672x4096": 2,
        "down_proj_4096x14336": 3,
    }
    rows.sort(key=lambda row: order.get(row["label"], 99))

    write_csv(rows, RAW_DIR / "test_mm_shape_sweep.csv")
    write_csv(rows, PROCESSED_DIR / "test_mm_shape_sweep_summary.csv")
    write_markdown(rows)
    plot_latency(rows)
    plot_speedup(rows)

    print(f"[Info] Parsed {len(rows)} logs")
    print(f"[Info] Wrote {PROCESSED_DIR / 'test_mm_shape_sweep_summary.csv'}")
    print(f"[Info] Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
