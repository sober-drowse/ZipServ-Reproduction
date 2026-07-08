#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "experiments" / "n_sweep"
RAW_DIR = ROOT / "results" / "raw"
PROCESSED_DIR = ROOT / "results" / "processed"
FIG_DIR = ROOT / "figures" / "reproduced"


def extract_float(pattern: str, text: str):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_log(path: Path):
    text = path.read_text(errors="ignore")
    shape = re.search(r"test_mm_(\d+)_(\d+)_(\d+)_(\d+)\.log", path.name)
    if not shape:
        return None
    m, k, n, splitk = map(int, shape.groups())
    return {
        "log_file": str(path.relative_to(ROOT)),
        "M": m,
        "K": k,
        "N": n,
        "SplitK": splitk,
        "compression_ratio": extract_float(r"Compression ratio:\s+([0-9.]+)x", text),
        "zipserv_time_ms": extract_float(r"BF16_triple_bitmap\s*->\s*Time/ms:\s*([0-9.]+)", text),
        "zipserv_tflops": extract_float(r"BF16_triple_bitmap\s*->\s*Time/ms:\s*[0-9.]+\s*Performance/TFLOPs:\s*([0-9.]+)", text),
        "cublas_tc_time_ms": extract_float(r"CuBLAS_TC\s*->\s*Time/ms:\s*([0-9.]+)", text),
        "cublas_tc_tflops": extract_float(r"CuBLAS_TC\s*->\s*Time/ms:\s*[0-9.]+\s*Performance/TFLOPs:\s*([0-9.]+)", text),
        "cublas_nontc_time_ms": extract_float(r"CuBLAS_non-TC\s*->\s*Time/ms:\s*([0-9.]+)", text),
        "cublas_nontc_tflops": extract_float(r"CuBLAS_non-TC\s*->\s*Time/ms:\s*[0-9.]+\s*Performance/TFLOPs:\s*([0-9.]+)", text),
    }


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_latency(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_values = [row["N"] for row in rows]
    plt.figure(figsize=(7, 4.2))
    plt.plot(n_values, [row["zipserv_time_ms"] for row in rows], marker="o", label="BF16 triple bitmap")
    plt.plot(n_values, [row["cublas_tc_time_ms"] for row in rows], marker="s", label="cuBLAS TC")
    plt.plot(n_values, [row["cublas_nontc_time_ms"] for row in rows], marker="^", label="cuBLAS non-TC")
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("Time (ms)")
    plt.title("N sweep latency, M=4096 K=4096 SplitK=1")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_n_sweep_latency.png", dpi=300)
    plt.savefig(FIG_DIR / "fig15_n_sweep_latency.pdf")
    plt.close()


def plot_speedup(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [str(row["N"]) for row in rows]
    speedups = [row["speedup_vs_cublas_tc"] for row in rows]
    plt.figure(figsize=(7, 4.2))
    plt.bar(labels, speedups)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("N")
    plt.ylabel("Speedup vs cuBLAS TC")
    plt.title("N sweep speedup, M=4096 K=4096 SplitK=1")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig15_n_sweep_speedup.png", dpi=300)
    plt.savefig(FIG_DIR / "fig15_n_sweep_speedup.pdf")
    plt.close()


def write_markdown(rows):
    path = PROCESSED_DIR / "n_sweep_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# N sweep 实验结果汇总",
        "",
        "## 实验设置",
        "",
        "- M = 4096",
        "- K = 4096",
        "- SplitK = 1",
        "- N = 1, 8, 16, 32, 64, 128, 256, 512",
        "",
        "## 结果表",
        "",
        "| N | ZipServ Time/ms | cuBLAS TC Time/ms | cuBLAS non-TC Time/ms | Speedup vs cuBLAS TC | Compression Ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['N']} | {row['zipserv_time_ms']:.4f} | {row['cublas_tc_time_ms']:.4f} | "
            f"{row['cublas_nontc_time_ms']:.4f} | {row['speedup_vs_cublas_tc']:.4f} | "
            f"{row['compression_ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "该实验是论文 Figure 15 的简化复现。当前硬件为 NVIDIA L20，和论文中的 RTX4090/L40S 不一致，因此主要用于观察趋势和形成可追溯复现实验流程。",
            "",
            "生成图表：",
            "",
            "- `figures/reproduced/fig15_n_sweep_latency.png`",
            "- `figures/reproduced/fig15_n_sweep_latency.pdf`",
            "- `figures/reproduced/fig15_n_sweep_speedup.png`",
            "- `figures/reproduced/fig15_n_sweep_speedup.pdf`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = []
    for log_path in sorted(LOG_DIR.glob("test_mm_4096_4096_*_1.log")):
        row = parse_log(log_path)
        if row is None:
            continue
        if row["zipserv_time_ms"] and row["cublas_tc_time_ms"]:
            row["speedup_vs_cublas_tc"] = row["cublas_tc_time_ms"] / row["zipserv_time_ms"]
        else:
            row["speedup_vs_cublas_tc"] = None
        if row["zipserv_time_ms"] and row["cublas_nontc_time_ms"]:
            row["speedup_vs_cublas_nontc"] = row["cublas_nontc_time_ms"] / row["zipserv_time_ms"]
        else:
            row["speedup_vs_cublas_nontc"] = None
        rows.append(row)

    rows.sort(key=lambda item: item["N"])
    if not rows:
        raise SystemExit(f"No N sweep logs found in {LOG_DIR}")

    write_csv(rows, RAW_DIR / "test_mm_n_sweep.csv")
    write_csv(rows, PROCESSED_DIR / "test_mm_n_sweep_summary.csv")
    write_markdown(rows)
    plot_latency(rows)
    plot_speedup(rows)
    print(f"[Info] Parsed {len(rows)} logs")
    print(f"[Info] Wrote {PROCESSED_DIR / 'test_mm_n_sweep_summary.csv'}")
    print(f"[Info] Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
