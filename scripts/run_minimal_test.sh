#!/usr/bin/env bash
set -e

GPU_ID="${1:-0}"
M="${2:-4096}"
K="${3:-4096}"
N="${4:-128}"
SPLITK="${5:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/code/ZipServ_ASPLOS26_patched/kernel_benchmark"
LOG_DIR="$ROOT_DIR/logs/experiments"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/test_mm_${M}_${K}_${N}_${SPLITK}.log"

echo "[Info] GPU_ID=$GPU_ID"
echo "[Info] Shape: M=$M K=$K N=$N SplitK=$SPLITK"
echo "[Info] Log file: $LOG_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_before_test_mm_${M}_${K}_${N}_${SPLITK}.log"
fi

cd "$BENCH_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

./test_mm "$M" "$K" "$N" "$SPLITK" 2>&1 | tee "$LOG_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_after_test_mm_${M}_${K}_${N}_${SPLITK}.log"
fi
