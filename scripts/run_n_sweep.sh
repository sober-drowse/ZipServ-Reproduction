#!/usr/bin/env bash
set -e

GPU_ID="${1:-0}"
M="${2:-4096}"
K="${3:-4096}"
SPLITK="${4:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/code/ZipServ_ASPLOS26_patched/kernel_benchmark"
BUILD_DIR="$ROOT_DIR/code/ZipServ_ASPLOS26_patched/build"
LOG_DIR="$ROOT_DIR/logs/experiments/n_sweep"

mkdir -p "$LOG_DIR"

echo "[Info] GPU_ID=$GPU_ID"
echo "[Info] M=$M K=$K SplitK=$SPLITK"
echo "[Info] Logs: $LOG_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_before_n_sweep.log"
fi

cd "$BENCH_DIR"
source test_env
export LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for N in 1 8 16 32 64 128 256 512; do
  echo "========== Running N=$N =========="
  ./test_mm "$M" "$K" "$N" "$SPLITK" 2>&1 | tee "$LOG_DIR/test_mm_${M}_${K}_${N}_${SPLITK}.log"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_after_n_sweep.log"
fi

echo "[Info] N sweep completed."
