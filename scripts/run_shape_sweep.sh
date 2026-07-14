#!/usr/bin/env bash
set -e

GPU_ID="${1:-0}"
N="${2:-32}"
SPLITK="${3:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/code/ZipServ_ASPLOS26_patched/kernel_benchmark"
BUILD_DIR="$ROOT_DIR/code/ZipServ_ASPLOS26_patched/build"
LOG_DIR="$ROOT_DIR/logs/experiments/shape_sweep"

mkdir -p "$LOG_DIR"

echo "[Info] GPU_ID=$GPU_ID"
echo "[Info] N=$N SplitK=$SPLITK"
echo "[Info] Logs: $LOG_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_before_shape_sweep.log"
fi

cd "$BENCH_DIR"
source test_env
export LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# label M K
SHAPES=(
  "square_4096 4096 4096"
  "llama_ffn_gateup_11008x4096 11008 4096"
  "mistral_like_gateup_28672x4096 28672 4096"
  "down_proj_4096x14336 4096 14336"
)

for item in "${SHAPES[@]}"; do
  read -r LABEL M K <<< "$item"
  echo "========== Running $LABEL M=$M K=$K N=$N SplitK=$SPLITK =========="
  ./test_mm "$M" "$K" "$N" "$SPLITK" 2>&1 | tee "$LOG_DIR/test_mm_${LABEL}_${M}_${K}_${N}_${SPLITK}.log"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_after_shape_sweep.log"
fi

echo "[Info] Shape sweep completed."
