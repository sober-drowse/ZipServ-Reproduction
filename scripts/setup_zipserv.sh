#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$ROOT_DIR/code/ZipServ_ASPLOS26_patched"
LOG_DIR="$ROOT_DIR/logs/build"

mkdir -p "$LOG_DIR"

echo "[Info] Project root: $ROOT_DIR"
echo "[Info] Code dir: $CODE_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[Error] nvcc not found. Please install CUDA Toolkit or load CUDA environment."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[Warning] nvidia-smi not found. GPU status cannot be checked."
else
  nvidia-smi | tee "$LOG_DIR/nvidia-smi_before_build.log"
fi

cd "$CODE_DIR"

echo "[Info] Building ZipServ core library..."
source Init.sh
cd build
make clean || true
make CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr 2>&1 | tee "$LOG_DIR/setup_zipserv_core_build.log"

echo "[Info] Building kernel benchmark..."
cd ../kernel_benchmark
source test_env
make clean || true
make CUDA_INSTALL_PATH=/usr CUDA_HOME=/usr 2>&1 | tee "$LOG_DIR/setup_zipserv_kernel_benchmark_build.log"

echo "[Info] Build finished."
ls -lh test_mm test_decompress
