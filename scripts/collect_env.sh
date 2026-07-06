#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/build

date | tee logs/build/date.log
pwd | tee logs/build/project_path.log
uname -a | tee logs/build/uname.log
nvidia-smi | tee logs/build/nvidia-smi.log
nvcc --version | tee logs/build/nvcc.log || true
gcc --version | tee logs/build/gcc.log || true
g++ --version | tee logs/build/gpp.log || true
make --version | tee logs/build/make.log || true
git --version | tee logs/build/git.log || true
conda info | tee logs/build/conda_info.log || true
conda list | tee logs/build/conda_list.log || true
python --version | tee logs/build/python.log || true
pip list | tee logs/build/pip_list.log || true
