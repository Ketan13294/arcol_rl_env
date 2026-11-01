#!/usr/bin/env bash
# Evaluates each WBC-ARM checkpoint as training writes it. Usage: ./start_eval.sh [run_dir]
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate arcol_rl

# Pinned off the training cores (6-14) so eval doesn't slow PPO.
CPU_CORES="4"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

taskset -c "$CPU_CORES" python scripts/eval_wbc_arm.py ${1:+--run-dir "$1"}
