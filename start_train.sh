#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=5

# Set numa_balancing if permitted without prompting for sudo password
if [ "$(cat /proc/sys/kernel/numa_balancing 2>/dev/null)" != "0" ]; then
  sudo -n sysctl -w kernel.numa_balancing=0 2>/dev/null || true
fi

taskset -c 5-9 python scripts/train.py Unitree-G1-Flat-WBC \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --agent.load_run=2026-08-29_20-12-33 \
  --agent.load_checkpoint=model_12000.pt

# --agent.run_name=addedheightMovementPenalty_5 \
