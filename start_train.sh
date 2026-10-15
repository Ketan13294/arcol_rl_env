#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=5

# Set numa_balancing if permitted without prompting for sudo password
if [ "$(cat /proc/sys/kernel/numa_balancing 2>/dev/null)" != "0" ]; then
  sudo -n sysctl -w kernel.numa_balancing=0 2>/dev/null || true
fi

# taskset -c 5-9 python scripts/train.py Unitree-G1-Flat-WBC \
#   --env.scene.num-envs=5096 \
#   --agent.run_name=CorrectionsDone \


taskset -c 5-9 python scripts/train.py Unitree-G1-Flat-WBC-ARM \
  --env.scene.num-envs=5096 \
  --agent.run_name=InitalArm \
  --agent.load_run=2026-10-11_22-32-30_InitalArm\
  --agent.load_checkpoint=model_13500.pt \
  --agent.resume=True \

