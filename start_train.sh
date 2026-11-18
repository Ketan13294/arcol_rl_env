#!/usr/bin/env bash
# Tuned for a single RTX 4090 (24 GB).
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate arcol_rl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# rsl_rl reads WANDB_USERNAME as the entity; key comes from .wandb.env via train.py.
export WANDB_USERNAME=kmehta4-virginia-tech
# Sim and PPO run on the GPU; the CPU only launches kernels, so a few pinned cores suffice.
# i9-14900KF: P-cores are CPUs 0-15 (HT pairs), E-cores 16-31. One thread per
# physical P-core 3-7 (6/7 boost to 6.4 GHz); skip core 0, which takes most IRQs.
CPU_CORES="6,8,10,12,14"
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=$OMP_NUM_THREADS
# Pin to the 4090 so a second GPU or display device is never picked up.
export CUDA_VISIBLE_DEVICES=0

# Set numa_balancing if permitted without prompting for sudo password
if [ "$(cat /proc/sys/kernel/numa_balancing 2>/dev/null)" != "0" ]; then
  sudo -n sysctl -w kernel.numa_balancing=0 2>/dev/null || true
fi

# Power of two divides evenly into PPO mini-batches (8192*24/4 = 49152 per batch).
# Flat G1 (njmax=300, nconmax=32) fits well within 24 GB; drop to 6144 if OOM.
NUM_ENVS=8192

# taskset -c "$CPU_CORES" python scripts/train.py Unitree-G1-Flat-WBC \
#   --env.scene.num-envs=$NUM_ENVS \
#   --agent.run_name=CorrectionsDone

# taskset -c "$CPU_CORES" python scripts/train.py Unitree-G1-Flat-WBC-ARM \
#   --env.scene.num-envs=$NUM_ENVS \
#   --agent.run_name=ArmPosKernelCont \
#   --agent.load_run=2026-10-26_05-23-26_ArmPosKernel \
#   --agent.load_checkpoint=model_131000.pt \
#   --agent.max-iterations=100000 \
#   --agent.resume=True

# SuperMimic (world model + BPTT) on the split actor. 4096 envs as in the paper; one
# iteration = 24 env steps + 10 model/policy updates. Warm start is a copy of the split WarmStart.
# taskset -c "$CPU_CORES" python scripts/train.py Unitree-G1-Flat-WBC-ARM-SuperMimic \
#   --env.scene.num-envs=4096 \
#   --agent.run_name=SuperMimic \
#   --agent.load_run=2026-11-01_17-44-10_WarmStart \
#   --agent.load_checkpoint=model_225000.pt \
#   --agent.max-iterations=20000 \
#   --agent.save-interval=500 \
#   --agent.resume=True

# Warm start made by scripts/convert_split_ckpt.py from g1_wbc ArmPosKernelCont/model_225000.pt.
# taskset -c "$CPU_CORES" python scripts/train.py Unitree-G1-Flat-WBC-ARM-Split \
#   --env.scene.num-envs=$NUM_ENVS \
#   --agent.run_name=SplitWarmStart \
#   --agent.load_run=2026-11-01_17-44-10_WarmStart \
#   --agent.load_checkpoint=model_225000.pt \
#   --agent.max-iterations=50000 \
#   --agent.save-interval=500 \
#   --agent.resume=True
# Wrist pitch/yaw action scale 0.075 -> 0.439; checkpoint carried over by scripts/rescale_action_ckpt.py.
taskset -c "$CPU_CORES" python scripts/train.py Unitree-G1-Flat-WBC-ARM-Split \
  --env.scene.num-envs=$NUM_ENVS \
  --agent.run_name=HeightPoseStd \
  --agent.load_run=2026-11-05_05-09-59_WristScaleCont \
  --agent.load_checkpoint=model_296500.pt \
  --agent.max-iterations=25000 \
  --agent.save-interval=500 \
  --agent.resume=True
