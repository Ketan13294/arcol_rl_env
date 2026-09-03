#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sudo sysctl -w kernel.numa_balancing=0

python -c 'import warp as wp; wp.init(); wp.set_device("cuda:0")'

python scripts/train.py Unitree-G1-Flat-WBC --env.scene.num-envs=4096 
#--agent.resume=True --agent.load_run=2026-08-31_17-59-52 --agent.load_checkpoint=model_2000.pt