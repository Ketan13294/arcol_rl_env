# Project Context: `arcol_rl_env`

## 1. Overview & Purpose

`arcol_rl_env` (derived from `unitree_rl_mjlab`) is a reinforcement learning (RL) robotics simulation and Sim-to-Real deployment pipeline designed for Unitree legged and humanoid robots. It utilizes **MuJoCo** with **MuJoCo Warp** for GPU-accelerated parallel rigid-body physics and integrates with **RSL-RL** (PPO) for policy optimization using modular **Isaac Lab / mjlab** abstractions.

### Supported Robots
- **Humanoids**: Unitree G1 (standard 29 DoF and 23 DoF variants), Unitree H1_2, Unitree H2
- **Quadrupeds**: Unitree Go2, Unitree A2, Unitree As2, Unitree R1

### Key Capabilities & Customizations
- **Whole-Body Control (WBC)**: Custom whole-body control task for the Unitree G1 humanoid featuring simultaneous 3D twist velocity tracking and base height commanding.
- **Stepping & Unwanted Motion Suppression**: Penalization logic preventing stepping and body shifting when zero or height-only commands are issued.
- **Motion Tracking / Imitation**: BeyondMimic-style motion tracking for G1 to imitate reference trajectories (e.g., dancing or walking animations).
- **Sim-to-Real Pipeline**: Direct C++ deployment stack using ONNX Runtime, CycloneDDS, and `unitree_sdk2`, validated via `unitree_mujoco` simulation prior to physical robot deployment.

---

## 2. Tech Stack & Dependencies

### Python Environment
- **Python**: 3.11 (Conda environment: `unitree_rl_mjlab`)
- **Physics Engine**: `mujoco==3.5.0`, `mujoco-warp==3.5.0`
- **Simulation Framework**: `mjlab==1.2.0`, `warp-lang==1.12.0`
- **Reinforcement Learning**: `rsl-rl` (PPO runner)
- **Math & Utilities**: `torch`, `scipy>=1.15.0`, `tyro`, `prettytable`, `wandb`

### C++ Stack (Deployment & Simulation)
- **Compiler**: C++17, CMake (>= 3.16)
- **Runtime Inference**: ONNX Runtime (v1.22.0, x86_64 & aarch64 pre-bundled in `deploy/thirdparty/`)
- **Communications**: `cyclonedds`, `unitree_sdk2`
- **Libraries**: `Eigen3`, `Boost`, `spdlog`, `fmt`, `yaml-cpp`, `cnpy`

---

## 3. Directory Layout

```
arcol_rl_env/
├── CONTEXT.md                    # This architecture and operational context file
├── README.md                     # Upstream overview and quickstart (English)
├── README_zh.md                  # Upstream overview (Chinese)
├── setup.py                      # Python package setup for 'unitree_rl_mjlab'
├── start_train.sh                # Reference training launch script (WBC on G1)
├── .wandb.env                    # Repo-local Weights & Biases credentials (ignored by git)
│
├── src/                          # Core Python codebase
│   ├── assets/                   # Robot URDF/MJCF models, meshes, and motion clips
│   │   ├── motions/              # Motion clips (CSV / NPZ) for G1 / G1_23DoF
│   │   └── robots/               # Robot MJCF descriptions (g1, go2, h1_2, etc.)
│   └── tasks/                    # Registered RL environments & MDP configurations
│       ├── wbc/                  # WBC tasks with height control & twist tracking
│       │   ├── config/g1/        # G1 Flat & Rough WBC task registrations & configs
│       │   ├── mdp/              # WBC reward terms, observations, velocity commands
│       │   ├── rl/               # WBC PPO runner overrides
│       │   └── velocity_env_cfg.py # WBC environment configuration factory
│       ├── velocity/             # Standard velocity locomotion tasks (all robots)
│       │   ├── config/           # Robot-specific velocity configs (Go2, G1, H1_2, etc.)
│       │   ├── mdp/              # Velocity MDP terms (curricula, observations, rewards)
│       │   └── velocity_env_cfg.py # Base velocity environment builder
│       └── tracking/             # Motion imitation tasks (BeyondMimic)
│           ├── config/           # G1 tracking configs
│           └── mdp/              # Tracking observations, rewards, and commands
│
├── scripts/                      # Training, evaluation, and utility scripts
│   ├── train.py                  # Main RL training entrypoint (RSL-RL)
│   ├── play.py                   # Policy playback and visualization in MuJoCo viewer
│   ├── list_envs.py              # CLI utility to list all registered mjlab tasks
│   ├── csv_to_npz.py             # Converts CSV motion capture data to NPZ for tracking
│   └── visualize_terrain.py      # Terrain mesh visualization tool
│
├── simulate/                     # C++ standalone MuJoCo simulator ('unitree_mujoco')
│   ├── CMakeLists.txt            # Build script for simulator
│   ├── config.yaml               # Simulator robot and scene configuration
│   └── src/                      # Simulator entrypoint, joystick bridge, and DDS bridge
│
├── deploy/                       # Sim-to-Real C++ deployment framework
│   ├── include/                  # Common FSM, interpolator, and IsaacLab/mjlab wrappers
│   ├── robots/                   # Robot-specific deployment configs and mains
│   │   └── g1/                   # G1 deployment program (g1_ctrl) & FSM configs
│   └── thirdparty/               # Bundled ONNX Runtime & cnpy libraries
│
├── logs/                         # Training checkpoints, Tensorboard & RSL-RL run logs
│   └── rsl_rl/                   # Directory organized by <experiment_name>/<timestamp>/
└── doc/                          # Documentation, installation guides, and assets
```

---

## 4. Key Environments & Tasks

All tasks are registered via `mjlab.tasks.registry.register_mjlab_task` and can be inspected using `python scripts/list_envs.py`.

### 1. Whole-Body Control (WBC) Tasks (Custom Arcol Environments)
- `Unitree-G1-Flat-WBC`: Flat ground locomotion with height commanding and twist tracking.
- `Unitree-G1-Rough-WBC`: Rough/uneven terrain locomotion with height commanding and height-scan raycasting.

### 2. Standard Velocity Tracking Tasks
- `Unitree-G1-Flat` / `Unitree-G1-Rough`
- `Unitree-G1-23Dof-Flat` / `Unitree-G1-23Dof-Rough`
- `Unitree-Go2-Flat` / `Unitree-Go2-Rough`
- `Unitree-H1_2-Flat` / `Unitree-H1_2-Rough`
- `Unitree-A2-Flat` / `Unitree-As2-Flat` / `Unitree-R1-Flat`

### 3. Motion Imitation Tasks
- `Unitree-G1-Tracking` / `Unitree-G1-Tracking-No-State-Estimation`
- `Unitree-G1-23Dof-Tracking` / `Unitree-G1-23Dof-Tracking-No-State-Estimation`

---

## 5. Whole-Body Control (WBC) Design

The WBC implementation in `src/tasks/wbc/` extends basic velocity tracking to support humanoid height adjustment while preventing erratic foot stepping during standing or height transitions:

1. **4D Velocity Command (`UniformVelocityCommand`)**:
   - `command[:, 0]`: Linear velocity $v_x$ (forward/backward)
   - `command[:, 1]`: Linear velocity $v_y$ (lateral)
   - `command[:, 2]`: Angular velocity $\omega_{yaw}$ (turning rate)
   - `command[:, 3]`: Linear vertical velocity / height command $v_z$ or target base height.

2. **Custom Reward Formulations (`src/tasks/wbc/mdp/rewards.py`)**:
   - `track_linear_velocity`: Compares world-frame root linear velocity against $xy$ and $z$ commands.
   - `undesired_velocity`: Penalizes residual horizontal drift ($xy$) and yaw velocity when velocity commands are below threshold, allowing decoupled vertical/height motion.
   - `undesired_stepping`: Penalizes breaking foot contact when horizontal and yaw velocity commands are near zero, ensuring the humanoid does not step or march in place while adjusting height.
   - `stand_still_upper`: Regulates upper-body joint deviations from default pose when stationary.
   - `feet_swing_height` vs `feet_clearance`: Manages foot clearance over terrain and during swing phase.

3. **Actor & Critic Observations**:
   - Actor: Base angular velocity (IMU), projected gravity, command vector, gait phase, joint positions/velocities relative to default, previous actions, terrain raycast height scan.
   - Critic: Actor terms + privileged base linear velocity (IMU) and height scan.

---

## 6. Common Development & Execution Workflows

### Environment Setup
Activate the dedicated conda environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
```

### Launching Training
To train the G1 humanoid with WBC:
```bash
# Using the pre-configured training shell script:
bash start_train.sh

# Or directly calling scripts/train.py:
python scripts/train.py Unitree-G1-Flat-WBC \
  --env.scene.num-envs=4096 \
  --agent.seed=42
```

### Resuming Checkpoints
```bash
python scripts/train.py Unitree-G1-Flat-WBC \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --agent.load_run="2026-08-29_20-12-33" \
  --agent.load_checkpoint="model_18000.pt"
```

### Multi-GPU Training
```bash
python scripts/train.py Unitree-G1-Flat-WBC \
  --gpu-ids 0 1 \
  --env.scene.num-envs=8192
```

### Policy Playback & Visualization
Visualize trained policy rollouts in MuJoCo viewer:
```bash
python scripts/play.py Unitree-G1-Flat-WBC \
  --checkpoint_file="logs/rsl_rl/g1_wbc/<date_time>/model_<iteration>.pt"
```

---

## 7. Sim-to-Real Deployment & Simulation Pipeline

The path from trained policy to physical robot follows:
`Train (PyTorch)` → `Export (ONNX)` → `Simulate (unitree_mujoco)` → `Deploy (g1_ctrl)`

1. **Model Export**:
   - `scripts/train.py` automatically exports `policy.onnx` and `policy.onnx.data` into the experiment log directory upon completion or periodic evaluation.
2. **Pre-Deployment Simulation**:
   - Build `simulate/`:
     ```bash
     cd simulate && mkdir -p build && cd build
     cmake .. && make -j8
     ./unitree_mujoco
     ```
3. **Deployment Compilation**:
   - Copy `policy.onnx` and `policy.onnx.data` to `deploy/robots/g1/config/policy/velocity/v0/exported/`.
   - Compile robot controller:
     ```bash
     cd deploy/robots/g1 && mkdir -p build && cd build
     cmake .. && make
     ```
4. **Execution**:
   - **Simulation Mode**: `./g1_ctrl --network=lo`
   - **Real Robot Mode**: `./g1_ctrl --network=<ethernet_interface>` (e.g. `enp5s0`, robot IP: `192.168.123.222`).

---

## 8. Development Rules & Constraints

- **Do Not Modify `mjlab` Source Code**:
  - Never modify, patch, or change anything in the `mjlab` source code or library installation (e.g. in `site-packages/mjlab/`).
  - All custom tasks, environments, observations, reward terms, commands, and runner overrides must be implemented cleanly in `src/tasks/` or `scripts/` using `mjlab`'s public registration APIs (`register_mjlab_task`, `ManagerBasedRlEnvCfg`).
- **Sim-to-Real Parity**: Any changes to observations in `src/tasks/wbc/velocity_env_cfg.py` must maintain exact parity with the deployment parser in `deploy/robots/g1/src/State_RLBase.cpp` and `deploy/robots/g1/config/config.yaml`.
- **PyTorch CUDA Allocation**: Always set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid out-of-memory errors with large environment counts.
- **NUMA Balancing**: High-throughput CPU-GPU simulation benefits from disabling kernel NUMA balancing (`sudo sysctl -w kernel.numa_balancing=0`).
- **Warp Initialization**: When running custom scripts or interactive sessions with Warp, ensure device initialization before torch allocations: `python -c 'import warp as wp; wp.init(); wp.set_device("cuda:0")'`.
- **W&B Credentials**: Place credentials in `.wandb.env` in the repository root (`WANDB_API_KEY=...`). It is automatically loaded by `scripts/train.py`.

