#!/usr/bin/env bash
# Helper script for deploy runtime environment variables.
# Source from repo root:
#   source Lite3_rl_deploy/environment_variables.sh
#
# Safe default policy:
# - behavior-changing debug overrides are OFF by default
# - debug dumps are ON by default (behavior-neutral)

# Clear behavior-changing overrides in case they leaked from older shells.
unset LITE3_FORCE_RL_START
unset LITE3_FIXED_CMD
unset LITE3_DEFAULT_CMD
unset LITE3_DEPLOY_STATE
unset LITE3_HISTORY_SEED_FILE
unset LITE3_HISTORY_SEED_MODE
unset LITE3_DISABLE_POSTURE_CHECK
unset LITE3_POSTURE_LIMIT_ROLL_DEG
unset LITE3_POSTURE_LIMIT_PITCH_DEG
unset LITE3_POLICY_ASYNC
unset LITE3_POLICY_DECIMATION
unset LITE3_POLICY_CONTROL_DT
unset LITE3_MUJOCO_DT
unset LITE3_MUJOCO_OMEGA_SOURCE
unset LITE3_RANDOM_RESET
unset LITE3_IMU_GYRO_NOISE_STD
unset LITE3_IMU_RPY_NOISE_STD
unset LITE3_IMU_ACC_NOISE_STD

# Keep debug dumps enabled by default (does not affect policy/control outputs).
export LITE3_DEBUG_DUMPS="${LITE3_DEBUG_DUMPS:-5}"

# Optional: override ONNX model path loaded by deploy.
# - Absolute paths are used as-is.
# - Relative paths are resolved from cwd (typically Lite3_rl_deploy/build).
# export LITE3_POLICY_ONNX="/home/sail/Lite3/Lite3_rl_deploy/policy/ppo/policy.onnx"

# Optional behavior-changing toggles (debug only, opt-in):
# export LITE3_FORCE_RL_START=1
# export LITE3_FIXED_CMD="0 0 0"
# export LITE3_DEFAULT_CMD="0 0 0"
# export LITE3_DEPLOY_STATE="/path/to/deploy_snapshot.json"
# export LITE3_HISTORY_SEED_FILE="/path/to/history_seed.txt"
# export LITE3_DISABLE_POSTURE_CHECK=1
# export LITE3_POSTURE_LIMIT_ROLL_DEG=40
# export LITE3_POSTURE_LIMIT_PITCH_DEG=90
# export LITE3_MUJOCO_OMEGA_SOURCE="world_to_body"  # debug only; baseline is qvel_body
