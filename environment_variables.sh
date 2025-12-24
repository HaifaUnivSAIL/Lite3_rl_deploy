#!/usr/bin/env bash
# Helper script to export the environment variables commonly used while
# debugging the Lite3 deployment pipeline. Source it from the project root:
#   source Lite3_rl_deploy/environment_variables.sh

# Fixed command sent to the policy (lin_vel_x lin_vel_y ang_vel_yaw)
export LITE3_FIXED_CMD="0.8 0 0"

# Optional: override which ONNX model the deploy binary loads.
# - Absolute paths are used as-is.
# - Relative paths are resolved from the current working directory (typically `Lite3_rl_deploy/build`).
# export LITE3_POLICY_ONNX="/home/sail/Lite3/Lite3_rl_deploy/policy/ppo/policy.onnx"

# Optional: disable random reset for MuJoCo C++ sim (USE_MJCPP)
# export LITE3_RANDOM_RESET=0

# Disable the posture safety guard (set to 0 to re-enable)
export LITE3_DISABLE_POSTURE_CHECK=1

# Optional: adjust posture guard thresholds (degrees) when enabled
export LITE3_POSTURE_LIMIT_ROLL_DEG=40
export LITE3_POSTURE_LIMIT_PITCH_DEG=90

# Path to the saved Mujoco snapshot used to seed play.py / training resets
export LITE3_DEPLOY_STATE="/home/sail/Lite3/Lite3_rl_training/legged_gym/legged_gym/envs/base/deploy_snapshot.json"
