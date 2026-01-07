#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --ckpt /path/to/model_XXXX.pt" >&2
  exit 1
}

CKPT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT_ARG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "$CKPT_ARG" ]]; then
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CKPT_PATH="$CKPT_ARG"
if [[ ! -f "$CKPT_PATH" ]]; then
  if [[ -f "$SCRIPT_DIR/$CKPT_ARG" ]]; then
    CKPT_PATH="$SCRIPT_DIR/$CKPT_ARG"
  elif [[ -f "$REPO_ROOT/$CKPT_ARG" ]]; then
    CKPT_PATH="$REPO_ROOT/$CKPT_ARG"
  else
    echo "Checkpoint not found: $CKPT_ARG" >&2
    exit 1
  fi
fi

COMPARE_INPUT="$REPO_ROOT/Lite3_rl_training/debug_training_obs/flat_step_00.txt"
if [[ ! -f "$COMPARE_INPUT" ]]; then
  echo "Missing compare input: $COMPARE_INPUT" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/ppo"
ONNX_OUT="$SCRIPT_DIR/ppo/policy.onnx"
TS_OUT="$(mktemp -t lite3_policy_ts_XXXXXX.pt)"

cleanup() {
  rm -f "$TS_OUT"
}
trap cleanup EXIT

echo "[1/2] Building TorchScript policy from checkpoint..."
export CKPT_PATH TS_OUT REPO_ROOT
python - <<'PY'
import os
import sys
from pathlib import Path

import torch

ckpt_path = Path(os.environ["CKPT_PATH"]).resolve()
ts_out = Path(os.environ["TS_OUT"]).resolve()
repo_root = Path(os.environ["REPO_ROOT"]).resolve()

sys.path.insert(0, str(repo_root / "Lite3_rl_training" / "rsl_rl"))
from rsl_rl.modules.actor_critic import ActorCritic

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model_state_dict"]

n_priv = state["env_factor_encoder.0.weight"].shape[1]
enc_h0 = state["env_factor_encoder.0.weight"].shape[0]
enc_h1 = state["env_factor_encoder.2.weight"].shape[0]
enc_lat = state["env_factor_encoder.4.weight"].shape[0]
enc_hidden = [enc_h0, enc_h1]

adapt_h0 = state["adaptation_module.0.weight"].shape[0]
adapt_h1 = state["adaptation_module.2.weight"].shape[0]
adapt_in = state["adaptation_module.0.weight"].shape[1]
adapt_hidden = [adapt_h0, adapt_h1]

actor_hidden = [
    state["actor.0.weight"].shape[0],
    state["actor.2.weight"].shape[0],
    state["actor.4.weight"].shape[0],
]
critic_hidden = [
    state["critic.0.weight"].shape[0],
    state["critic.2.weight"].shape[0],
    state["critic.4.weight"].shape[0],
]

n_actions = state["actor.6.weight"].shape[0]
num_obs = 117
num_obs_history = adapt_in

ac = ActorCritic(
    num_obs=num_obs,
    num_privileged_obs=n_priv,
    num_obs_history=num_obs_history,
    num_actions=n_actions,
    actor_hidden_dims=actor_hidden,
    critic_hidden_dims=critic_hidden,
    encoder_hidden_dims=enc_hidden,
    adaptation_hidden_dims=adapt_hidden,
    encoder_latent_dims=enc_lat,
    activation="elu",
).eval()
ac.load_state_dict(state, strict=True)

scripted = torch.jit.script(ac.export_policy())
scripted.save(str(ts_out))
print("Saved TorchScript:", ts_out)
PY

echo "[2/2] Exporting ONNX + parity compare..."
python "$SCRIPT_DIR/pt2onnx.py" \
  --torchscript "$TS_OUT" \
  --out "$ONNX_OUT" \
  --num-obs 117 \
  --history-len 40 \
  --compare-input "$COMPARE_INPUT"

echo "Done. ONNX saved to: $ONNX_OUT"
