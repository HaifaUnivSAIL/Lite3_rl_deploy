# pt2onnx_.py — robust ONNX export for ActorCritic (deploy-ready)

import os, sys
from pathlib import Path
import torch
import onnx
import onnxruntime as ort

# ----------------------------
# Locate training repo (sibling-aware; nested rsl_rl/rsl_rl supported)
# ----------------------------
THIS_FILE   = Path(__file__).resolve()         # .../Lite3_rl_deploy/policy/pt2onnx_.py
DEPLOY_ROOT = THIS_FILE.parents[1]             # .../Lite3_rl_deploy
PARENT_ROOT = DEPLOY_ROOT.parent               # common parent of training & deploy

def _is_pkg(p: Path) -> bool:
    return (p / "__init__.py").is_file()

def _find_dist_root(cand: Path):
    """Return path to add to sys.path so 'import rsl_rl' works."""
    if not cand or not cand.exists():
        return None
    # flat: repo_root/rsl_rl/__init__.py  → add repo_root to sys.path
    if _is_pkg(cand / "rsl_rl"):
        return cand
    # nested: repo_root/rsl_rl/rsl_rl/__init__.py → add repo_root/rsl_rl to sys.path
    if _is_pkg(cand / "rsl_rl" / "rsl_rl"):
        return cand / "rsl_rl"
    return None

env_hint = os.getenv("LITE3_TRAINING_DIR")
candidates = []
if env_hint:
    ph = Path(env_hint).expanduser().resolve()
    candidates += [ph, ph / "rsl_rl"]

# common sibling names
candidates += [
    PARENT_ROOT / "Lite3_rl_training",
    PARENT_ROOT / "Lite3_rl_training" / "rsl_rl",
    PARENT_ROOT / "rl_training",
    PARENT_ROOT / "rl_training" / "rsl_rl",
]

# scan siblings
try:
    for sib in PARENT_ROOT.iterdir():
        if sib.is_dir():
            candidates += [sib, sib / "rsl_rl"]
except Exception:
    pass

DIST_ROOT = None
for cand in candidates:
    dist = _find_dist_root(cand)
    if dist:
        DIST_ROOT = dist
        sys.path.insert(0, str(dist))  # ensure local training code wins over any installed egg
        break

if DIST_ROOT is None:
    raise ImportError(
        "Could not locate 'rsl_rl'. Set LITE3_TRAINING_DIR=/path/to/Lite3_rl_training "
        "or keep Lite3_rl_training as a sibling of Lite3_rl_deploy."
    )

from rsl_rl.modules.actor_critic import ActorCritic  # now resolvable


# ----------------------------
# Helpers: load checkpoint & infer architecture from weights
# ----------------------------
def load_state_dict_any(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt  # raw state_dict
        

def infer_arch_dims_from_state(state):
    """
    Infer dims to exactly match training so we can load with strict=True.
    Returns:
        n_priv (int), enc_lat (int),
        actor_hidden (list[int]), critic_hidden (list[int]),
        enc_hidden (list[int]), adapt_hidden (list[int]),
        n_actions (int),
        actor_in_features (int), adapt_in_features (int)
    """
    # --- encoder / privileged ---
    # env_factor_encoder: 0 -> hidden0, 2 -> hidden1, 4 -> latent
    n_priv  = state["env_factor_encoder.0.weight"].shape[1]
    enc_h0  = state["env_factor_encoder.0.weight"].shape[0]
    enc_h1  = state["env_factor_encoder.2.weight"].shape[0]
    enc_lat = state["env_factor_encoder.4.weight"].shape[0]
    enc_hidden = [enc_h0, enc_h1]

    # --- adaptation module (history encoder) ---
    adapt_h0 = state["adaptation_module.0.weight"].shape[0]
    adapt_h1 = state["adaptation_module.2.weight"].shape[0]
    adapt_in = state["adaptation_module.0.weight"].shape[1]  # should equal num_obs_history
    adapt_hidden = [adapt_h0, adapt_h1]

    # --- actor / critic ---
    # actor: 0->h1, 2->h2, 4->h3, 6->out
    a_h1 = state["actor.0.weight"].shape[0]
    a_h2 = state["actor.2.weight"].shape[0]
    a_h3 = state["actor.4.weight"].shape[0]
    n_actions = state["actor.6.weight"].shape[0]
    actor_hidden = [a_h1, a_h2, a_h3]
    actor_in_features = state["actor.0.weight"].shape[1]  # num_obs + encoder_latent_dims

    c_h1 = state["critic.0.weight"].shape[0]
    c_h2 = state["critic.2.weight"].shape[0]
    c_h3 = state["critic.4.weight"].shape[0]
    critic_hidden = [c_h1, c_h2, c_h3]

    return (n_priv, enc_lat, actor_hidden, critic_hidden,
            enc_hidden, adapt_hidden, n_actions, actor_in_features, adapt_in)


# ----------------------------
# Export
# ----------------------------
@torch.no_grad()
def export_actor_critic_to_onnx(
    ckpt_path: str,
    onnx_out: str,
    num_obs: int,
    obs_his_num: int,
    # opset & exporter settings
    opset: int = 17,
):
    device = torch.device("cpu")

    # Load state dict
    state = load_state_dict_any(ckpt_path, device=device)

    # Infer dims from checkpoint so load_state_dict(strict=True) passes
    (n_priv, enc_lat, actor_hidden, critic_hidden,
     enc_hidden, adapt_hidden, n_actions,
     actor_in_features, adapt_in_features) = infer_arch_dims_from_state(state)

    # Sanity: ensure our runtime num_obs aligns with actor input side
    if actor_in_features != (num_obs + enc_lat):
        raise ValueError(
            f"Actor input mismatch: ckpt expects {actor_in_features} = num_obs({num_obs}) + enc_lat({enc_lat}). "
            f"Check your num_obs or encoder_latent_dims."
        )

    # Compute deploy input sizes
    num_obs_history = num_obs * obs_his_num
    total_in = num_obs + num_obs_history

    # Sanity: ensure history width matches checkpoint adaptation input
    if adapt_in_features != num_obs_history:
        raise ValueError(
            f"History size mismatch: ckpt adaptation expects {adapt_in_features}, "
            f"but num_obs_history computed as {num_obs_history} = {num_obs} * {obs_his_num}."
        )

    # Rebuild AC with EXACT training widths so we can strict-load
    ac = ActorCritic(
        num_obs=num_obs,
        num_privileged_obs=n_priv,           # match ckpt (even if unused at deploy)
        num_obs_history=num_obs_history,
        num_actions=n_actions,
        actor_hidden_dims=actor_hidden,      # e.g. [512,256,128]
        critic_hidden_dims=critic_hidden,
        encoder_hidden_dims=enc_hidden,      # e.g. [256,128]
        adaptation_hidden_dims=adapt_hidden, # e.g. [256,32]
        encoder_latent_dims=enc_lat,         # e.g. 18
        activation="elu",
    ).to(device).eval()

    ac.load_state_dict(state, strict=True)

    # Wrapper with ONNX-friendly splitting (no slicing views)
    class DeployPolicy(torch.nn.Module):
        def __init__(self, actor, adaptation_module, num_obs, num_obs_history):
            super().__init__()
            self.actor = actor
            self.adaptation_module = adaptation_module
            self.num_obs = num_obs
            self.num_obs_history = num_obs_history
            # Gather indices → ONNX "Gather"
            self.register_buffer("idx_obs",  torch.arange(num_obs, dtype=torch.long), persistent=False)
            self.register_buffer("idx_hist", torch.arange(num_obs, num_obs + num_obs_history, dtype=torch.long), persistent=False)

        def forward(self, obs_flat: torch.Tensor):
            # 1) kill view/stride weirdness up front (maps to ONNX Identity)
            obs_flat = obs_flat.clone().contiguous()
            # 2) ensure a plain 2D shape (maps to ONNX Reshape)
            obs_flat = torch.reshape(obs_flat, (-1, self.num_obs + self.num_obs_history))
            # 3) ONNX-friendly splits (Gather)
            obs      = obs_flat.index_select(1, self.idx_obs)    # [B, num_obs]
            obs_hist = obs_flat.index_select(1, self.idx_hist)   # [B, num_obs_history]
            latent   = self.adaptation_module(obs_hist)          # [B, enc_lat]
            return self.actor(torch.cat((obs, latent), dim=1))   # [B, n_actions]

    deploy = DeployPolicy(ac.actor, ac.adaptation_module, num_obs, num_obs_history).to(device).eval()

    # Dummy input with exact static shape the runner uses
    dummy = torch.ones(1, total_in, dtype=torch.float32, device=device)

    # Export ONNX with names runner expects
    torch.onnx.export(
        deploy, (dummy,), onnx_out,
        input_names=["obs"], output_names=["action"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,                     # keep static [1, total_in]
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False                           # <- use classic eager exporter
    )


    # Validate & smoke test
    m = onnx.load(onnx_out); onnx.checker.check_model(m)
    print("Exported:", onnx_out)
    print("INPUTS :", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("OUTPUTS:", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])

    sess = ort.InferenceSession(onnx_out, providers=["CPUExecutionProvider"])
    import numpy as np
    out = sess.run(["action"], {"obs": np.ones((1, total_in), np.float32)})[0]
    assert out.shape == (1, n_actions), f"unexpected action shape {out.shape}"
    print("ONNX runtime smoke test OK, action shape:", out.shape)


if __name__ == "__main__":
    # ==== fill from your deploy config ====
    NUM_OBS     = 117          # current obs dim (no history)
    OBS_HIS_NUM = 40           # number of stacked past obs frames
    CKPT_PATH   = "model_3000.pt"     # path to your PPO checkpoint
    ONNX_OUT    = "policy_3000.onnx"  # output path

    export_actor_critic_to_onnx(
        ckpt_path=CKPT_PATH,
        onnx_out=ONNX_OUT,
        num_obs=NUM_OBS,
        obs_his_num=OBS_HIS_NUM,
        opset=17,
    )
