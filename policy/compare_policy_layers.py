#!/usr/bin/env python3
"""Compare intermediate activations between a PT policy checkpoint and ONNX policy.

Supports:
1. Dummy input (legacy mode).
2. Real experiment inputs from deploy dumps (`debug_cpp_step*.txt`).
3. Real experiment inputs from training dumps (`debug_play_step*.npz`).
4. Combined deploy+train input cases for side-by-side PT/ONNX layer parity.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_ONNX = REPO_ROOT / "Lite3_rl_deploy" / "policy" / "ppo" / "policy.onnx"
DEFAULT_PT = (
    REPO_ROOT
    / "rl_training_new"
    / "logs"
    / "rsl_rl"
    / "two_leg_stand"
    / "2026-02-12_13-02-05_parity_latest"
    / "model_6000.pt"
)
DEFAULT_OUT_DIR = REPO_ROOT / "rl_training_new" / "lite3_debug" / "layer_compare"
DEFAULT_DEPLOY_DUMP_DIR = REPO_ROOT / "rl_training_new" / "lite3_debug" / "deploy"
DEFAULT_TRAIN_DUMP_DIR = REPO_ROOT / "rl_training_new" / "lite3_debug" / "train"

LAYER_ORDER = [
    "input.obs_flat",
    "split.obs",
    "split.obs_hist",
    "adaptation_module.0",
    "adaptation_module.1",
    "adaptation_module.2",
    "adaptation_module.3",
    "adaptation_module.4",
    "concat.obs_latent",
    "actor.0",
    "actor.1",
    "actor.2",
    "actor.3",
    "actor.4",
    "actor.5",
    "actor.6",
]


def _is_pkg(path: Path) -> bool:
    return (path / "__init__.py").is_file()


def _find_dist_root(candidate: Path) -> Path | None:
    if not candidate.exists():
        return None
    if _is_pkg(candidate / "rsl_rl"):
        return candidate
    if _is_pkg(candidate / "rsl_rl" / "rsl_rl"):
        return candidate / "rsl_rl"
    return None


def _ensure_rsl_rl_importable() -> None:
    env_hint = os.getenv("LITE3_TRAINING_DIR")
    candidates: List[Path] = []

    if env_hint:
        hint = Path(env_hint).expanduser().resolve()
        candidates += [hint, hint / "rsl_rl"]

    candidates += [
        REPO_ROOT / "rl_training_new",
        REPO_ROOT / "rl_training_new" / "rsl_rl",
        REPO_ROOT / "Lite3_rl_training",
        REPO_ROOT / "Lite3_rl_training" / "rsl_rl",
        REPO_ROOT / "rl_training",
        REPO_ROOT / "rl_training" / "rsl_rl",
    ]

    try:
        for sibling in REPO_ROOT.iterdir():
            if sibling.is_dir():
                candidates += [sibling, sibling / "rsl_rl"]
    except OSError:
        pass

    for candidate in candidates:
        dist_root = _find_dist_root(candidate)
        if dist_root is not None:
            sys.path.insert(0, str(dist_root))
            return

    raise ImportError(
        "Could not locate rsl_rl package. "
        "Set LITE3_TRAINING_DIR or keep rl_training_new/Lite3_rl_training as repo siblings."
    )


def _load_state_dict_any(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "weights"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")

    if any(k.startswith("actor_critic.") for k in checkpoint):
        stripped = {}
        for k, v in checkpoint.items():
            stripped[k.replace("actor_critic.", "", 1)] = v
        checkpoint = stripped

    return checkpoint


def _infer_arch_dims_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int | List[int]]:
    n_priv = int(state["env_factor_encoder.0.weight"].shape[1])
    enc_h0 = int(state["env_factor_encoder.0.weight"].shape[0])
    enc_h1 = int(state["env_factor_encoder.2.weight"].shape[0])
    enc_lat = int(state["env_factor_encoder.4.weight"].shape[0])

    adapt_h0 = int(state["adaptation_module.0.weight"].shape[0])
    adapt_h1 = int(state["adaptation_module.2.weight"].shape[0])
    adapt_in = int(state["adaptation_module.0.weight"].shape[1])

    a_h1 = int(state["actor.0.weight"].shape[0])
    a_h2 = int(state["actor.2.weight"].shape[0])
    a_h3 = int(state["actor.4.weight"].shape[0])
    n_actions = int(state["actor.6.weight"].shape[0])
    actor_in = int(state["actor.0.weight"].shape[1])

    c_h1 = int(state["critic.0.weight"].shape[0])
    c_h2 = int(state["critic.2.weight"].shape[0])
    c_h3 = int(state["critic.4.weight"].shape[0])

    return {
        "n_priv": n_priv,
        "enc_lat": enc_lat,
        "enc_hidden": [enc_h0, enc_h1],
        "adapt_hidden": [adapt_h0, adapt_h1],
        "adapt_in": adapt_in,
        "actor_hidden": [a_h1, a_h2, a_h3],
        "actor_in": actor_in,
        "critic_hidden": [c_h1, c_h2, c_h3],
        "n_actions": n_actions,
    }


class DeployPolicy(torch.nn.Module):
    def __init__(self, actor: torch.nn.Sequential, adaptation_module: torch.nn.Sequential, num_obs: int, num_obs_history: int):
        super().__init__()
        self.actor = actor
        self.adaptation_module = adaptation_module
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs = obs_flat[:, : self.num_obs]
        obs_hist = obs_flat[:, self.num_obs : self.num_obs + self.num_obs_history]
        latent = self.adaptation_module(obs_hist)
        return self.actor(torch.cat((obs, latent), dim=-1))


def _build_deploy_policy_from_ckpt(pt_path: Path, num_obs: int, history_len: int):
    _ensure_rsl_rl_importable()
    from rsl_rl.modules.actor_critic import ActorCritic

    state = _load_state_dict_any(pt_path)
    dims = _infer_arch_dims_from_state(state)
    num_obs_history = num_obs * history_len

    if dims["adapt_in"] != num_obs_history:
        raise ValueError(
            f"History mismatch: checkpoint expects {dims['adapt_in']}, "
            f"but num_obs*history_len={num_obs_history}."
        )
    if dims["actor_in"] != num_obs + dims["enc_lat"]:
        raise ValueError(
            f"Actor input mismatch: checkpoint expects {dims['actor_in']}, "
            f"but num_obs+enc_lat={num_obs + dims['enc_lat']}."
        )

    with contextlib.redirect_stdout(io.StringIO()):
        ac = ActorCritic(
            num_obs=num_obs,
            num_privileged_obs=int(dims["n_priv"]),
            num_obs_history=num_obs_history,
            num_actions=int(dims["n_actions"]),
            actor_hidden_dims=list(dims["actor_hidden"]),
            critic_hidden_dims=list(dims["critic_hidden"]),
            encoder_hidden_dims=list(dims["enc_hidden"]),
            adaptation_hidden_dims=list(dims["adapt_hidden"]),
            encoder_latent_dims=int(dims["enc_lat"]),
            activation="elu",
        ).eval()
    ac.load_state_dict(state, strict=True)

    deploy = DeployPolicy(
        actor=ac.actor,
        adaptation_module=ac.adaptation_module,
        num_obs=num_obs,
        num_obs_history=num_obs_history,
    ).eval()
    return deploy, int(dims["n_actions"]), num_obs_history


def _capture_pt_intermediates(policy: DeployPolicy, obs_flat: np.ndarray) -> OrderedDict[str, np.ndarray]:
    x = torch.from_numpy(obs_flat.astype(np.float32))
    out = OrderedDict()

    with torch.no_grad():
        obs = x[:, : policy.num_obs]
        obs_hist = x[:, policy.num_obs : policy.num_obs + policy.num_obs_history]
        out["input.obs_flat"] = x.numpy().copy()
        out["split.obs"] = obs.numpy().copy()
        out["split.obs_hist"] = obs_hist.numpy().copy()

        latent = obs_hist
        for i, layer in enumerate(policy.adaptation_module):
            latent = layer(latent)
            out[f"adaptation_module.{i}"] = latent.detach().cpu().numpy().copy()

        concat = torch.cat((obs, latent), dim=-1)
        out["concat.obs_latent"] = concat.detach().cpu().numpy().copy()

        actor = concat
        for i, layer in enumerate(policy.actor):
            actor = layer(actor)
            out[f"actor.{i}"] = actor.detach().cpu().numpy().copy()

    return out


def _augment_onnx_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model)
    value_info = {}
    for collection in (inferred.graph.value_info, inferred.graph.input, inferred.graph.output):
        for v in collection:
            value_info[v.name] = v

    output_names = {o.name for o in model.graph.output}
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in output_names:
                continue
            if name in value_info:
                model.graph.output.append(value_info[name])
                output_names.add(name)
    return model


def _create_onnx_session_all_outputs(onnx_path: Path) -> Tuple[onnx.ModelProto, ort.InferenceSession, List[str]]:
    model = onnx.load(str(onnx_path))
    model = _augment_onnx_outputs(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    output_names = [meta.name for meta in sess.get_outputs()]
    return model, sess, output_names


def _run_onnx_all_outputs(
    sess: ort.InferenceSession,
    output_names: List[str],
    obs_flat: np.ndarray,
) -> Dict[str, np.ndarray]:
    input_name = sess.get_inputs()[0].name
    values = sess.run(None, {input_name: obs_flat.astype(np.float32, copy=False)})
    return {n: v for n, v in zip(output_names, values)}


def _alias_onnx_intermediates(model: onnx.ModelProto, raw_outputs: Dict[str, np.ndarray]) -> OrderedDict[str, np.ndarray]:
    aliased: OrderedDict[str, np.ndarray] = OrderedDict()
    gather_index = 0
    last_linear_alias = None
    activation_ops = {"Elu", "Relu", "LeakyRelu", "SELU", "Sigmoid", "Tanh"}

    for node in model.graph.node:
        if not node.output:
            continue
        output_name = node.output[0]
        value = raw_outputs.get(output_name)
        if value is None:
            continue

        alias = None
        if node.op_type == "Reshape":
            alias = "input.obs_flat"
        elif node.op_type == "Gather":
            alias = "split.obs" if gather_index == 0 else "split.obs_hist"
            gather_index += 1
        elif node.op_type == "Concat":
            alias = "concat.obs_latent"
        elif node.op_type == "Gemm":
            if len(node.input) >= 2 and node.input[1].endswith(".weight"):
                alias = node.input[1].removesuffix(".weight")
                last_linear_alias = alias
        elif node.op_type in activation_ops and last_linear_alias is not None:
            parts = last_linear_alias.split(".")
            if len(parts) == 2 and parts[1].isdigit():
                alias = f"{parts[0]}.{int(parts[1]) + 1}"
            else:
                alias = f"{last_linear_alias}.act"

        if alias is None:
            continue
        aliased[alias] = value

    if "action" in raw_outputs and "actor.6" not in aliased:
        aliased["actor.6"] = raw_outputs["action"]

    return aliased


def _flatten(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1).astype(np.float64, copy=False)


def _compare_layers(pt_layers: OrderedDict[str, np.ndarray], onnx_layers: OrderedDict[str, np.ndarray]):
    common = [k for k in LAYER_ORDER if k in pt_layers and k in onnx_layers]
    rows = []
    for name in common:
        pt = _flatten(pt_layers[name])
        ox = _flatten(onnx_layers[name])

        shape_match = pt.shape == ox.shape
        row = {
            "layer": name,
            "pt_shape": list(pt_layers[name].shape),
            "onnx_shape": list(onnx_layers[name].shape),
            "shape_match": bool(shape_match),
        }
        if not shape_match:
            row.update(
                {
                    "mean_abs_diff": None,
                    "max_abs_diff": None,
                    "rmse": None,
                    "l2_diff": None,
                    "rel_l2": None,
                    "nrmse_rms": None,
                    "cosine_similarity": None,
                    "pt_mean": float(pt.mean()) if pt.size else 0.0,
                    "onnx_mean": float(ox.mean()) if ox.size else 0.0,
                    "pt_std": float(pt.std()) if pt.size else 0.0,
                    "onnx_std": float(ox.std()) if ox.size else 0.0,
                }
            )
            rows.append(row)
            continue

        diff_vec = pt - ox
        abs_diff = np.abs(diff_vec)
        mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
        max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
        rmse = float(math.sqrt(np.mean(diff_vec * diff_vec))) if abs_diff.size else 0.0
        l2_diff = float(np.linalg.norm(diff_vec))
        pt_l2 = float(np.linalg.norm(pt))
        pt_rms = float(math.sqrt(np.mean(pt * pt))) if pt.size else 0.0

        rms_floor = 1e-6
        l2_floor = float(math.sqrt(max(pt.size, 1))) * rms_floor
        rel_l2 = float(l2_diff / max(pt_l2, l2_floor))
        nrmse_rms = float(rmse / max(pt_rms, rms_floor))

        denom = (np.linalg.norm(pt) * np.linalg.norm(ox)) + 1e-12
        cosine = float(np.dot(pt, ox) / denom) if pt.size else 1.0
        row.update(
            {
                "mean_abs_diff": mean_abs,
                "max_abs_diff": max_abs,
                "rmse": rmse,
                "l2_diff": l2_diff,
                "rel_l2": rel_l2,
                "nrmse_rms": nrmse_rms,
                "cosine_similarity": cosine,
                "pt_mean": float(pt.mean()) if pt.size else 0.0,
                "onnx_mean": float(ox.mean()) if ox.size else 0.0,
                "pt_std": float(pt.std()) if pt.size else 0.0,
                "onnx_std": float(ox.std()) if ox.size else 0.0,
            }
        )
        rows.append(row)
    return common, rows


def _plot_layer_traces(
    out_path: Path,
    layers: Iterable[str],
    pt_layers: Dict[str, np.ndarray],
    onnx_layers: Dict[str, np.ndarray],
    max_points: int,
    title_prefix: str | None = None,
) -> None:
    layer_list = list(layers)
    if not layer_list:
        return

    cols = 3
    rows = math.ceil(len(layer_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.2 * rows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, name in enumerate(layer_list):
        ax = axes_flat[idx]
        pt = _flatten(pt_layers[name])
        ox = _flatten(onnx_layers[name])
        n = min(max_points, pt.size, ox.size)
        if n == 0:
            ax.set_title(f"{name} (empty)")
            ax.axis("off")
            continue
        x = np.arange(n)
        ax.plot(x, pt[:n], label="PT", linewidth=1.3)
        ax.plot(x, ox[:n], label="ONNX", linewidth=1.3, linestyle="--")
        ax.set_title(f"{name} (first {n}/{pt.size})", fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

    for ax in axes_flat[len(layer_list) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_diff_bars(out_path: Path, rows: List[Dict[str, object]], title_prefix: str | None = None) -> None:
    valid = [r for r in rows if r["shape_match"]]
    if not valid:
        return

    labels = [str(r["layer"]) for r in valid]
    mean_abs = np.array([float(r["mean_abs_diff"]) for r in valid])
    max_abs = np.array([float(r["max_abs_diff"]) for r in valid])

    x = np.arange(len(labels))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 0.8), 4.8))
    ax.bar(x - width / 2.0, mean_abs, width=width, label="mean |PT-ONNX|")
    ax.bar(x + width / 2.0, max_abs, width=width, label="max |PT-ONNX|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Absolute difference")
    title = "Per-layer PT vs ONNX error"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_heatmap(
    out_path: Path,
    case_reports: List[Dict[str, object]],
    metric_key: str,
    title: str,
    log10_scale: bool = False,
) -> None:
    if not case_reports:
        return

    case_labels = [str(c["case"]) for c in case_reports]
    observed_layers = {
        str(row["layer"])
        for c in case_reports
        for row in c["rows"]  # type: ignore[index]
        if isinstance(row, dict) and row.get("shape_match")
    }
    if not observed_layers:
        return

    layers = [name for name in LAYER_ORDER if name in observed_layers]
    extra_layers = sorted(observed_layers - set(layers))
    layers += extra_layers

    matrix = np.full((len(layers), len(case_labels)), np.nan, dtype=np.float64)
    for col, case in enumerate(case_reports):
        rows = case["rows"]  # type: ignore[index]
        layer_to_row = {
            str(r["layer"]): r
            for r in rows
            if isinstance(r, dict) and r.get("shape_match")
        }
        for row_idx, layer_name in enumerate(layers):
            row = layer_to_row.get(layer_name)
            if row is None:
                continue
            value = row.get(metric_key)
            if value is None:
                continue
            matrix[row_idx, col] = float(value)

    if np.all(np.isnan(matrix)):
        return

    display = matrix.copy()
    cbar_label = metric_key
    if log10_scale:
        display = np.where(np.isnan(display), np.nan, np.log10(np.maximum(display, 1e-12)))
        cbar_label = f"log10({metric_key})"

    fig_w = max(8.0, 1.0 * len(case_labels) + 3.0)
    fig_h = max(5.0, 0.36 * len(layers) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#e5e7eb")
    im = ax.imshow(display, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(case_labels)))
    ax.set_xticklabels(case_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_case_csv(path: Path, case_reports: List[Dict[str, object]]) -> None:
    keys = [
        "case",
        "source",
        "step",
        "input_file",
        "layer",
        "pt_shape",
        "onnx_shape",
        "shape_match",
        "mean_abs_diff",
        "max_abs_diff",
        "rmse",
        "l2_diff",
        "rel_l2",
        "nrmse_rms",
        "cosine_similarity",
        "pt_mean",
        "onnx_mean",
        "pt_std",
        "onnx_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for case in case_reports:
            for row in case["rows"]:  # type: ignore[index]
                merged = {
                    "case": case["case"],
                    "source": case["source"],
                    "step": case["step"],
                    "input_file": case["input_file"],
                }
                merged.update(row)
                writer.writerow(merged)


def _sanitize_case_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _write_case_layers_npz(path: Path, case_layers: Dict[str, OrderedDict[str, np.ndarray]]) -> None:
    payload: Dict[str, np.ndarray] = {}
    for case_name, layers in case_layers.items():
        prefix = _sanitize_case_id(case_name)
        for layer_name, value in layers.items():
            payload[f"{prefix}__{layer_name.replace('.', '__')}"] = value
    np.savez(path, **payload)


def _write_case_inputs_npz(path: Path, cases: List[Dict[str, object]]) -> None:
    payload: Dict[str, np.ndarray] = {}
    for case in cases:
        key = _sanitize_case_id(str(case["name"]))
        payload[key] = np.asarray(case["obs_flat"])  # type: ignore[arg-type]
    np.savez(path, **payload)


def _parse_step_index(path: Path) -> int | None:
    match = re.search(r"step(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _discover_files(root: Path, pattern: str) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files = list(root.glob(pattern))
    if files:
        return files
    return list(root.rglob(pattern))


def _index_by_step(files: List[Path]) -> Dict[int, Path]:
    by_step: Dict[int, Path] = {}
    for path in files:
        step = _parse_step_index(path)
        if step is None:
            continue
        if step not in by_step:
            by_step[step] = path
            continue
        try:
            if path.stat().st_mtime > by_step[step].stat().st_mtime:
                by_step[step] = path
        except FileNotFoundError:
            continue
    return by_step


def _parse_steps_arg(steps_spec: str | None) -> List[int] | None:
    if steps_spec is None:
        return None
    text = steps_spec.strip()
    if not text or text.lower() in {"all", "*"}:
        return None

    steps: set[int] = set()
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid step range token: '{token}'")
            start, end = int(parts[0]), int(parts[1])
            if end < start:
                raise ValueError(f"Invalid descending step range: '{token}'")
            for step in range(start, end + 1):
                steps.add(step)
            continue
        steps.add(int(token))

    if not steps:
        return None
    return sorted(steps)


def _parse_deploy_dump_obs_flat(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("obs_flat "):
                continue
            parts = line.split()
            if len(parts) <= 1:
                break
            values = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if values.size > 0:
                return values
            break
    raise ValueError(f"Could not parse 'obs_flat' from deploy dump: {path}")


def _load_train_dump_obs_flat(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        if "obs_flat" not in npz.files:
            raise KeyError(f"'obs_flat' missing in training dump: {path}")
        return np.asarray(npz["obs_flat"], dtype=np.float32).reshape(-1)


def _ensure_obs_shape(obs_flat: np.ndarray, total_in: int, label: str) -> np.ndarray:
    x = np.asarray(obs_flat, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        pass
    else:
        raise ValueError(f"{label}: expected obs_flat shape [N] or [1,N], got {list(x.shape)}")
    if x.shape[1] != total_in:
        raise ValueError(f"{label}: obs_flat dim mismatch, got {x.shape[1]}, expected {total_in}")
    return x


def _require_steps(available: List[int], requested: List[int] | None, source_name: str) -> List[int]:
    if requested is None:
        return sorted(available)
    missing = [step for step in requested if step not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing {source_name} dump steps: {missing}; available={sorted(available)}"
        )
    return requested


def _build_input_cases(
    input_source: str,
    total_in: int,
    seed: int,
    dummy_scale: float,
    deploy_dir: Path | None,
    train_dir: Path | None,
    steps: List[int] | None,
) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []

    if input_source == "dummy":
        rng = np.random.default_rng(seed)
        obs_flat = (rng.standard_normal((1, total_in)) * dummy_scale).astype(np.float32)
        cases.append(
            {
                "name": "dummy_step0",
                "source": "dummy",
                "step": 0,
                "input_file": None,
                "obs_flat": obs_flat,
            }
        )
        return cases

    if input_source in {"deploy", "both"}:
        if deploy_dir is None:
            raise ValueError("--deploy-dir is required when input source includes deploy.")
        deploy_candidates = _discover_files(deploy_dir, "debug_cpp_step*.txt")
        deploy_files = _index_by_step(deploy_candidates)
        if not deploy_files:
            raise FileNotFoundError(f"No deploy dumps found in: {deploy_dir}")
        selected = _require_steps(sorted(deploy_files.keys()), steps, "deploy")
        for step in selected:
            path = deploy_files[step]
            obs = _ensure_obs_shape(
                _parse_deploy_dump_obs_flat(path),
                total_in=total_in,
                label=f"deploy step {step}",
            )
            cases.append(
                {
                    "name": f"deploy_step{step}",
                    "source": "deploy",
                    "step": step,
                    "input_file": str(path),
                    "obs_flat": obs,
                }
            )

    if input_source in {"train", "both"}:
        if train_dir is None:
            raise ValueError("--train-dir is required when input source includes train.")
        train_candidates = _discover_files(train_dir, "debug_play_step*.npz")
        train_files = _index_by_step(train_candidates)
        if not train_files:
            raise FileNotFoundError(f"No train dumps found in: {train_dir}")
        selected = _require_steps(sorted(train_files.keys()), steps, "train")
        for step in selected:
            path = train_files[step]
            obs = _ensure_obs_shape(
                _load_train_dump_obs_flat(path),
                total_in=total_in,
                label=f"train step {step}",
            )
            cases.append(
                {
                    "name": f"train_step{step}",
                    "source": "train",
                    "step": step,
                    "input_file": str(path),
                    "obs_flat": obs,
                }
            )

    if not cases:
        raise RuntimeError("No input cases were built.")
    cases.sort(key=lambda x: (int(x["step"]), str(x["source"])))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ONNX and PT policy intermediate activations.")
    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX), help="Path to policy.onnx")
    parser.add_argument("--pt-path", default=str(DEFAULT_PT), help="Path to policy checkpoint .pt")
    parser.add_argument("--num-obs", type=int, default=117, help="Single-frame observation dimension")
    parser.add_argument("--history-len", type=int, default=40, help="Number of history frames")
    parser.add_argument(
        "--input-source",
        choices=["dummy", "deploy", "train", "both"],
        default="dummy",
        help="Source of obs_flat inputs for inference.",
    )
    parser.add_argument(
        "--deploy-dir",
        default=str(DEFAULT_DEPLOY_DUMP_DIR),
        help="Directory or file for deploy dumps (debug_cpp_step*.txt).",
    )
    parser.add_argument(
        "--train-dir",
        default=str(DEFAULT_TRAIN_DUMP_DIR),
        help="Directory or file for training dumps (debug_play_step*.npz).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated steps/ranges (e.g. 0,1,2,3,4 or 0-4). Use 'all' for all available.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Dummy input random seed (dummy mode only)")
    parser.add_argument("--dummy-scale", type=float, default=1.0, help="Stddev of dummy input (dummy mode only)")
    parser.add_argument("--max-plot-points", type=int, default=128, help="Max values plotted per layer")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory to save reports and plots")
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path).expanduser().resolve()
    pt_path = Path(args.pt_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = out_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not pt_path.is_file():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    deploy_policy, n_actions, num_obs_history = _build_deploy_policy_from_ckpt(
        pt_path=pt_path,
        num_obs=args.num_obs,
        history_len=args.history_len,
    )
    total_in = args.num_obs + num_obs_history

    steps = _parse_steps_arg(args.steps)
    deploy_dir = Path(args.deploy_dir).expanduser().resolve() if args.deploy_dir else None
    train_dir = Path(args.train_dir).expanduser().resolve() if args.train_dir else None
    input_cases = _build_input_cases(
        input_source=args.input_source,
        total_in=total_in,
        seed=args.seed,
        dummy_scale=args.dummy_scale,
        deploy_dir=deploy_dir,
        train_dir=train_dir,
        steps=steps,
    )

    onnx_model, onnx_sess, onnx_output_names = _create_onnx_session_all_outputs(onnx_path)

    case_reports: List[Dict[str, object]] = []
    pt_case_layers: Dict[str, OrderedDict[str, np.ndarray]] = {}
    onnx_case_layers: Dict[str, OrderedDict[str, np.ndarray]] = {}

    for case in input_cases:
        case_name = str(case["name"])
        case_id = _sanitize_case_id(case_name)
        source = str(case["source"])
        step = int(case["step"])
        obs_flat = np.asarray(case["obs_flat"], dtype=np.float32)

        pt_layers = _capture_pt_intermediates(deploy_policy, obs_flat)
        raw_onnx_outputs = _run_onnx_all_outputs(onnx_sess, onnx_output_names, obs_flat)
        onnx_layers = _alias_onnx_intermediates(onnx_model, raw_onnx_outputs)
        common_layers, rows = _compare_layers(pt_layers, onnx_layers)

        traces_png = per_case_dir / f"layer_traces_{case_id}.png"
        diff_png = per_case_dir / f"layer_diffs_{case_id}.png"
        _plot_layer_traces(
            traces_png,
            common_layers,
            pt_layers,
            onnx_layers,
            args.max_plot_points,
            title_prefix=f"{case_name}",
        )
        _plot_diff_bars(diff_png, rows, title_prefix=case_name)

        case_reports.append(
            {
                "case": case_name,
                "source": source,
                "step": step,
                "input_file": case["input_file"],
                "input_shape": list(obs_flat.shape),
                "common_layers": common_layers,
                "rows": rows,
                "artifacts": {
                    "layer_traces_png": str(traces_png),
                    "layer_diffs_png": str(diff_png),
                },
            }
        )
        pt_case_layers[case_name] = pt_layers
        onnx_case_layers[case_name] = onnx_layers

    summary_json = out_dir / "comparison_summary.json"
    summary_csv = out_dir / "comparison_summary.csv"
    pt_npz = out_dir / "pt_intermediates_by_case.npz"
    onnx_npz = out_dir / "onnx_intermediates_by_case.npz"
    inputs_npz = out_dir / "inputs_by_case.npz"
    heatmap_max_abs = out_dir / "layer_heatmap_max_abs.png"
    heatmap_nrmse = out_dir / "layer_heatmap_nrmse_rms.png"

    _write_case_csv(summary_csv, case_reports)
    _write_case_layers_npz(pt_npz, pt_case_layers)
    _write_case_layers_npz(onnx_npz, onnx_case_layers)
    _write_case_inputs_npz(inputs_npz, input_cases)
    _plot_metric_heatmap(
        heatmap_max_abs,
        case_reports,
        metric_key="max_abs_diff",
        title="PT vs ONNX max_abs_diff by layer and case",
        log10_scale=True,
    )
    _plot_metric_heatmap(
        heatmap_nrmse,
        case_reports,
        metric_key="nrmse_rms",
        title="PT vs ONNX nrmse_rms by layer and case",
        log10_scale=False,
    )

    worst_overall = None
    for case in case_reports:
        valid_rows = [r for r in case["rows"] if r["shape_match"]]  # type: ignore[index]
        if not valid_rows:
            continue
        worst_case_row = max(valid_rows, key=lambda r: float(r["max_abs_diff"]))
        if worst_overall is None or float(worst_case_row["max_abs_diff"]) > float(worst_overall["max_abs_diff"]):
            worst_overall = {
                "case": case["case"],
                "layer": worst_case_row["layer"],
                "max_abs_diff": worst_case_row["max_abs_diff"],
                "mean_abs_diff": worst_case_row["mean_abs_diff"],
                "nrmse_rms": worst_case_row.get("nrmse_rms"),
                "rel_l2": worst_case_row.get("rel_l2"),
            }

    payload = {
        "inputs": {
            "onnx_path": str(onnx_path),
            "pt_path": str(pt_path),
            "num_obs": args.num_obs,
            "history_len": args.history_len,
            "num_obs_history": num_obs_history,
            "total_input_dim": total_in,
            "num_actions": n_actions,
            "input_source": args.input_source,
            "deploy_dir": str(deploy_dir) if deploy_dir is not None else None,
            "train_dir": str(train_dir) if train_dir is not None else None,
            "steps": steps,
            "seed": args.seed,
            "dummy_scale": args.dummy_scale,
        },
        "cases": case_reports,
        "aggregate": {
            "num_cases": len(case_reports),
            "worst_overall": worst_overall,
        },
        "artifacts": {
            "summary_csv": str(summary_csv),
            "pt_npz": str(pt_npz),
            "onnx_npz": str(onnx_npz),
            "inputs_npz": str(inputs_npz),
            "heatmap_max_abs": str(heatmap_max_abs),
            "heatmap_nrmse_rms": str(heatmap_nrmse),
            "per_case_dir": str(per_case_dir),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] Compared {len(case_reports)} input case(s).")
    for case in case_reports:
        valid_rows = [r for r in case["rows"] if r["shape_match"]]  # type: ignore[index]
        if not valid_rows:
            print(f"[WARN] {case['case']}: no matching layers between PT and ONNX aliases.")
            continue
        worst = max(valid_rows, key=lambda r: float(r["max_abs_diff"]))
        print(
            "[INFO] "
            f"{case['case']}: "
            f"worst={worst['layer']} "
            f"max_abs={float(worst['max_abs_diff']):.6e} "
            f"mean_abs={float(worst['mean_abs_diff']):.6e} "
            f"nrmse_rms={float(worst['nrmse_rms']):.6e} "
            f"rel_l2={float(worst['rel_l2']):.6e}"
        )

    print(f"[INFO] Summary JSON: {summary_json}")
    print(f"[INFO] Summary CSV:  {summary_csv}")
    print(f"[INFO] PT NPZ:       {pt_npz}")
    print(f"[INFO] ONNX NPZ:     {onnx_npz}")
    print(f"[INFO] Inputs NPZ:   {inputs_npz}")
    print(f"[INFO] Heatmap max:  {heatmap_max_abs}")
    print(f"[INFO] Heatmap nrmse:{heatmap_nrmse}")
    print(f"[INFO] Per-case png: {per_case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
