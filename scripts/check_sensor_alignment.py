#!/usr/bin/env python3
"""Compare deploy debug dumps between robot and sim for sensor convention alignment.

Usage:
  python3 Lite3_rl_deploy/scripts/check_sensor_alignment.py \
      --robot-dir /path/to/robot_dumps \
      --sim-dir /path/to/sim_dumps
"""

from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


FIELD_DIMS: Dict[str, int] = {
    "base_rpy": 3,
    "body_omega": 3,
    "base_acc": 3,
    "joint_pos_policy": 12,
    "joint_vel_policy": 12,
}


@dataclass
class DumpSeries:
    name: str
    steps: List[int]
    data: Dict[str, np.ndarray]


def _step_key(path: str) -> int:
    m = re.search(r"step(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def parse_dump_file(path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            vals: List[float] = []
            for tok in parts[1:]:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if vals:
                out[key] = np.asarray(vals, dtype=np.float64)
    return out


def load_series(name: str, dump_dir: str, max_steps: int) -> DumpSeries:
    files = sorted(glob.glob(os.path.join(dump_dir, "debug_cpp_step*.txt")), key=_step_key)
    if max_steps > 0:
        files = files[:max_steps]
    steps = [_step_key(p) for p in files]

    rows: Dict[str, List[np.ndarray]] = {k: [] for k in FIELD_DIMS}
    for p in files:
        d = parse_dump_file(p)
        for key, dim in FIELD_DIMS.items():
            if key in d and d[key].shape[0] == dim:
                rows[key].append(d[key])

    arrs: Dict[str, np.ndarray] = {}
    for key, seq in rows.items():
        arrs[key] = np.vstack(seq) if seq else np.empty((0, FIELD_DIMS[key]), dtype=np.float64)
    return DumpSeries(name=name, steps=steps, data=arrs)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _best_axis_mapping(a: np.ndarray, b: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...], float]:
    """Find best permutation + sign flip from b to match a by mean abs correlation."""
    dim = a.shape[1]
    best_score = -1.0
    best_perm: Tuple[int, ...] = tuple(range(dim))
    best_signs: Tuple[int, ...] = tuple([1] * dim)
    best_is_identity = True
    identity_perm = tuple(range(dim))
    identity_signs = tuple([1] * dim)
    for perm in itertools.permutations(range(dim)):
        bp = b[:, perm]
        for signs in itertools.product([-1, 1], repeat=dim):
            bs = bp * np.asarray(signs, dtype=np.float64)
            score = float(np.mean([abs(_corr(a[:, i], bs[:, i])) for i in range(dim)]))
            is_identity = (perm == identity_perm and tuple(int(s) for s in signs) == identity_signs)
            if score > best_score + 1e-9:
                best_score = score
                best_perm = perm
                best_signs = tuple(int(s) for s in signs)
                best_is_identity = is_identity
            elif abs(score - best_score) <= 1e-9 and is_identity and not best_is_identity:
                # In exact ties, prefer identity mapping to avoid false warnings.
                best_perm = perm
                best_signs = tuple(int(s) for s in signs)
                best_is_identity = True
    return best_perm, best_signs, best_score


def summarize_stream(s: DumpSeries) -> List[str]:
    out = [f"[{s.name}] steps={len(s.steps)}"]
    for key in ("base_rpy", "body_omega", "base_acc"):
        arr = s.data[key]
        if arr.shape[0] == 0:
            out.append(f"  - {key}: missing")
            continue
        rms = float(np.sqrt(np.mean(arr * arr)))
        mx = float(np.max(np.abs(arr)))
        out.append(f"  - {key}: rms={rms:.4f}, max_abs={mx:.4f}")
        if key == "base_acc":
            norms = np.linalg.norm(arr, axis=1)
            out.append(f"    acc_norm: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")
    return out


def compare_field(name: str, a: np.ndarray, b: np.ndarray) -> List[str]:
    out = [f"[{name}]"]
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        out.append("  - skipped (missing in one stream)")
        return out
    a = a[:n]
    b = b[:n]
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    mx = float(np.max(np.abs(diff)))
    rms_a = float(np.sqrt(np.mean(a * a)))
    rms_b = float(np.sqrt(np.mean(b * b)))
    ratio = rms_a / (rms_b + 1e-12)
    out.append(f"  - aligned_steps={n}, mean_abs_diff={mae:.6f}, max_abs_diff={mx:.6f}, rms_ratio(robot/sim)={ratio:.3f}")

    if 40.0 <= ratio <= 80.0 or 0.012 <= ratio <= 0.03:
        out.append("  - warning: scale ratio suggests a possible deg<->rad mismatch")

    dim = a.shape[1]
    corr = [abs(_corr(a[:, i], b[:, i])) for i in range(dim)]
    out.append(f"  - same-axis abs corr: mean={np.mean(corr):.3f}, min={np.min(corr):.3f}")
    if dim == 3:
        perm, signs, score = _best_axis_mapping(a, b)
        out.append(f"  - best axis mapping sim->robot: perm={perm}, signs={signs}, score={score:.3f}")
        if perm != (0, 1, 2) or signs != (1, 1, 1):
            out.append("  - warning: non-identity axis mapping fits better than direct mapping")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare robot vs sim sensor conventions from deploy debug dumps.")
    parser.add_argument("--robot-dir", required=True, help="Directory with robot debug_cpp_step*.txt files.")
    parser.add_argument("--sim-dir", required=True, help="Directory with sim debug_cpp_step*.txt files.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum number of steps to load from each side.")
    args = parser.parse_args()

    robot = load_series("robot", args.robot_dir, args.max_steps)
    sim = load_series("sim", args.sim_dir, args.max_steps)

    print("=== Stream Summary ===")
    for line in summarize_stream(robot):
        print(line)
    for line in summarize_stream(sim):
        print(line)

    print("\n=== Field Comparison ===")
    for field in ("base_rpy", "body_omega", "base_acc", "joint_pos_policy", "joint_vel_policy"):
        for line in compare_field(field, robot.data[field], sim.data[field]):
            print(line)

    print("\nNotes:")
    print("- This check validates conventions/sign/scale trends, not strict trajectory equality.")
    print("- For best results, capture both sides under the same script: zero command, same initial pose, first 100-200 policy steps.")


if __name__ == "__main__":
    main()
