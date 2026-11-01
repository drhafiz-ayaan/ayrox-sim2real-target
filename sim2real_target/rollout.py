"""Shared helpers: launching Isaac Sim, policies, and flying one flight."""

from __future__ import annotations

import argparse
import importlib.util
import time

import numpy as np

import config as cfg


def launch(headless: bool):
    from isaacsim import SimulationApp

    return SimulationApp({"headless": headless})


def common_args(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--split", default="train_varied", choices=list(cfg.SPLITS))
    ap.add_argument("--flights", type=int, default=10)
    ap.add_argument("--first", type=int, default=0, help="index of the first flight in the split")
    ap.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    return ap


class ExpertPolicy:
    """Privileged teacher: uses the env's planned path and the true state."""

    name = "expert"

    def reset(self, env):
        pass

    def act(self, out: dict, env) -> np.ndarray:
        return env.expert.act(out["true"]["pos"], out["true"]["vel"])


def load_policy(spec: str):
    """'expert' or 'path/to/file.py:ClassName'. A student class gets only `out['obs']`."""
    if spec == "expert":
        return ExpertPolicy()
    path, cls = spec.split(":")
    mod_spec = importlib.util.spec_from_file_location("user_policy", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return getattr(mod, cls)()


def fly(env, policy, split_name: str, index: int, record: bool = False) -> tuple[dict, dict | None]:
    split = cfg.SPLITS[split_name]
    seed = split.seed_offset + index
    out = env.reset(split.campus_seed, split.mode, seed)
    policy.reset(env)
    rec = {k: [] for k in ("t", "goal_rel", "vel", "alt", "ranges", "action", "expert_action", "pos", "quat", "true_vel")}
    t0, done, outcome = time.time(), False, ""
    while not done:
        cmd = np.asarray(policy.act(out, env), float)
        if record:
            o, s = out["obs"], out["true"]
            rec["t"].append(env.t)
            for k in ("goal_rel", "vel", "alt", "ranges"):
                rec[k].append(o[k])
            rec["action"].append(cmd)
            # expert label at every visited state (for DAgger-style corrections when a student flies)
            rec["expert_action"].append(env.expert.act(s["pos"], s["vel"]))
            rec["pos"].append(s["pos"])
            rec["quat"].append(s["quat"])
            rec["true_vel"].append(s["vel"])
        out, done, outcome = env.step(cmd)

    straight = float(np.linalg.norm(env.goal - env.start))
    result = {
        "split": split_name,
        "index": index,
        "campus_seed": split.campus_seed,
        "mode": split.mode,
        "world": split.world,
        "drone": split.drone,
        "flight_seed": seed,
        "policy": getattr(policy, "name", type(policy).__name__),
        "outcome": outcome,
        "success": outcome == "success",
        "flight_time_s": round(env.t, 2),
        "path_flown_m": round(env.path_flown, 1),
        "straight_dist_m": round(straight, 1),
        "expert_path_m": round(env.expert.path_length, 1),
        "final_goal_dist_m": round(float(np.linalg.norm(out["true"]["pos"][:2] - env.goal)), 2),
        "wall_time_s": round(time.time() - t0, 1),
        "start": env.start.round(2).tolist(),
        "goal": env.goal.round(2).tolist(),
        "randomization": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in env.params.items()},
    }
    arrays = {k: np.asarray(v, np.float32) for k, v in rec.items()} if record else None
    return result, arrays
