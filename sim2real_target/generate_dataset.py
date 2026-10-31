"""Fly expert flights and log them as a synthetic dataset.

    ~/Documents/isaacsim/python.sh generate_dataset.py --split train_varied --flights 100

Each flight -> data/<split>/flight_<index>.npz (10 Hz onboard observations, actions, true state),
plus one line in data/<split>/index.jsonl with the outcome and the randomization drawn.
Re-running skips flights that already exist, so long runs can be resumed.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import common_args, launch

ap = common_args(__doc__)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--keep-failed", action="store_true", help="also save flights where the expert did not reach the goal")
args, _ = ap.parse_known_args()

app = launch(headless=not args.gui)

import numpy as np  # noqa: E402

from env import TargetFlightEnv  # noqa: E402
from rollout import ExpertPolicy, fly  # noqa: E402

out_dir = os.path.join(args.out, args.split)
os.makedirs(out_dir, exist_ok=True)
env = TargetFlightEnv(app, headless=not args.gui)
policy = ExpertPolicy()
n_ok = 0
for i in range(args.first, args.first + args.flights):
    path = os.path.join(out_dir, f"flight_{i:06d}.npz")
    if os.path.exists(path):
        continue
    result, arrays = fly(env, policy, args.split, i, record=True)
    if result["success"] or args.keep_failed:
        np.savez_compressed(path, **arrays)
        result["file"] = os.path.basename(path)
    with open(os.path.join(out_dir, "index.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    n_ok += result["success"]
    print(f"[{args.split} {i}] {result['outcome']:<15} {result['flight_time_s']:6.1f}s sim  {result['path_flown_m']:6.1f} m  ({result['wall_time_s']}s wall)", flush=True)

print(f"done: {n_ok}/{args.flights} expert flights reached the goal -> {out_dir}")
app.close()
