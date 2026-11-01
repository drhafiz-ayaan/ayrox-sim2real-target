"""Measure a policy's success rate on a test split (layouts it never trained on).

    ~/Documents/isaacsim/python.sh evaluate.py --split test_new_layouts --flights 200 --policy expert
    ~/Documents/isaacsim/python.sh evaluate.py --split test_unseen_campus --policy my_policy.py:Student

A student policy class needs `reset(env)` and `act(out, env) -> [vx, vy, vz, yaw_rate]`, and must
use only `out["obs"]` (onboard sensing). `TargetFlightEnv.flatten(out["obs"])` gives a 23-d vector.
"""

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import common_args, launch

ap = common_args(__doc__)
ap.add_argument("--policy", default="expert")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
args, _ = ap.parse_known_args()

app = launch(headless=not args.gui)

from env import TargetFlightEnv  # noqa: E402
from rollout import fly, load_policy  # noqa: E402

import config as cfg  # noqa: E402

split = cfg.SPLITS[args.split]
env = TargetFlightEnv(app, headless=not args.gui, world=split.world, drone=split.drone)
policy = load_policy(args.policy)
results = []
for i in range(args.first, args.first + args.flights):
    r, _ = fly(env, policy, args.split, i)
    results.append(r)
    print(f"[{args.split} {i}] {r['outcome']:<15} {r['flight_time_s']:6.1f}s  goal dist {r['final_goal_dist_m']:6.1f} m", flush=True)

counts = collections.Counter(r["outcome"] for r in results)
rate = counts["success"] / len(results)
summary = {"split": args.split, "policy": args.policy, "flights": len(results), "success_rate": round(rate, 3), "outcomes": dict(counts)}
os.makedirs(args.out, exist_ok=True)
name = f"{args.split}__{os.path.basename(args.policy).replace(':', '_').replace('.py', '')}.json"
with open(os.path.join(args.out, name), "w") as f:
    json.dump({"summary": summary, "flights": results}, f, indent=1)
print(json.dumps(summary, indent=1))
app.close()
