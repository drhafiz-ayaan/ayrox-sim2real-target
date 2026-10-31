# Sim2real target process (Isaac Sim)

A small, working version of the [Synthetic Data Drone Study](../STUDY.md) pipeline. A Crazyflie flies from a start point to a goal **70–150 m away at 5 m above the ground**, past buildings, trees and power lines in a procedural campus. Every flight is randomized, flown by an expert pilot and logged as synthetic training data. An evaluator then measures success rates on layouts, campuses and conditions the training data never covered.

Built for **Isaac Sim 6.0.1** (standalone, `~/Documents/isaacsim`). It needs network access to NVIDIA's asset server for the Crazyflie USD.

## Quick start

```bash
cd sim2real_target

# watch it fly (Isaac Sim window, chase camera)
~/Documents/isaacsim/python.sh generate_dataset.py --gui --split train_varied --flights 5

# generate training data (headless)
~/Documents/isaacsim/python.sh generate_dataset.py --split train_clean  --flights 50
~/Documents/isaacsim/python.sh generate_dataset.py --split train_varied --flights 3000

# measure success rate on a test split
~/Documents/isaacsim/python.sh evaluate.py --split test_new_layouts --flights 200 --policy expert
```

Run one Isaac Sim process at a time. Each uses about 6 GB of RAM and several CPU cores, and running three at once got the GUI killed by the out-of-memory guard. Headless runs at about 1.6× real time, roughly 30 s of wall time per flight.

## How it works

```
 flight seed ──► layout + start/goal + wind + noise + drone response
                        │
        expert (privileged: true map + true state, A* + pure pursuit)
                        │  velocity setpoint [vx, vy, vz, yaw_rate] @ 10 Hz
                        ▼
        low-level controller (PI velocity → geometric attitude) @ 100 Hz
                        │  thrust + torques, through randomized plant
                        ▼
        Isaac Sim / PhysX: Crazyflie cf2x in the campus
                        │
        onboard sensing (noisy) ──► logged as the student's observation
```

### The sim2real interface

The policy only outputs **velocity setpoints**. The real Crazyflie takes the same command through `cflib`'s `commander.send_velocity_world_setpoint(vx, vy, vz, yawrate)`, and its firmware turns it into motor commands. A trained policy therefore plugs into the real drone without changes. The gap that remains (mass, thrust, motor lag, latency, wind, sensor noise) is what the randomization covers.

### Onboard observation (23 values, `TargetFlightEnv.flatten`)

| Field | Size | Source in sim | Real counterpart |
|---|---|---|---|
| `goal_rel` | 3 | goal − noisy position estimate | state estimate (Flow deck / Lighthouse / UWB) |
| `vel` | 3 | true velocity + noise | onboard velocity estimate |
| `alt` | 1 | downward PhysX raycast + noise | ToF range sensor |
| `ranges` | 16 | horizontal ring of PhysX raycasts, max 20 m, + noise | multi-ranger / ToF array |

This answers open question 4 in the study README ("which sensors?") for this version: a position estimate, velocity, a height sensor and a 16-beam horizontal range ring. To match real hardware, change `NUM_RAYS` in [config.py](config.py), for example to 4 for the Crazyflie Multi-ranger deck.

### What is randomized per flight

Ranges are in [config.py](config.py), and everything is drawn from the flight seed, so any flight can be replayed exactly.

| Parameter | `clean` | `varied` (training) | `harsh` (stress test) |
|---|---|---|---|
| Extra obstacles near the route | 0 | 8–28 | 28–40 |
| Mean wind | 0 | 0.2–1.1 m/s | 1.1–1.8 m/s |
| Gusts (Ornstein–Uhlenbeck) | 0 | 0.05–0.3 m/s | 0.3–0.5 m/s |
| Sensor noise (std) | 0 | 4–11 cm | 11–16 cm |
| Mass scale | 1.0 | 0.9–1.1 | 0.85–1.15 |
| Thrust gain | 1.0 | 0.88–1.12 | 0.82–1.18 |
| Motor lag | 25 ms | 15–40 ms | 40–60 ms |
| Command latency | 0 | 0–2 periods (0–200 ms) | 1–3 periods |

The start and goal are random in every mode.

### Splits

The splits never share a flight seed.

| Split | Campus | Mode | Mirrors the study's… |
|---|---|---|---|
| `train_clean` | A | clean | "clean flights" |
| `train_varied` | A | varied | "varied flights" |
| `test_new_layouts` | A | varied | "NUST, new layouts" |
| `test_unseen_campus` | B | varied | "UET Taxila, never seen" |
| `test_harsh` | A | harsh | "NUST, harsher than training" |

Campus A and campus B are each about 260 m × 260 m, with 28 buildings, about 45 trees and 2 power lines, generated from different seeds.

### Outcomes

A flight ends in one of these outcomes:

| Outcome | Condition |
|---|---|
| `success` | within 2 m of the goal |
| `crash_obstacle` | closer than 25 cm to any obstacle surface |
| `crash_ground` | below 0.3 m |
| `out_of_bounds` | left the campus |
| `timeout` | ran past 2.5 × the straight-line time at cruise speed + 15 s |

## Dataset format

`data/<split>/flight_<index>.npz` holds one flight, logged at 10 Hz:

| Key | Shape | Meaning |
|---|---|---|
| `goal_rel`, `vel`, `alt`, `ranges` | (T, …) | onboard observation (what a student policy sees) |
| `action` | (T, 4) | the command that was flown |
| `expert_action` | (T, 4) | the expert's label at the same state (for DAgger-style "corrections") |
| `pos`, `quat`, `true_vel` | (T, …) | ground-truth state, for analysis only |
| `t` | (T,) | time, in seconds |

`data/<split>/index.jsonl` has one line per flight, with the outcome, flight time, distances and every randomization value that was drawn. By default only successful expert flights are saved; pass `--keep-failed` to keep the rest. A re-run skips flights that already exist, so long runs can be stopped and resumed.

## Plugging in a student policy

```python
# my_policy.py
import numpy as np
from env import TargetFlightEnv

class Student:
    name = "bc_v1"
    def reset(self, env): ...
    def act(self, out, env):
        x = TargetFlightEnv.flatten(out["obs"])   # 23-d onboard observation only
        return np.array([vx, vy, vz, 0.0])        # velocity setpoint
```

```bash
~/Documents/isaacsim/python.sh evaluate.py --split test_unseen_campus --flights 200 --policy my_policy.py:Student
```

A student must use only `out["obs"]`. `out["true"]` and `env.expert` are privileged information. Results are written to `results/<split>__<policy>.json`.

## Files

| File | Purpose |
|---|---|
| [config.py](config.py) | task numbers, randomization ranges, splits |
| [campus.py](campus.py) | procedural layout sampling and geometry (pure numpy) |
| [expert.py](expert.py) | A* planner on an inflated 2D grid + pure-pursuit velocity commands |
| [controller.py](controller.py) | velocity → thrust and torque flight stack |
| [env.py](env.py) | Isaac Sim world, Crazyflie, randomized plant, wind, sensors, outcomes |
| [rollout.py](rollout.py) | app launch, policy loading, flying one flight |
| [generate_dataset.py](generate_dataset.py) | expert flights → dataset |
| [evaluate.py](evaluate.py) | success rate of a policy on a split |
| [make_dataset.sh](make_dataset.sh) | datasets + expert evaluation in one go (`CLEAN`, `VARIED`, `EVAL` env vars) |
| [capture_showcase.py](capture_showcase.py) | Isaac Sim screenshots for the docs → `docs/images/` |
| [make_figures.py](make_figures.py) | result figures from `data/` and `results/` → `docs/figures/` (plain Python + matplotlib) |

## Limits of this version

- **Simplified plant.** The drone is driven as one rigid body by collective thrust and body torques. Propeller aerodynamics and per-motor saturation are not modelled, and wind is linear drag on relative air speed.
- **Fixed heading.** The expert flies at yaw 0, so world-frame and body-frame xy are the same.
- **Flat world.** The campus is flat ground with box, cylinder and cable primitives. Switching to the NUST H-12 twin means loading its USD in place of `_set_campus` and providing an occupancy map for the expert. The PhysX raycast sensors work on any USD collider as they are.
- **No policy training yet.** The next steps are behaviour cloning on `train_clean` and `train_varied`, then DAgger "corrections" using the logged `expert_action`. That needs `torch` installed into Isaac Sim's Python.
