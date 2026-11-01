# Sim2real target process (Isaac Sim)

A working version of the [Synthetic Data Drone Study](../STUDY.md) pipeline. A drone flies from a start point to a goal **70–150 m away at 5 m above the local ground**, past buildings, trees and power lines. Every flight is randomized, flown by an expert pilot and logged as synthetic training data. An evaluator then measures success rates on layouts, campuses and conditions the training data never covered.

There are two worlds and two drones, picked per split in [config.py](config.py):

| | World | Drone |
|---|---|---|
| Stage 1 | procedural 260 m campus ([campus.py](campus.py)) | Crazyflie `cf2x`, from NVIDIA's asset server |
| Stage 2 (the study's setup) | **NUST H-12 / UET Taxila digital twins** ([twin.py](twin.py)) | **AK hybrid drone**, 1.5 kg, built from AyroX's Gazebo model |

Built for **Isaac Sim 6.0.1** (standalone, `~/Documents/isaacsim`). Stage 1 needs network access for the Crazyflie USD. Stage 2 needs the `ak_hybrid_drone` Gazebo package, converted once (see [Campus twins](#campus-twins-stage-2)).

## Quick start

```bash
cd sim2real_target

# watch it fly (Isaac Sim window, chase camera)
~/Documents/isaacsim/python.sh generate_dataset.py --gui --split train_varied --flights 5
~/Documents/isaacsim/python.sh generate_dataset.py --gui --split nust_train_varied --flights 3   # twin, after building assets/

# generate training data (headless)
~/Documents/isaacsim/python.sh generate_dataset.py --split train_clean  --flights 50
~/Documents/isaacsim/python.sh generate_dataset.py --split train_varied --flights 3000

# measure success rate on a test split
~/Documents/isaacsim/python.sh evaluate.py --split test_new_layouts --flights 200 --policy expert
```

Each Isaac Sim process uses about 5–6 GB of RAM (one grew to 10.7 GB) and several CPU cores. On a 31 GB machine, run at most two or three at once; more caused out-of-memory kills. After about 85 flights in one process, the RTX renderer fails with `Out of resource descriptors` and hangs, so for long runs use [run_sharded.sh](run_sharded.sh), which splits a split across processes and restarts Isaac every 40 flights. Headless runs at about 1.6× real time on the procedural campus and about 2.4× on the twins.

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

The study's setup runs the same five roles on the twins with the AK hybrid drone. It trains on NUST only; UET Taxila is a different real campus, used only for testing.

| Split | World | Mode | Mirrors the study's… |
|---|---|---|---|
| `nust_train_clean` | NUST H-12 | clean | "clean flights" |
| `nust_train_varied` | NUST H-12 | varied | "varied flights" |
| `nust_test_new_layouts` | NUST H-12 | varied | "NUST, new layouts" |
| `taxila_test_unseen` | UET Taxila | varied | "UET Taxila, never seen" |
| `nust_test_harsh` | NUST H-12 | harsh | "NUST, harsher than training" |

On a twin, start/goal pairs are sampled anywhere in the modelled area at least 150 m from the edge. A pair is kept only when at least 3% of the cells within 15 m of the straight line are built up, which keeps flights in campus-like areas rather than open fields.

### Outcomes

A flight ends in one of these outcomes:

| Outcome | Condition |
|---|---|
| `success` | within 2 m of the goal |
| `crash_obstacle` | closer than 25 cm to any obstacle surface (procedural: analytic geometry; twins: a PhysX overlap sphere against the real meshes) |
| `crash_ground` | less than 0.3 m above the local ground, or touching the terrain mesh |
| `out_of_bounds` | left the campus |
| `timeout` | ran past 2.5 × the straight-line time at cruise speed + 15 s |

## Campus twins (stage 2)

The twins come from AyroX's `ak_hybrid_drone` ROS 2 / Gazebo package (`worlds/nust_campus_hybrid_drone.sdf`, `worlds/uet_taxila_campus_hybrid_drone.sdf`, `models/`). Two one-off steps turn them into what the env loads. Their output goes to `assets/`, which is not committed (about 220 MB):

```bash
~/Documents/isaacsim/python.sh tools/build_twins.py --src ~/Documents/ak_hybrid_drone_src/ak_hybrid_drone
python3 tools/build_maps.py
```

| Step | Output | What happens |
|---|---|---|
| [tools/build_twins.py](tools/build_twins.py) | `assets/twins/<campus>.usd`, `assets/ak_hybrid_drone.usd` | Every DAE tile of the terrain, buildings, roads, trees, power lines and far terrain goes through `omni.kit.asset_converter`. The converter tags its output Y-up on Z-up geometry, so the tag is corrected. The tiles are assembled into one stage per campus, with static triangle-mesh colliders (no approximation) on the terrain, buildings, trees and power lines. The drone becomes one rigid body: mass and diagonal inertia from `hybrid_drone_backtrack/model.sdf` (the drone placed in the NUST world; `ak_hybrid_quadron` has zero inertia), frame and props from its meshes, and a box collider around the frame. |
| [tools/build_maps.py](tools/build_maps.py) | `assets/twins/<campus>_map.npz` | The world-space triangles are sampled at 0.5 m into a 1 m grid holding the terrain height and the lowest and highest obstacle surface per cell. NUST takes 9 M cells and about 30 s. |

At run time the twin is referenced at `/World/Twin`. The env places clutter, the drone, the goal marker and the altitude target relative to the terrain height under them. The expert plans A* on a window of the map around the route, where a cell is blocked when an obstacle spans 5 m ± 1.5 m above the local ground. The window is inflated by 1.6 m, the same safety margin as on the procedural campus.

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
| [twin.py](twin.py) | twin map queries, start/goal and clutter sampling on a twin, twin flight planning (pure numpy) |
| [run_twin_study.sh](run_twin_study.sh) | twin datasets + expert evaluation on NUST / UET Taxila |
| [run_sharded.sh](run_sharded.sh) | one split as parallel processes on separate flight ranges, Isaac restarted every 40 flights |
| [tools/gpu_benchmark.py](tools/gpu_benchmark.py) | GPU PhysX + Warp raycast benchmark for N drones at once |
| [tools/build_twins.py](tools/build_twins.py) | Gazebo package → Isaac Sim USD: campus twins and the AK hybrid drone |
| [tools/build_maps.py](tools/build_maps.py) | twin triangles → 1 m planning map |

## Limits of this version

- **Simplified plant.** The drone is driven as one rigid body by collective thrust and body torques. Propeller aerodynamics and per-motor saturation are not modelled, and wind is linear drag on relative air speed.
- **Fixed heading.** The expert flies at yaw 0, so world-frame and body-frame xy are the same.
- **Planner map resolution.** On the twins the expert plans on a 1 m grid, while sensing and crashes use the full meshes, so the expert keeps a 1.6 m margin plus up to one cell. Overhead power lines block the plane when they hang within 1.5 m of flight altitude; the expert never flies under or over them.
- **Hybrid drone as a rigid-body quadrotor.** The AK hybrid drone is flown as one rigid body under collective thrust and body torques. The Gazebo model's lift-drag and per-joint propeller plugins and its ArduPilot interface are not carried over.
- **No policy training yet.** The next steps are behaviour cloning on `train_clean` and `train_varied`, then DAgger "corrections" using the logged `expert_action`. That needs `torch` installed into Isaac Sim's Python.
