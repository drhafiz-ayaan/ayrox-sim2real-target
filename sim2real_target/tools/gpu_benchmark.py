"""Can this machine fly many drones at once? GPU PhysX + GPU raycast benchmark.

    ~/Documents/isaacsim/python.sh tools/gpu_benchmark.py --counts 1,16,64,256

Per drone count N: the NUST twin is loaded once, N AK hybrid drones hover 5 m above the terrain
(spread out, so they never touch), and PhysX steps on the GPU with batched force application
and a state readback every step, as a vectorized flight controller would need. Then the
onboard sensor rays (16 horizontal + 1 down per drone) are cast against the twin's triangles
with Warp on the GPU. Prints steps/s, drone-steps/s and GPU memory.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg  # noqa: E402  (before SimulationApp, see capture_showcase.py)

ap = argparse.ArgumentParser()
ap.add_argument("--counts", default="1,16,64,256")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--world", default="nust")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from twin import TwinMap  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda:0"


def gpu_mem_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, mem = (s.strip() for s in line.split(","))
        if int(pid) == os.getpid():
            return int(mem)
    return -1


def spawn_points(tm: TwinMap, n: int, spacing: float = 40.0) -> np.ndarray:
    """n free spots on a grid around the campus centre, 5 m above the terrain."""
    pts, side = [], int(np.ceil(np.sqrt(n * 2)))
    for i in range(side):
        for j in range(side):
            xy = np.array([-0.5 * side * spacing + i * spacing, -0.5 * side * spacing + j * spacing])
            if tm.inside(xy, 50) and tm.clearance(xy) > 3.0:
                pts.append([xy[0], xy[1], tm.ground_at(xy) + cfg.FLIGHT_ALT])
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError(f"only found {len(pts)} free spawn points")


def bench_physics(tm: TwinMap, n: int) -> dict:
    stage_utils.create_new_stage()
    SimulationManager.setup_simulation(dt=1.0 / cfg.PHYSICS_HZ, device=DEVICE)
    stage_utils.add_reference_to_stage(os.path.join(HERE, "assets", "twins", f"{args.world}.usd"), "/World/Twin")
    paths = [f"/World/drone_{i}" for i in range(n)]
    usd = os.path.join(HERE, cfg.DRONES["ak_hybrid"]["usd"])
    for p in paths:
        stage_utils.add_reference_to_stage(usd, p)
    app.update()
    roots = XformPrim(paths)
    roots.reset_xform_op_properties()
    roots.set_world_poses(positions=spawn_points(tm, n), orientations=np.tile([1.0, 0, 0, 0], (n, 1)))

    t0 = time.time()
    app_utils.play()
    app.update()
    setup_s = time.time() - t0
    view = RigidPrim(paths)._physics_rigid_body_view
    mass = float(RigidPrim(paths[:1]).get_masses().numpy().reshape(-1)[0])
    forces = wp.array(np.tile([0.0, 0.0, mass * 9.81], (n, 1)).astype(np.float32), device=DEVICE)
    torques = wp.zeros((n, 3), dtype=wp.float32, device=DEVICE)
    idx = wp.array(np.arange(n, dtype=np.int32), device=DEVICE)

    def step():
        tf = view.get_transforms()  # what a batched controller reads every physics step
        vel = view.get_velocities()
        view.apply_forces_and_torques_at_position(forces, torques, None, idx, True)
        SimulationManager.step()
        return tf, vel

    for _ in range(20):  # warm-up (kernel compilation, first contacts)
        step()
    wp.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        tf, vel = step()
    wp.synchronize()
    dt = time.time() - t0
    z = tf.numpy()[:, 2]
    app_utils.stop()
    return {
        "setup_s": setup_s,
        "steps_per_s": args.steps / dt,
        "drone_steps_per_s": n * args.steps / dt,
        "realtime_x": (args.steps / cfg.PHYSICS_HZ) / dt,
        "gpu_mem_mb": gpu_mem_mb(),
        "hover_drift_m": float(np.abs(z - spawn_points(tm, n)[:, 2]).max()),
    }


@wp.kernel
def cast_rays(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3), max_d: float, out: wp.array2d(dtype=float)):
    i, k = wp.tid()
    q = wp.mesh_query_ray(mesh, origins[i] + dirs[k] * 0.31, dirs[k], max_d)
    out[i, k] = wp.where(q.result, q.t + 0.31, max_d)


def bench_rays(tm: TwinMap, counts) -> dict:
    tris = np.load(os.path.join(HERE, "assets", "twins", f"{args.world}_tris.npz"))
    t = np.concatenate([tris["terrain"], tris["obstacles"]]).reshape(-1, 3)
    t0 = time.time()
    mesh = wp.Mesh(points=wp.array(t, dtype=wp.vec3, device=DEVICE), indices=wp.array(np.arange(len(t), dtype=np.int32), device=DEVICE))
    build_s = time.time() - t0
    ang = np.linspace(0, 2 * np.pi, cfg.NUM_RAYS, endpoint=False)
    dirs = np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)].tolist() + [[0.0, 0.0, -1.0]]
    d_dirs = wp.array(np.array(dirs, np.float32), dtype=wp.vec3, device=DEVICE)
    res = {"mesh_triangles": len(t) // 3, "bvh_build_s": build_s}
    for n in counts:
        o = wp.array(spawn_points(tm, n).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        out = wp.zeros((n, len(dirs)), dtype=float, device=DEVICE)
        for _ in range(5):
            wp.launch(cast_rays, dim=(n, len(dirs)), inputs=[mesh.id, o, d_dirs, cfg.RAY_MAX, out], device=DEVICE)
        wp.synchronize()
        t0 = time.time()
        reps = 200
        for _ in range(reps):
            wp.launch(cast_rays, dim=(n, len(dirs)), inputs=[mesh.id, o, d_dirs, cfg.RAY_MAX, out], device=DEVICE)
        wp.synchronize()
        res[n] = {"ms_per_sense": 1000 * (time.time() - t0) / reps, "height_reading_m": float(np.median(out.numpy()[:, -1]))}
    return res


tm = TwinMap(args.world)
counts = [int(c) for c in args.counts.split(",")]
print("GPU", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
for n in counts:
    r = bench_physics(tm, n)
    print(f"PHYSICS n={n:4d}: {r['steps_per_s']:7.1f} steps/s  {r['drone_steps_per_s']:9.0f} drone-steps/s  "
          f"{r['realtime_x']:6.2f}x realtime per drone  setup {r['setup_s']:.1f}s  GPU mem {r['gpu_mem_mb']} MB  hover drift {r['hover_drift_m']:.2f} m", flush=True)
rays = bench_rays(tm, counts)
print(f"RAYS mesh {rays['mesh_triangles']} triangles, BVH build {rays['bvh_build_s']:.2f}s", flush=True)
for n in counts:
    print(f"RAYS n={n:4d}: {rays[n]['ms_per_sense']:.3f} ms for {n * 17} rays  (median height reading {rays[n]['height_reading_m']:.2f} m)", flush=True)
app.close()
