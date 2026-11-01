"""Isaac Sim target-reaching environment: a drone in a randomized campus.

Create a `SimulationApp` first, then `TargetFlightEnv(app, world=..., drone=...)`. The world is the
procedural mini-campus or a campus twin (NUST H-12, UET Taxila); the drone is the Crazyflie or the
AyroX hybrid drone (see config.DRONES). One `reset()` = one flight: it samples the extra obstacles,
start/goal, wind, sensor noise and drone response from the flight seed, so any flight can be
replayed exactly from (campus_seed, mode, flight_seed).
"""

from __future__ import annotations

import os

import numpy as np

import config as cfg
from campus import Layout, base_campus, clearance
from controller import G, VelocityController, quat_to_rot
from expert import plan_flight

HERE = os.path.dirname(os.path.abspath(__file__))
WIND_DRAG = 0.35  # 1/s, linear drag per unit mass against relative air speed
RAY_DIRS = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, cfg.NUM_RAYS, endpoint=False)])


class TargetFlightEnv:
    def __init__(self, app, headless: bool = True, render_every: int = 1, world: str = "procedural", drone: str = "crazyflie"):
        # Omniverse imports only work after SimulationApp exists.
        import carb
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import DistantLight, DomeLight, GroundPlane
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.storage.native import get_assets_root_path
        from omni.physx import get_physx_scene_query_interface

        self.app, self.headless, self.render_every = app, headless, render_every
        self._carb = carb.settings.get_settings()
        self._app_utils, self._stage_utils, self._sm = app_utils, stage_utils, SimulationManager
        self._sq = get_physx_scene_query_interface()
        # The viewport's reference grid is drawn at z = 0 and shows through wherever twin terrain dips below it
        for key in ("/app/viewport/grid/enabled", "/persistent/app/viewport/grid/enabled",
                    "/app/viewport/Viewport/Viewport0/guide/grid/visible", "/persistent/app/viewport/Viewport/Viewport0/guide/grid/visible"):
            self._carb.set_bool(key, False)

        stage_utils.create_new_stage()
        SimulationManager.setup_simulation(dt=1.0 / cfg.PHYSICS_HZ, device="cpu")
        self.world, self.drone = world, cfg.DRONES[drone]
        if world == "procedural":
            GroundPlane("/World/Ground", sizes=2 * cfg.CAMPUS_HALF + 200, colors=[0.22, 0.3, 0.18], templates=None)
            self.tmap = None
        else:
            from twin import TwinMap

            stage_utils.add_reference_to_stage(os.path.join(HERE, "assets", "twins", f"{world}.usd"), "/World/Twin")
            self.tmap = TwinMap(world)
        DistantLight("/World/Sun").set_intensities(1800)
        sky = DomeLight("/World/Sky")
        sky.set_intensities(450)
        sky.set_colors([[0.62, 0.76, 0.95]])
        usd = self.drone["usd"]
        usd = get_assets_root_path() + usd[len("nucleus:") :] if usd.startswith("nucleus:") else os.path.join(HERE, usd)
        stage_utils.add_reference_to_stage(usd, self.drone["root"])
        app.update()
        from isaacsim.core.experimental.prims import XformPrim

        self.drone_root = XformPrim(self.drone["root"])
        self.drone_root.reset_xform_op_properties()

        self.campus_seed: int | None = None
        self.chase_offset = self.drone["chase"]  # m behind, m above the drone (GUI chase camera)
        self.base: Layout | None = None
        self.dt = 1.0 / cfg.PHYSICS_HZ
        self.substeps = cfg.PHYSICS_HZ // cfg.CONTROL_HZ

    # --- World building ---------------------------------------------------------

    def _delete(self, path: str):
        if self._stage_utils.get_current_stage().GetPrimAtPath(path).IsValid():
            self._stage_utils.delete_prim(path)

    def ground_at(self, xy) -> float:
        return self.tmap.ground_at(xy) if self.tmap else 0.0

    def _build_obstacles(self, root: str, layout: Layout, colors: tuple):
        """Boxes / cylinders / cables standing on the ground (terrain height under each one on a twin)."""
        from isaacsim.core.experimental.objects import Cube, Cylinder
        from isaacsim.core.experimental.prims import GeomPrim

        self._stage_utils.define_prim(root, "Xform")
        b, c, k = layout.boxes, layout.cylinders, layout.cables
        if len(b):
            zb = np.array([self.ground_at(p) for p in b[:, :2]])
            boxes = Cube(
                [f"{root}/box_{i}" for i in range(len(b))],
                sizes=1.0,
                positions=np.stack([b[:, 0], b[:, 1], zb + b[:, 4] / 2], 1),
                scales=b[:, [2, 3, 4]],
                colors=colors[0],
            )
            GeomPrim(boxes.paths, apply_collision_apis=True)
        if len(c):
            zc = np.array([self.ground_at(p) for p in c[:, :2]])
            cyl = Cylinder(
                [f"{root}/cyl_{i}" for i in range(len(c))],
                radii=c[:, 2],
                heights=c[:, 3],
                axes="Z",
                positions=np.stack([c[:, 0], c[:, 1], zc + c[:, 3] / 2], 1),
                colors=colors[1],
            )
            GeomPrim(cyl.paths, apply_collision_apis=True)
        if len(k):
            mid = np.stack([(k[:, 0] + k[:, 2]) / 2, (k[:, 1] + k[:, 3]) / 2, k[:, 4]], 1)
            length = np.hypot(k[:, 2] - k[:, 0], k[:, 3] - k[:, 1])
            yaw = np.arctan2(k[:, 3] - k[:, 1], k[:, 2] - k[:, 0])
            cables = Cube(
                [f"{root}/cable_{i}" for i in range(len(k))],
                sizes=1.0,
                positions=mid,
                orientations=np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1),
                scales=np.stack([length, 0.1 + 0 * length, 0.1 + 0 * length], 1),
                colors=colors[2],
            )
            GeomPrim(cables.paths, apply_collision_apis=True)

    def _set_campus(self, campus_seed: int):
        if campus_seed == self.campus_seed:
            return
        self._delete("/World/Campus")
        self.base = base_campus(campus_seed)
        self._build_obstacles("/World/Campus", self.base, ([0.62, 0.6, 0.58], [0.2, 0.45, 0.2], [0.1, 0.1, 0.1]))
        self.campus_seed = campus_seed

    # --- Episode --------------------------------------------------------------------

    def reset(self, campus_seed: int, mode: str, flight_seed: int) -> dict:
        from isaacsim.core.experimental.objects import Sphere
        from isaacsim.core.experimental.prims import RigidPrim

        # playSimulations must be on for stop/play to reset PhysX state between flights.
        self._carb.set_bool("/app/player/playSimulations", True)
        self._app_utils.stop()
        self.app.update()
        rnd = cfg.MODES[mode]
        rng = np.random.default_rng([campus_seed, flight_seed])
        if self.tmap is None:
            self._set_campus(campus_seed)
            start, goal, extra, n_extra, self.expert = plan_flight(self.base, rnd, rng)
            self.layout = self.base.merged(extra)
        else:
            from twin import plan_flight as plan_twin_flight

            start, goal, extra, n_extra, self.expert = plan_twin_flight(self.tmap, rnd, rng)
            self.layout = extra
        self.start, self.goal = start, goal
        z0, self.goal_z = self.ground_at(start) + cfg.FLIGHT_ALT, self.ground_at(goal) + cfg.FLIGHT_ALT

        self._delete("/World/Flight")
        self._build_obstacles("/World/Flight", extra, ([0.75, 0.45, 0.2], [0.3, 0.55, 0.25], [0.1, 0.1, 0.1]))
        Sphere("/World/Flight/goal_marker", radii=cfg.GOAL_RADIUS, positions=[[goal[0], goal[1], self.goal_z]], colors=[0.1, 0.9, 0.2])
        self.drone_root.set_world_poses(positions=[[start[0], start[1], z0]], orientations=[[1.0, 0, 0, 0]])

        # Play once with app-driven physics so everything initializes, then step physics ourselves.
        self._carb.set_bool("/app/player/playSimulations", True)
        self._app_utils.play()
        self.app.update()
        self._carb.set_bool("/app/player/playSimulations", False)

        self.body = RigidPrim(self.drone["body"])
        # The RigidPrim wrapper allocates warp arrays on every call; at physics rate we go straight
        # to its PhysX tensor view with preallocated CPU buffers instead.
        import warp as wp

        self._view = self.body._physics_rigid_body_view
        self._f_buf = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
        self._t_buf = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
        self._idx = wp.array([0], dtype=wp.int32, device="cpu")
        spawn_err = np.linalg.norm(self.true_state()["pos"][:2] - start)
        assert spawn_err < 0.5, f"drone did not respawn at start ({spawn_err:.2f} m off)"
        links = RigidPrim(self.drone["links"])
        masses = links.get_masses().numpy().reshape(-1)
        self.mass_nominal = float(masses.sum())
        J = np.diag(np.diag(self.body.get_inertias().numpy().reshape(3, 3)))

        # Domain randomization (all drawn from the flight seed).
        u = lambda r: float(rng.uniform(*r))
        wind_dir = rng.uniform(0, 2 * np.pi)
        self.params = {
            "n_extra_obstacles": n_extra,
            "wind_speed": u(rnd.wind_speed),
            "wind_dir": float(wind_dir),
            "gust_std": u(rnd.gust_std),
            "sensor_noise": u(rnd.sensor_noise),
            "mass_scale": u(rnd.mass_scale),
            "thrust_gain": u(rnd.thrust_gain),
            "motor_tau": u(rnd.motor_tau),
            "ctrl_latency_steps": int(rng.integers(rnd.ctrl_latency_steps[0], rnd.ctrl_latency_steps[1] + 1)),
        }
        p = self.params
        body_mass = masses[0] * p["mass_scale"]
        self.body.set_masses([body_mass])
        self.mass_true = float(body_mass + masses[1:].sum())
        self.wind_mean = p["wind_speed"] * np.array([np.cos(wind_dir), np.sin(wind_dir), 0.0])
        self.gust = np.zeros(3)

        self.ctrl = VelocityController(self.mass_nominal, J, self.dt)
        self.ctrl.reset()
        self.motor_state = np.array([self.mass_nominal * G, 0.0, 0.0, 0.0])
        self.cmd_queue = [np.zeros(4)] * p["ctrl_latency_steps"]
        self.rng = rng
        self._eye = None  # smoothed chase-camera heading
        self.t, self.step_count = 0.0, 0
        self.timeout = cfg.TIMEOUT_SLACK * np.linalg.norm(goal - start) / cfg.CRUISE_SPEED + 15.0
        self.path_flown = 0.0
        self._last_pos = np.array([start[0], start[1], z0])
        return self.observe()

    # --- State and sensing -----------------------------------------------------------

    def true_state(self) -> dict:
        tf = self._view.get_transforms().numpy()[0].astype(float)  # x y z qx qy qz qw
        vel = self._view.get_velocities().numpy()[0].astype(float)  # lin, ang (world)
        quat = tf[[6, 3, 4, 5]]
        R = quat_to_rot(quat)
        return {"pos": tf[:3], "quat": quat, "R": R, "vel": vel[:3], "w_body": R.T @ vel[3:]}

    def _apply(self, force_world: np.ndarray, torque_world: np.ndarray):
        self._f_buf.numpy()[0] = force_world
        self._t_buf.numpy()[0] = torque_world
        self._view.apply_forces_and_torques_at_position(self._f_buf, self._t_buf, None, self._idx, True)

    def _ray(self, origin: np.ndarray, d: np.ndarray, max_dist: float) -> float:
        # start outside the drone's own collider (0.12 m clears the Crazyflie, not the 0.36 m hybrid frame)
        r0 = max(0.12, self.drone["radius"] + 0.05)
        hit = self._sq.raycast_closest(tuple(origin + r0 * d), tuple(d), max_dist)
        return float(hit["distance"] + r0) if hit["hit"] else max_dist

    def observe(self) -> dict:
        """Onboard-only observation (what the real drone could measure), plus privileged truth."""
        s = self.true_state()
        sigma = self.params["sensor_noise"]
        n = lambda size, scale=1.0: self.rng.normal(0, sigma * scale, size) if sigma > 0 else np.zeros(size)
        ranges = np.array([self._ray(s["pos"], d, cfg.RAY_MAX) for d in RAY_DIRS])
        alt = self._ray(s["pos"], np.array([0.0, 0.0, -1.0]), 20.0)
        pos_est = s["pos"] + n(3)
        obs = {
            "goal_rel": (np.append(self.goal, self.goal_z) - pos_est).astype(np.float32),  # from noisy state estimate
            "vel": (s["vel"] + n(3, 2.0)).astype(np.float32),
            "alt": np.float32(alt + n(1)[0]),
            "ranges": np.clip(ranges + n(cfg.NUM_RAYS), 0.0, cfg.RAY_MAX).astype(np.float32),
        }
        return {"obs": obs, "true": s}

    @staticmethod
    def flatten(obs: dict) -> np.ndarray:
        return np.concatenate([obs["goal_rel"], obs["vel"], [obs["alt"]], obs["ranges"]]).astype(np.float32)

    def _touching(self, pos: np.ndarray) -> str:
        """Twin crash check against the real meshes: what (other than the drone) overlaps it."""
        hits = []

        def report(hit):
            if not hit.collision.startswith(self.drone["root"]):
                hits.append(hit.collision)
            return True

        self._sq.overlap_sphere(self.drone["radius"] + cfg.MIN_CLEARANCE, tuple(pos), report, False)
        if not hits:
            return ""
        return "crash_ground" if all("/Twin/terrain" in h for h in hits) else "crash_obstacle"

    def _chase_camera(self):
        """GUI only: keep the drone in view from behind and above."""
        from isaacsim.core.rendering_manager import ViewportManager

        s = self.true_state()
        heading = s["vel"][:2] if np.linalg.norm(s["vel"][:2]) > 0.3 else self.goal - s["pos"][:2]
        heading = heading / max(np.linalg.norm(heading), 1e-6)
        # smooth the viewing direction, not the eye position, so the camera never lags behind
        self._eye = heading if self._eye is None else 0.8 * self._eye + 0.2 * heading
        h = self._eye / max(np.linalg.norm(self._eye), 1e-6)
        back, up = self.chase_offset
        eye = s["pos"] + np.array([-back * h[0], -back * h[1], up])
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=s["pos"])

    # --- Stepping --------------------------------------------------------------------

    def step(self, cmd: np.ndarray) -> tuple[dict, bool, str]:
        """Apply a velocity setpoint [vx, vy, vz, yaw_rate] for one control period (1/CONTROL_HZ s)."""
        p = self.params
        self.cmd_queue.append(np.asarray(cmd, float))
        active = self.cmd_queue.pop(0)
        alpha = self.dt / (p["motor_tau"] + self.dt)
        for _ in range(self.substeps):
            s = self.true_state()
            thrust, torque = self.ctrl(s["R"], s["vel"], s["w_body"], active)
            # plant mismatch: thrust gain and first-order motor lag
            self.motor_state += alpha * (np.array([thrust * p["thrust_gain"], *torque]) - self.motor_state)
            # wind: mean + Ornstein-Uhlenbeck gusts, felt as linear drag on relative air speed
            if p["gust_std"] > 0:
                self.gust += -self.gust * self.dt / 2.0 + p["gust_std"] * np.sqrt(self.dt) * self.rng.normal(0, 1, 3) * np.array([1, 1, 0.3])
            wind = self.wind_mean + self.gust
            f_world = s["R"][:, 2] * self.motor_state[0] + WIND_DRAG * self.mass_true * (wind - s["vel"]) * np.array([1, 1, 0.5])
            tau_world = s["R"] @ self.motor_state[1:]
            self._apply(f_world, tau_world)
            self._sm.step()
        self.step_count += 1
        self.t += 1.0 / cfg.CONTROL_HZ
        if not self.headless and self.step_count % self.render_every == 0:
            self._chase_camera()
            self.app.update()

        out = self.observe()
        pos = out["true"]["pos"]
        self.path_flown += float(np.linalg.norm(pos - self._last_pos))
        self._last_pos = pos
        if not np.all(np.isfinite(pos)):
            return out, True, "crash_obstacle"
        if pos[2] - self.ground_at(pos[:2]) < 0.3:
            return out, True, "crash_ground"
        if self.tmap is None:
            if clearance(self.layout, pos) < cfg.MIN_CLEARANCE:
                return out, True, "crash_obstacle"
        elif crash := self._touching(pos):
            return out, True, crash
        if np.linalg.norm(pos[:2] - self.goal) < cfg.GOAL_RADIUS and abs(pos[2] - self.goal_z) < 1.5:
            return out, True, "success"
        if (self.tmap is None and np.any(np.abs(pos[:2]) > cfg.CAMPUS_HALF)) or (self.tmap is not None and not self.tmap.inside(pos[:2])):
            return out, True, "out_of_bounds"
        if self.t > self.timeout:
            return out, True, "timeout"
        return out, False, ""
