"""Low-level flight stack (pure numpy): velocity setpoint -> collective thrust + body torques.

This mirrors the real Crazyflie firmware split: the policy only ever outputs velocity setpoints
(cflib `send_velocity_world_setpoint(vx, vy, vz, yawrate)`), and an onboard cascade turns them
into motor commands. Keeping that interface identical in sim and on hardware is what lets the
policy transfer; the randomized drone response below is what makes it robust to the mismatch.
"""

from __future__ import annotations

import numpy as np

G = 9.81
MAX_TILT = np.radians(30)


def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # np.cross has a lot of per-call overhead for 3-vectors; this runs every physics step
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class VelocityController:
    """Geometric (Lee et al.) attitude control under a PI velocity loop.

    `mass` and `inertia` are the controller's *nominal* model. The true plant may differ
    (mass scale, thrust gain, motor lag are randomized in the env), as on a real drone.
    """

    def __init__(self, mass: float, inertia: np.ndarray, dt: float):
        self.m, self.J, self.dt = mass, inertia, dt
        self.kv, self.ki = 2.5, 0.8
        self.kR, self.kW = 220.0, 26.0  # in angular-acceleration units (scaled by J below)
        self.i_err = np.zeros(3)
        self.yaw_d = 0.0

    def reset(self, yaw: float = 0.0):
        self.i_err[:] = 0.0
        self.yaw_d = yaw

    def __call__(self, R: np.ndarray, v: np.ndarray, w_body: np.ndarray, cmd: np.ndarray) -> tuple[float, np.ndarray]:
        v_cmd, yaw_rate = cmd[:3], cmd[3]
        err = v_cmd - v
        self.i_err = np.clip(self.i_err + err * self.dt, -2.0, 2.0)
        a = self.kv * err + self.ki * self.i_err
        # limit tilt: horizontal accel <= g tan(max_tilt)
        a_xy = np.linalg.norm(a[:2])
        lim = G * np.tan(MAX_TILT)
        if a_xy > lim:
            a[:2] *= lim / a_xy
        F = self.m * (a + np.array([0.0, 0.0, G]))

        b3 = F / np.linalg.norm(F)
        self.yaw_d += yaw_rate * self.dt
        b1c = np.array([np.cos(self.yaw_d), np.sin(self.yaw_d), 0.0])
        b2 = _cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([_cross(b2, b3), b2, b3])

        thrust = float(np.clip(F @ R[:, 2], 0.0, 2.0 * self.m * G))
        eR = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        eW = w_body - np.array([0.0, 0.0, yaw_rate])
        torque = self.J @ (-self.kR * eR - self.kW * eW) + _cross(w_body, self.J @ w_body)
        return thrust, torque
