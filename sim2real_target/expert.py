"""Privileged expert pilot: A* on an inflated 2D grid at flight altitude + pure-pursuit velocity commands.

The expert sees the true map and the true drone state. It plays the role of the study's
"pilot that flew the training flights". The student policy only ever gets onboard observations.
"""

from __future__ import annotations

import heapq

import numpy as np

import config as cfg
from campus import CABLE_RADIUS, Layout, extra_obstacles, sample_start_goal

RES = 1.0  # m per grid cell
INFLATE = 1.6  # m safety margin around every obstacle
ALT_BAND = 1.5  # obstacles/cables reaching within this of flight altitude block the plane


def occupancy(layout: Layout, alt: float = cfg.FLIGHT_ALT, xs=None, ys=None, inflate: float = INFLATE):
    """Rasterize a layout (heights above ground) onto a grid; the whole campus unless xs/ys are given."""
    if xs is None:
        xs = ys = np.arange(-layout.half, layout.half + RES, RES)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    occ = np.zeros(X.shape, bool)
    for cx, cy, sx, sy, h in layout.boxes:
        if h > alt - ALT_BAND:
            occ |= (np.abs(X - cx) < sx / 2 + inflate) & (np.abs(Y - cy) < sy / 2 + inflate)
    for cx, cy, r, h in layout.cylinders:
        if h > alt - ALT_BAND:
            occ |= np.hypot(X - cx, Y - cy) < r + inflate
    for x1, y1, x2, y2, z in layout.cables:
        if abs(z - alt) < ALT_BAND:
            a, b = np.array([x1, y1]), np.array([x2, y2])
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
            occ |= np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])) < CABLE_RADIUS + inflate
    return occ


def _to_cell(p, xs, ys):
    i = int(np.clip(np.round((p[0] - xs[0]) / RES), 0, len(xs) - 1))
    j = int(np.clip(np.round((p[1] - ys[0]) / RES), 0, len(ys) - 1))
    return i, j


def astar(occ: np.ndarray, start: tuple, goal: tuple) -> list[tuple] | None:
    if occ[start] or occ[goal]:
        return None
    nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    n, m = occ.shape
    g = {start: 0.0}
    parent = {}
    pq = [(0.0, start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy, c in nbrs:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < n and 0 <= nb[1] < m) or occ[nb]:
                continue
            ng = g[cur] + c
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(pq, (ng + np.hypot(nb[0] - goal[0], nb[1] - goal[1]), nb))
    return None


def _line_free(occ, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0, 1, n):
        if occ[int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1])))]:
            return False
    return True


def shortcut(occ, path):
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(occ, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


class ExpertPilot:
    """Plans once on an occupancy grid (axes xs, ys) and then tracks the path.

    `ground(xy)` gives terrain height for twins; the procedural campus is flat (ground = 0).
    """

    def __init__(self, occ, xs, ys, start: np.ndarray, goal: np.ndarray, cruise: float = cfg.CRUISE_SPEED, ground=None):
        cells = astar(occ, _to_cell(start, xs, ys), _to_cell(goal, xs, ys))
        if cells is None:
            raise RuntimeError("no collision-free path")
        cells = shortcut(occ, cells)
        self.waypoints = np.array([[xs[c[0]], ys[c[1]]] for c in cells])
        self.waypoints[0], self.waypoints[-1] = start, goal
        self.goal = goal
        self.cruise = cruise
        self.lookahead = 3.0
        self.ground = ground or (lambda xy: 0.0)

    @classmethod
    def from_layout(cls, layout: Layout, start: np.ndarray, goal: np.ndarray, **kw) -> "ExpertPilot":
        xs = np.arange(-layout.half, layout.half + RES, RES)
        return cls(occupancy(layout), xs, xs, start, goal, **kw)

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1).sum())

    def _carrot(self, xy: np.ndarray) -> np.ndarray:
        """Point `lookahead` metres ahead of the closest point on the path."""
        w = self.waypoints
        a, b = w[:-1], w[1:]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", xy - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        proj = a + t[:, None] * ab
        i = int(np.argmin(np.linalg.norm(proj - xy, axis=1)))
        p, left = proj[i], self.lookahead
        for j in range(i, len(w) - 1):
            seg = w[j + 1] - p
            L = np.linalg.norm(seg)
            if L >= left:
                return p + seg / max(L, 1e-9) * left
            left -= L
            p = w[j + 1]
        return w[-1]

    def act(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Velocity setpoint [vx, vy, vz, yaw_rate] in the world frame."""
        xy = pos[:2]
        to_goal = np.linalg.norm(self.goal - xy)
        d = self._carrot(xy) - xy
        speed = min(self.cruise, 0.6 * to_goal + 0.3)
        v_xy = d / max(np.linalg.norm(d), 1e-6) * speed
        vz = np.clip(1.2 * (self.ground(xy) + cfg.FLIGHT_ALT - pos[2]), -1.0, 1.0)
        return np.array([v_xy[0], v_xy[1], vz, 0.0])


def plan_flight(base: Layout, rnd: cfg.Randomization, rng: np.random.Generator):
    """Sample start/goal and per-flight clutter until the expert finds a path.

    Pure numpy, so a flight's layout can be rebuilt outside Isaac Sim (e.g. for plots)
    by replaying the same rng: np.random.default_rng([campus_seed, flight_seed]).
    """
    for _ in range(50):
        start, goal = sample_start_goal(base, rng)
        n_extra = int(rng.integers(rnd.extra_obstacles[0], rnd.extra_obstacles[1] + 1))
        extra = extra_obstacles(base, start, goal, n_extra, rng)
        try:
            return start, goal, extra, n_extra, ExpertPilot.from_layout(base.merged(extra), start, goal)
        except RuntimeError:
            continue
    raise RuntimeError("no solvable flight for this seed")
