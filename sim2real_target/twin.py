"""Campus digital twins (NUST H-12, UET Taxila): 2.5D map queries and per-flight sampling (pure numpy).

The map comes from tools/build_maps.py. Heights in the map are absolute (twin frame); everything
the drone does is relative to the local ground, so flight altitude is 5 m above the terrain.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

import config as cfg
from campus import Layout, clearance_2d, empty_layout
from expert import ALT_BAND, INFLATE, ExpertPilot, occupancy

HERE = os.path.dirname(os.path.abspath(__file__))
MARGIN = 60.0  # m of map around start/goal given to the planner


class TwinMap:
    def __init__(self, campus: str):
        m = np.load(os.path.join(HERE, "assets", "twins", f"{campus}_map.npz"))
        self.ground, self.top, self.bottom = m["ground"], m["top"], m["bottom"]
        self.x0, self.y0 = (float(v) for v in m["origin"])
        self.res = float(m["res"])
        self.nx, self.ny = self.ground.shape
        self.extent = (self.x0, self.x0 + self.nx * self.res, self.y0, self.y0 + self.ny * self.res)
        # cells blocked at flight altitude (AGL), and the distance from each cell to the nearest one
        h_top, h_bot = self.top - self.ground, self.bottom - self.ground
        with np.errstate(invalid="ignore"):
            self.blocked = np.isfinite(self.top) & (h_top > cfg.FLIGHT_ALT - ALT_BAND) & (h_bot < cfg.FLIGHT_ALT + ALT_BAND)
        self.free_dist = ndimage.distance_transform_edt(~self.blocked) * self.res

    def cell(self, xy) -> tuple[int, int]:
        i = int(np.clip((xy[0] - self.x0) / self.res, 0, self.nx - 1))
        j = int(np.clip((xy[1] - self.y0) / self.res, 0, self.ny - 1))
        return i, j

    def ground_at(self, xy) -> float:
        return float(self.ground[self.cell(xy)])

    def clearance(self, xy) -> float:
        return float(self.free_dist[self.cell(xy)])

    def inside(self, xy, margin: float = 0.0) -> bool:
        x0, x1, y0, y1 = self.extent
        return x0 + margin <= xy[0] <= x1 - margin and y0 + margin <= xy[1] <= y1 - margin

    def window(self, lo, hi):
        """Planner grid over [lo, hi]: blocked cells and the x / y cell-centre axes."""
        i0, j0 = self.cell(lo)
        i1, j1 = self.cell(hi)
        xs = self.x0 + (np.arange(i0, i1 + 1) + 0.5) * self.res
        ys = self.y0 + (np.arange(j0, j1 + 1) + 0.5) * self.res
        return self.blocked[i0 : i1 + 1, j0 : j1 + 1], xs, ys


def corridor_density(tmap: TwinMap, start, goal, half_width: float = 15.0) -> float:
    """Share of cells near the straight start->goal line that are blocked: keeps flights in the built-up campus."""
    occ, xs, ys = tmap.window(np.minimum(start, goal) - half_width, np.maximum(start, goal) + half_width)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    d = goal - start
    t = np.clip(((X - start[0]) * d[0] + (Y - start[1]) * d[1]) / (d @ d), 0, 1)
    near = np.hypot(X - start[0] - t * d[0], Y - start[1] - t * d[1]) < half_width
    return float(occ[near].mean())


def sample_start_goal(tmap: TwinMap, rng: np.random.Generator, min_density: float = 0.03):
    for _ in range(20_000):
        s = rng.uniform([tmap.extent[0] + 150, tmap.extent[2] + 150], [tmap.extent[1] - 150, tmap.extent[3] - 150])
        if tmap.clearance(s) < 4.0:
            continue
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(*cfg.GOAL_DIST)
        g = s + dist * np.array([np.cos(ang), np.sin(ang)])
        if tmap.inside(g, 100) and tmap.clearance(g) > 4.0 and corridor_density(tmap, s, g) >= min_density:
            return s, g
    raise RuntimeError("could not sample a start/goal pair in the built-up area")


def extra_obstacles(tmap: TwinMap, start, goal, n: int, rng: np.random.Generator) -> Layout:
    """Per-flight clutter around the route, kept clear of the twin's own buildings and trees.

    Heights are above local ground; env.py places each one on the terrain.
    """
    boxes, cyl = [], []
    d = goal - start
    L = np.linalg.norm(d)
    u, nrm = d / L, np.array([-d[1], d[0]]) / L
    cur = empty_layout()
    for _ in range(5000):
        if len(boxes) + len(cyl) >= n:
            break
        xy = start + u * rng.uniform(8, L - 8) + nrm * rng.normal(0, 8)
        if np.linalg.norm(xy - start) < 6 or np.linalg.norm(xy - goal) < 6:
            continue
        if rng.random() < 0.45:
            ob = np.array([xy[0], xy[1], rng.uniform(2, 7), rng.uniform(2, 7), rng.uniform(3, 10)])
            need = max(ob[2], ob[3]) / 2 + 2.5
            if tmap.clearance(xy) < need or clearance_2d(cur, xy) < need:
                continue
            boxes.append(ob)
        else:
            r = rng.choice([rng.uniform(0.15, 0.35), rng.uniform(0.8, 2.2)])  # pole or tree
            if tmap.clearance(xy) < r + 2.5 or clearance_2d(cur, xy) < r + 2.5:
                continue
            cyl.append([xy[0], xy[1], r, rng.uniform(6, 12)])
        cur = Layout(np.array(boxes).reshape(-1, 5), np.array(cyl).reshape(-1, 4), np.zeros((0, 5)))
    return cur


def plan_flight(tmap: TwinMap, rnd: cfg.Randomization, rng: np.random.Generator):
    """Twin counterpart of expert.plan_flight: start/goal, clutter and an expert path, from one rng."""
    for _ in range(50):
        start, goal = sample_start_goal(tmap, rng)
        n_extra = int(rng.integers(rnd.extra_obstacles[0], rnd.extra_obstacles[1] + 1))
        extra = extra_obstacles(tmap, start, goal, n_extra, rng)
        occ, xs, ys = tmap.window(np.minimum(start, goal) - MARGIN, np.maximum(start, goal) + MARGIN)
        # half a cell diagonal, so thin poles can't fall between cell centres
        occ = occ | occupancy(extra, xs=xs, ys=ys, inflate=tmap.res * 0.71)
        occ = ndimage.distance_transform_edt(~occ) * tmap.res < INFLATE + tmap.res / 2  # same safety margin as the campus planner
        try:
            expert = ExpertPilot(occ, xs, ys, start, goal, ground=tmap.ground_at)
            return start, goal, extra, n_extra, expert
        except RuntimeError:
            continue
    raise RuntimeError("no solvable flight for this seed")
