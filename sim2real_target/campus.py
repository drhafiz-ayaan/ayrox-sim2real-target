"""Procedural mini-campus: layout sampling and geometry queries (pure numpy, no Isaac imports).

A layout holds three obstacle kinds, all standing on flat ground (z = 0):
  boxes      [cx, cy, sx, sy, h]       axis-aligned buildings / containers
  cylinders  [cx, cy, r, h]            trees (crown as a cylinder) and poles
  cables     [x1, y1, x2, y2, z]       power-line spans, thin and horizontal
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg

CABLE_RADIUS = 0.05


@dataclass
class Layout:
    boxes: np.ndarray  # (N, 5)
    cylinders: np.ndarray  # (M, 4)
    cables: np.ndarray  # (K, 5)
    half: float = cfg.CAMPUS_HALF

    def merged(self, other: "Layout") -> "Layout":
        return Layout(
            np.vstack([self.boxes, other.boxes]),
            np.vstack([self.cylinders, other.cylinders]),
            np.vstack([self.cables, other.cables]),
            self.half,
        )

    def counts(self) -> dict:
        return {"boxes": len(self.boxes), "cylinders": len(self.cylinders), "cables": len(self.cables)}


def empty_layout() -> Layout:
    return Layout(np.zeros((0, 5)), np.zeros((0, 4)), np.zeros((0, 5)))


# --- Geometry ------------------------------------------------------------------


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from point p to segments a[i] -> b[i] (any dimension)."""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * ab - p, axis=1)


def clearance(layout: Layout, p: np.ndarray) -> float:
    """3D distance from point p to the nearest obstacle surface (ground excluded)."""
    d = [np.inf]
    if len(layout.boxes):
        b = layout.boxes
        lo = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, np.zeros(len(b))], 1)
        hi = np.stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2, b[:, 4]], 1)
        outside = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        d.append(np.linalg.norm(outside, axis=1).min())
    if len(layout.cylinders):
        c = layout.cylinders
        radial = np.maximum(np.hypot(c[:, 0] - p[0], c[:, 1] - p[1]) - c[:, 2], 0.0)
        vertical = np.maximum(p[2] - c[:, 3], 0.0)
        d.append(np.hypot(radial, vertical).min())
    if len(layout.cables):
        k = layout.cables
        a = np.stack([k[:, 0], k[:, 1], k[:, 4]], 1)
        bb = np.stack([k[:, 2], k[:, 3], k[:, 4]], 1)
        d.append((_seg_dist(p, a, bb) - CABLE_RADIUS).min())
    return float(min(d))


def clearance_2d(layout: Layout, xy: np.ndarray, alt: float = cfg.FLIGHT_ALT) -> float:
    """Horizontal clearance at flight altitude (used for spawning and placement)."""
    return clearance(layout, np.array([xy[0], xy[1], alt]))


# --- Sampling --------------------------------------------------------------------


def _boxes_overlap(box: np.ndarray, boxes: np.ndarray, gap: float) -> bool:
    if not len(boxes):
        return False
    dx = np.abs(boxes[:, 0] - box[0]) - (boxes[:, 2] + box[2]) / 2
    dy = np.abs(boxes[:, 1] - box[1]) - (boxes[:, 3] + box[3]) / 2
    return bool(np.any((dx < gap) & (dy < gap)))


def base_campus(seed: int) -> Layout:
    """The fixed part of a campus (the 'digital twin'): buildings, trees, power lines."""
    rng = np.random.default_rng(10_000 + seed)
    h = cfg.CAMPUS_HALF - 5
    boxes = []
    while len(boxes) < cfg.BASE_BUILDINGS:
        box = np.array([rng.uniform(-h, h), rng.uniform(-h, h), rng.uniform(8, 40), rng.uniform(8, 30), rng.uniform(6, 25)])
        if not _boxes_overlap(box, np.array(boxes).reshape(-1, 5), gap=8.0):
            boxes.append(box)
    layout = Layout(np.array(boxes), np.zeros((0, 4)), np.zeros((0, 5)))

    trees = []
    while len(trees) < cfg.BASE_TREES:
        r = rng.uniform(1.0, 2.5)
        xy = rng.uniform(-h, h, 2)
        if clearance_2d(layout, xy) > r + 2.0:
            trees.append([xy[0], xy[1], r, rng.uniform(6, 12)])
    cyl = np.array(trees)

    # Power lines: straight runs of poles every ~35 m with cables at one height per line.
    cables, poles = [], []
    for _ in range(cfg.BASE_POWER_LINES):
        a, b = rng.uniform(-h, h, 2), rng.uniform(-h, h, 2)
        n = max(2, int(np.linalg.norm(b - a) // 35) + 1)
        z = rng.uniform(4.0, 7.5)
        pts = [a + (b - a) * t for t in np.linspace(0, 1, n)]
        pts = [p for p in pts if clearance_2d(layout, p) > 1.0]
        for p in pts:
            poles.append([p[0], p[1], 0.25, 10.0])
        for p, q in zip(pts[:-1], pts[1:]):
            cables.append([p[0], p[1], q[0], q[1], z])
    cyl = np.vstack([cyl, np.array(poles).reshape(-1, 4)])
    return Layout(layout.boxes, cyl, np.array(cables).reshape(-1, 5))


def sample_start_goal(layout: Layout, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h = cfg.CAMPUS_HALF - 10
    for _ in range(10_000):
        s = rng.uniform(-h, h, 2)
        if clearance_2d(layout, s) < 4.0:
            continue
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(*cfg.GOAL_DIST)
        g = s + dist * np.array([np.cos(ang), np.sin(ang)])
        if np.all(np.abs(g) < h) and clearance_2d(layout, g) > 4.0:
            return s, g
    raise RuntimeError("could not sample a free start/goal pair")


def extra_obstacles(layout: Layout, start: np.ndarray, goal: np.ndarray, n: int, rng: np.random.Generator) -> Layout:
    """Per-flight clutter placed around the start->goal corridor (containers, trees, poles)."""
    boxes, cyl = [], []
    cur = layout
    d = goal - start
    L = np.linalg.norm(d)
    u, nrm = d / L, np.array([-d[1], d[0]]) / L
    tries = 0
    while len(boxes) + len(cyl) < n and tries < 5000:
        tries += 1
        xy = start + u * rng.uniform(8, L - 8) + nrm * rng.normal(0, 8)
        if np.linalg.norm(xy - start) < 6 or np.linalg.norm(xy - goal) < 6:
            continue
        if rng.random() < 0.45:
            ob = np.array([xy[0], xy[1], rng.uniform(2, 7), rng.uniform(2, 7), rng.uniform(3, 10)])
            if clearance_2d(cur, xy) < max(ob[2], ob[3]) / 2 + 2.5:
                continue
            boxes.append(ob)
        else:
            r = rng.choice([rng.uniform(0.15, 0.35), rng.uniform(0.8, 2.2)])  # pole or tree
            if clearance_2d(cur, xy) < r + 2.5:
                continue
            cyl.append([xy[0], xy[1], r, rng.uniform(6, 12)])
        cur = layout.merged(Layout(np.array(boxes).reshape(-1, 5), np.array(cyl).reshape(-1, 4), np.zeros((0, 5))))
    return Layout(np.array(boxes).reshape(-1, 5), np.array(cyl).reshape(-1, 4), np.zeros((0, 5)))
