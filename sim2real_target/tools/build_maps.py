"""Rasterize a twin's triangles into a 2.5D planning map (plain Python: numpy + scipy).

    python3 tools/build_maps.py --info ~/Documents/ak_hybrid_drone_src/ak_hybrid_drone/worlds

For every 1 m cell of the modelled extent: terrain height, and the lowest / highest obstacle
surface (buildings, trees, power lines) above it. The expert plans on this map; Isaac Sim
still simulates and senses the full meshes.
"""

import argparse
import json
import os

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWINS = os.path.join(HERE, "assets", "twins")
RES = 1.0


def sample_triangles(tris: np.ndarray, step: float, chunk: int = 4_000_000):
    """Yield points covering each triangle (edges and interior) at roughly `step` spacing."""
    edge = np.linalg.norm(tris[:, [1, 2, 0]] - tris, axis=2).max(1)
    n = np.clip(np.ceil(edge / step), 1, 400).astype(int)
    for k in np.unique(n):
        sel = tris[n == k]
        u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = u + v <= k
        u, v = u[keep] / k, v[keep] / k  # barycentric lattice incl. edges and vertices
        per = max(1, chunk // len(u))
        for i in range(0, len(sel), per):
            t = sel[i : i + per].astype(np.float64)
            a, ab, ac = t[:, 0, None], (t[:, 1] - t[:, 0])[:, None], (t[:, 2] - t[:, 0])[:, None]
            yield (a + u[None, :, None] * ab + v[None, :, None] * ac).reshape(-1, 3)


def rasterize(tris, x0, y0, shape, reducer):
    out = np.full(shape, np.nan if reducer is np.fmax else np.nan, np.float64)
    init = -np.inf if reducer is np.fmax else np.inf
    grid = np.full(shape[0] * shape[1], init)
    for p in sample_triangles(tris, RES / 2):
        i = np.floor((p[:, 0] - x0) / RES).astype(int)
        j = np.floor((p[:, 1] - y0) / RES).astype(int)
        ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        reducer.at(grid, i[ok] * shape[1] + j[ok], p[ok, 2])
    grid = grid.reshape(shape)
    out[np.isfinite(grid)] = grid[np.isfinite(grid)]
    return out


def build(campus: str, info_dir: str):
    info = json.load(open(os.path.join(info_dir, f"{campus}_campus_info.json")))
    ext = info["modelled_extent_m"]
    x0, y0 = ext["xmin"], ext["ymin"]
    shape = (int((ext["xmax"] - x0) / RES), int((ext["ymax"] - y0) / RES))
    tris = np.load(os.path.join(TWINS, f"{campus}_tris.npz"))

    ground = rasterize(tris["terrain"], x0, y0, shape, np.fmax)
    holes = np.isnan(ground)
    if holes.any():  # fill gaps from the nearest measured cell
        idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        ground = ground[tuple(idx)]
    top = rasterize(tris["obstacles"], x0, y0, shape, np.fmax)
    bottom = rasterize(tris["obstacles"], x0, y0, shape, np.fmin)

    path = os.path.join(TWINS, f"{campus}_map.npz")
    np.savez_compressed(path, ground=ground.astype(np.float32), top=top.astype(np.float32), bottom=bottom.astype(np.float32), origin=np.array([x0, y0]), res=RES)
    occ = np.isfinite(top) & (top - ground > 1.0)
    print(f"[{campus}] {shape[0]}x{shape[1]} cells, ground {np.nanmin(ground):.1f}..{np.nanmax(ground):.1f} m, "
          f"{100 * occ.mean():.1f}% cells with obstacles taller than 1 m -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default=os.path.expanduser("~/Documents/ak_hybrid_drone_src/ak_hybrid_drone/worlds"))
    ap.add_argument("--campuses", default="nust,uet_taxila")
    a = ap.parse_args()
    for c in a.campuses.split(","):
        build(c, a.info)
