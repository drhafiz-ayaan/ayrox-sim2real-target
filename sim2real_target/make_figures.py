"""Render the result figures from data/ and results/ (no Isaac Sim needed).

    python3 make_figures.py

Writes docs/figures/*.png and docs/figures/summary.json.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import config as cfg  # noqa: E402
from campus import base_campus  # noqa: E402
from expert import plan_flight  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
C_BASE, C_TREE, C_EXTRA, C_PATH, C_PLAN = "#9aa0a6", "#4c9a52", "#d08a2e", "#1f5fbf", "#222222"


def load_index(split):
    path = os.path.join(HERE, "data", split, "index.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def load_flight(split, rec):
    return np.load(os.path.join(HERE, "data", split, rec["file"]))


def replay_layout(rec):
    sp = cfg.SPLITS[rec["split"]]
    base = base_campus(sp.campus_seed)
    start, goal, extra, _, expert = plan_flight(base, cfg.MODES[sp.mode], np.random.default_rng([sp.campus_seed, rec["flight_seed"]]))
    return base, extra, expert


def draw_layout(ax, layout, color, tree_color=None, alpha=1.0):
    for cx, cy, sx, sy, _ in layout.boxes:
        ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc=color, ec="none", alpha=alpha))
    for cx, cy, r, _ in layout.cylinders:
        ax.add_patch(Circle((cx, cy), max(r, 0.6), fc=tree_color or color, ec="none", alpha=alpha))
    for x1, y1, x2, y2, _ in layout.cables:
        ax.plot([x1, x2], [y1, y2], color="#333", lw=0.8, ls="--")


def speed_line(ax, pos, vel, vmax=3.5):
    pts = pos[:, :2].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1), cmap="viridis", norm=plt.Normalize(0, vmax), lw=2.2)
    lc.set_array(np.linalg.norm(vel[:-1, :2], axis=1))
    ax.add_collection(lc)
    return lc


# --- 1. Flights on the campus map -------------------------------------------------


def fig_campus_overview(recs_by_split):
    panels = [
        (split, title, [r for r in recs_by_split.get(split, []) if r.get("file")][:25])
        for split, title in [("train_clean", "Clean flights (base campus only)"), ("train_varied", "Varied flights (each adds its own clutter)")]
    ]
    panels = [p for p in panels if p[2]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6.6), squeeze=False)
    for ax, (split, title, recs) in zip(axes[0], panels):
        base, _, _ = replay_layout(recs[0])
        draw_layout(ax, base, C_BASE, C_TREE)
        for r in recs:
            d = load_flight(split, r)
            ax.plot(d["pos"][:, 0], d["pos"][:, 1], lw=1.2, alpha=0.8)
            ax.plot(*r["start"], "o", ms=3, color="k")
            ax.plot(*r["goal"], "*", ms=8, color="#2ca02c")
        ax.set_title(f"{title}: {len(recs)} flights")
        h = cfg.CAMPUS_HALF
        ax.set(xlim=(-h, h), ylim=(-h, h), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    fig.suptitle("Campus A, top-down: buildings (grey), trees/poles (green), power lines (dashed). ● start  ★ goal", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "campus_flights.png"))
    plt.close(fig)


def fig_flight_detail(split, rec, name):
    base, extra, expert = replay_layout(rec)
    d = load_flight(split, rec)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    draw_layout(ax, base, C_BASE, C_TREE)
    draw_layout(ax, extra, C_EXTRA)
    ax.plot(expert.waypoints[:, 0], expert.waypoints[:, 1], "--", color=C_PLAN, lw=1, label="expert plan (A*)")
    lc = speed_line(ax, d["pos"], d["true_vel"])
    fig.colorbar(lc, ax=ax, shrink=0.7, label="flown speed [m/s]")
    ax.plot(*rec["start"], "o", color="k", ms=7, label="start")
    ax.add_patch(Circle(rec["goal"], cfg.GOAL_RADIUS, fc="#2ca02c", ec="k", label="goal (2 m)"))
    pts = np.vstack([d["pos"][:, :2], expert.waypoints])
    lo, hi = pts.min(0) - 15, pts.max(0) + 15
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    rz = rec["randomization"]
    ax.set_title(
        f"{split} #{rec['index']}: {rec['outcome']}, {rec['straight_dist_m']:.0f} m in {rec['flight_time_s']:.0f} s\n"
        f"{rz['n_extra_obstacles']} extra obstacles (orange) · wind {rz['wind_speed']:.2f} m/s · noise {rz['sensor_noise'] * 100:.0f} cm · "
        f"mass ×{rz['mass_scale']:.2f} · thrust ×{rz['thrust_gain']:.2f} · lag {rz['motor_tau'] * 1000:.0f} ms",
        fontsize=9.5,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# --- 2. One flight over time ---------------------------------------------------------


def fig_timeseries(split, rec, name="flight_timeseries.png", ground=None):
    d = load_flight(split, rec)
    t = d["t"]
    agl = d["pos"][:, 2] - (np.array([ground(p[:2]) for p in d["pos"]]) if ground else 0.0)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, np.linalg.norm(d["action"][:, :2], axis=1), label="commanded (expert)", color=C_PLAN, lw=1)
    axes[0].plot(t, np.linalg.norm(d["true_vel"][:, :2], axis=1), label="flown (true)", color="#d62728", lw=1)
    axes[0].plot(t, np.linalg.norm(d["vel"][:, :2], axis=1), label="onboard estimate (noisy)", color="#999", lw=0.7, alpha=0.8)
    axes[0].set_ylabel("horizontal\nspeed [m/s]")
    axes[0].legend(fontsize=8, ncol=3, loc="lower center")
    axes[1].plot(t, agl, color="#d62728", lw=1, label="true height above ground")
    axes[1].plot(t, d["alt"], color="#999", lw=0.7, label="height sensor")
    axes[1].axhline(cfg.FLIGHT_ALT, color="k", ls=":", lw=0.8)
    axes[1].set_ylabel("height above\nground [m]")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[2].plot(t, d["ranges"].min(1), color=C_EXTRA, lw=1)
    axes[2].set_ylabel("closest range\nreading [m]")
    axes[3].plot(t, np.linalg.norm(d["goal_rel"][:, :2], axis=1), color="#2ca02c", lw=1.2)
    axes[3].axhline(cfg.GOAL_RADIUS, color="k", ls=":", lw=0.8)
    axes[3].set_ylabel("distance to\ngoal [m]")
    axes[3].set_xlabel("time [s]")
    axes[0].set_title(f"{split} #{rec['index']}: what the drone saw and did, at 10 Hz", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def fig_sensor_ring(split, rec, name="sensor_ring.png"):
    d = load_flight(split, rec)
    i = int(np.argmin(d["ranges"].min(1)))
    ang = np.linspace(0, 2 * np.pi, cfg.NUM_RAYS, endpoint=False)
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(projection="polar")
    r = d["ranges"][i]
    ax.bar(ang, r, width=2 * np.pi / cfg.NUM_RAYS * 0.85, color=plt.cm.RdYlGn(r / cfg.RAY_MAX), alpha=0.9)
    g = d["goal_rel"][i]
    ax.annotate("", xy=(np.arctan2(g[1], g[0]), cfg.RAY_MAX * 0.9), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.set_ylim(0, cfg.RAY_MAX)
    ax.set_title(f"Onboard range ring at t = {d['t'][i]:.1f} s\n(16 beams, max {cfg.RAY_MAX:.0f} m; arrow = goal direction)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# --- 3. Randomization coverage --------------------------------------------------------


def fig_randomization(recs_by_split):
    keys = [
        ("n_extra_obstacles", "extra obstacles", 1),
        ("wind_speed", "mean wind [m/s]", 1),
        ("sensor_noise", "sensor noise [cm]", 100),
        ("mass_scale", "mass scale", 1),
        ("thrust_gain", "thrust gain", 1),
        ("motor_tau", "motor lag [ms]", 1000),
    ]
    groups = [(s, recs_by_split.get(s, [])) for s in ("train_clean", "train_varied") if recs_by_split.get(s)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (k, label, scale) in zip(axes.flat, keys):
        for (split, recs), color in zip(groups, ["#7f7f7f", C_PATH]):
            v = np.array([r["randomization"][k] for r in recs]) * scale
            if np.ptp(v) == 0:
                ax.axvline(v[0], color=color, lw=3, label=f"{split} (fixed)")
            else:
                ax.hist(v, bins=15, color=color, alpha=0.75, label=split)
        ax.set_xlabel(label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Domain randomization actually drawn per flight", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "randomization.png"))
    plt.close(fig)


# --- 3b. Campus twins --------------------------------------------------------------------


_TMAPS = {}


def tmap_of(world):
    from twin import TwinMap

    if world not in _TMAPS:
        _TMAPS[world] = TwinMap(world)
    return _TMAPS[world]


def replay_twin(rec):
    from twin import plan_flight as plan_twin

    sp = cfg.SPLITS[rec["split"]]
    return plan_twin(tmap_of(sp.world), cfg.MODES[sp.mode], np.random.default_rng([sp.campus_seed, rec["flight_seed"]]))


def draw_twin(ax, tm, window=None):
    """Obstacle height above ground (buildings, trees, power lines) with terrain contours."""
    with np.errstate(invalid="ignore"):
        h = np.where(np.isfinite(tm.top), tm.top - tm.ground, 0.0)
    x0, x1, y0, y1 = tm.extent
    sl = (slice(None), slice(None))
    if window is not None:
        (wx0, wy0), (wx1, wy1) = window
        i0, j0 = tm.cell((wx0, wy0))
        i1, j1 = tm.cell((wx1, wy1))
        sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        x0, x1, y0, y1 = tm.x0 + i0 * tm.res, tm.x0 + (i1 + 1) * tm.res, tm.y0 + j0 * tm.res, tm.y0 + (j1 + 1) * tm.res
    ext = [x0, x1, y0, y1]
    ax.imshow(np.clip(h[sl], 0, 25).T, origin="lower", extent=ext, cmap="Greys", vmin=-3, vmax=25, interpolation="nearest")
    g = tm.ground[sl]
    ax.contour(np.linspace(x0, x1, g.shape[0]), np.linspace(y0, y1, g.shape[1]), g.T, levels=12, colors="#8a6d3b", linewidths=0.4, alpha=0.6)
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", xlabel="x east [m]", ylabel="y north [m]")


def fig_twin_overview(world, recs, name, title):
    tm = tmap_of(world)
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_twin(ax, tm)
    for r in recs:
        if r.get("file"):
            d = load_flight(r["split"], r)
            ax.plot(d["pos"][:, 0], d["pos"][:, 1], lw=1.4, color="#1f77b4")
        else:
            *_, expert = replay_twin(r)
            ax.plot(expert.waypoints[:, 0], expert.waypoints[:, 1], lw=1.4, color="#d62728" if not r["success"] else "#9467bd")
        ax.plot(*r["goal"], "*", ms=6, color="#2ca02c")
    ax.set_title(title or f"NUST H-12 twin: {len(recs)} training flights by the AK hybrid drone (blue), ★ goals", fontsize=11)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def fig_twin_flight_detail(split, rec, name):
    tm = tmap_of(cfg.SPLITS[split].world)
    start, goal, extra, _, expert = replay_twin(rec)
    d = load_flight(split, rec)
    pts = np.vstack([d["pos"][:, :2], expert.waypoints])
    lo, hi = pts.min(0) - 20, pts.max(0) + 20
    fig, ax = plt.subplots(figsize=(9, 7.5))
    draw_twin(ax, tm, (lo, hi))
    draw_layout(ax, extra, C_EXTRA)
    ax.plot(expert.waypoints[:, 0], expert.waypoints[:, 1], "--", color=C_PLAN, lw=1, label="expert plan (A* on the twin map)")
    lc = speed_line(ax, d["pos"], d["true_vel"])
    fig.colorbar(lc, ax=ax, shrink=0.7, label="flown speed [m/s]")
    ax.plot(*start, "o", color="k", ms=7, label="start")
    ax.add_patch(Circle(goal, cfg.GOAL_RADIUS, fc="#2ca02c", ec="k", label="goal (2 m)"))
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    rz = rec["randomization"]
    ax.set_title(
        f"NUST H-12 twin, {split} #{rec['index']}: {rec['outcome']}, {rec['straight_dist_m']:.0f} m in {rec['flight_time_s']:.0f} s\n"
        f"grey = real buildings/trees (darker = taller), orange = {rz['n_extra_obstacles']} added obstacles · "
        f"wind {rz['wind_speed']:.2f} m/s · noise {rz['sensor_noise'] * 100:.0f} cm · mass ×{rz['mass_scale']:.2f}",
        fontsize=9.5,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# --- 4. Results ------------------------------------------------------------------------------


PROCEDURAL_ORDER = [("train_clean", "train clean"), ("train_varied", "train varied"), ("test_new_layouts", "new layouts"), ("test_unseen_campus", "unseen campus"), ("test_harsh", "harsher than training")]
TWIN_ORDER = [("nust_train_clean", "NUST train clean"), ("nust_train_varied", "NUST train varied"), ("nust_test_new_layouts", "NUST new layouts"), ("taxila_test_unseen", "UET Taxila (never seen)"), ("nust_test_harsh", "NUST harsher than training")]


def fig_results(order=PROCEDURAL_ORDER, name="results.png", title="Expert pilot success rate per split"):
    rows = []
    for split, label in order:
        recs = load_index(split)
        res = os.path.join(HERE, "results", f"{split}__expert.json")
        if not recs and os.path.exists(res):
            recs = json.load(open(res))["flights"]
        if recs:
            rows.append((label, recs))
    if not rows:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [l for l, _ in rows]
    rate = [100 * np.mean([r["success"] for r in recs]) for _, recs in rows]
    n = [len(recs) for _, recs in rows]
    bars = axes[0].bar(labels, rate, color=[C_BASE, C_PATH, "#2ca02c", "#9467bd", "#d62728"][: len(rows)])
    for b, v, k in zip(bars, rate, n):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%\n(n={k})", ha="center", fontsize=8.5)
    axes[0].set(ylim=(0, 115), ylabel="success rate [%]", title=title)
    axes[0].tick_params(axis="x", rotation=20)
    ratio = [[r["path_flown_m"] / r["straight_dist_m"] for r in recs if r["success"]] for _, recs in rows]
    axes[1].boxplot(ratio, labels=labels, showfliers=False)
    axes[1].set(ylabel="flown / straight-line distance", title="Detour needed to get around obstacles")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    return {l: {"flights": k, "success_rate": round(v / 100, 3)} for l, v, k in zip(labels, rate, n)}


def write_readme_table(summary):
    """Refresh the results table between the RESULTS_TABLE markers in the top-level README."""
    path = os.path.join(HERE, "..", "README.md")
    if not os.path.exists(path):
        return
    text = open(path).read()
    a, b = "<!-- RESULTS_TABLE -->", "<!-- /RESULTS_TABLE -->"
    if a not in text or b not in text:
        return
    rows = ["| World · drone | Split | Flights | Expert success |", "|---|---|---|---|"]
    for group, splits in summary["splits"].items():
        rows += [f"| {group} | {k} | {v['flights']} | {100 * v['success_rate']:.0f}% |" for k, v in splits.items()]
    ds = summary["dataset"]
    hours = sum(d["hours"] for d in ds.values())
    km = sum(d["km"] for d in ds.values())
    rows += ["", f"Synthetic dataset: **{sum(d['flights_saved'] for d in ds.values())} flights, {hours:.1f} h, {km:.1f} km flown**, 0 real flights. "
             "Exact numbers are in [`summary.json`](sim2real_target/docs/figures/summary.json)."]
    open(path, "w").write(text[: text.index(a) + len(a)] + "\n" + "\n".join(rows) + "\n" + text[text.index(b):])


if __name__ == "__main__":
    train = ("train_clean", "train_varied", "nust_train_clean", "nust_train_varied")
    recs_by_split = {s: load_index(s) for s in train}
    ok = {s: [r for r in recs if r.get("file")] for s, recs in recs_by_split.items()}
    # procedural campus (Crazyflie)
    fig_campus_overview(recs_by_split)
    if ok["train_clean"]:
        fig_flight_detail("train_clean", ok["train_clean"][0], "flight_clean.png")
    if ok["train_varied"]:
        rich = max(ok["train_varied"][:40], key=lambda r: r["randomization"]["n_extra_obstacles"])
        fig_flight_detail("train_varied", rich, "flight_varied.png")
        fig_timeseries("train_varied", rich)
        fig_sensor_ring("train_varied", rich)
    fig_randomization({s: recs_by_split[s] for s in ("train_clean", "train_varied")})
    # campus twins (AyroX hybrid drone)
    if ok["nust_train_varied"] or ok["nust_train_clean"]:
        fig_twin_overview("nust", ok["nust_train_clean"] + ok["nust_train_varied"], "twin_nust_flights.png", "")
    if ok["nust_train_varied"]:
        rich = max(ok["nust_train_varied"][:60], key=lambda r: r["randomization"]["n_extra_obstacles"])
        fig_twin_flight_detail("nust_train_varied", rich, "twin_flight_varied.png")
        fig_timeseries("nust_train_varied", rich, "twin_flight_timeseries.png", ground=tmap_of("nust").ground_at)
        fig_sensor_ring("nust_train_varied", rich, "twin_sensor_ring.png")
    taxila = os.path.join(HERE, "results", "taxila_test_unseen__expert.json")
    if os.path.exists(taxila):
        fig_twin_overview("uet_taxila", json.load(open(taxila))["flights"], "twin_taxila_flights.png",
                          "UET Taxila twin (never trained on): expert routes of the test flights")
    summary = {
        "splits": {
            "Procedural campus · Crazyflie": fig_results(),
            "NUST / UET Taxila twins · AK hybrid": fig_results(TWIN_ORDER, "results_twin.png", "Expert success rate, campus twins"),
        },
        "dataset": {
            s: {
                "flights_saved": len(ok[s]),
                "hours": round(sum(r["flight_time_s"] for r in ok[s]) / 3600, 2),
                "km": round(sum(r["path_flown_m"] for r in ok[s]) / 1000, 1),
            }
            for s in ok
        },
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    write_readme_table(summary)
    print(json.dumps(summary, indent=1))
    print("figures ->", OUT)
