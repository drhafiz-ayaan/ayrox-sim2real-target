"""Fly a few flights in the Isaac Sim window and save viewport screenshots for the docs.

    ~/Documents/isaacsim/python.sh capture_showcase.py

Per flight: an oblique overview of the route, a top-down view, and close chase-camera frames
along the way. Images go to docs/images/isaac_<split>_<index>_<view>.jpg.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg  # before SimulationApp: Isaac puts cv2's own `config` module on sys.path
from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

from env import TargetFlightEnv  # noqa: E402
from rollout import ExpertPolicy  # noqa: E402

SHOTS = [("train_clean", 0), ("train_varied", 11), ("test_harsh", 0)]
CHASE_AT = (0.15, 0.45, 0.75)  # fraction of the straight-line distance covered
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "images")
os.makedirs(OUT, exist_ok=True)

env = TargetFlightEnv(app, headless=False)
env.chase_offset = (0.5, 0.2)  # close enough that the 10 cm Crazyflie fills part of the frame
viewport = get_active_viewport()
policy = ExpertPolicy()


def settle(n=30):
    for _ in range(n):
        app.update()


def capture(name, eye=None, target=None):
    if eye is not None:
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye, target=target)
    settle()
    path = os.path.join(OUT, name + ".png")
    capture_viewport_to_file(viewport, path)
    settle(20)
    # PNG captures are ~1.3 MB each; JPEG keeps the repo light
    import cv2  # bundled with Isaac Sim (PIL is not)

    cv2.imwrite(path[:-4] + ".jpg", cv2.imread(path), [cv2.IMWRITE_JPEG_QUALITY, 88])
    os.remove(path)
    print("saved", path[:-4] + ".jpg", flush=True)


for split_name, index in SHOTS:
    split = cfg.SPLITS[split_name]
    out = env.reset(split.campus_seed, split.mode, split.seed_offset + index)
    tag = f"isaac_{split_name}_{index}"
    s, g = env.start, env.goal
    mid = np.append((s + g) / 2, 2.0)
    back = (s - g) / np.linalg.norm(s - g)
    capture(f"{tag}_overview", eye=np.append(s + back * 30, 28.0), target=mid)
    capture(f"{tag}_topdown", eye=mid + np.array([0.0, -1.0, 0.9 * np.linalg.norm(g - s)]), target=mid)

    pending = list(CHASE_AT)
    dist0 = np.linalg.norm(g - s)
    done = False
    while not done:
        out, done, outcome = env.step(policy.act(out, env))
        progress = 1 - np.linalg.norm(out["true"]["pos"][:2] - g) / dist0
        if pending and progress >= pending[0]:
            capture(f"{tag}_chase{int(pending.pop(0) * 100):02d}")
    print(f"{tag}: {outcome}", flush=True)

app.close()
