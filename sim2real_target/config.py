"""Task definition and domain-randomization ranges for the target-reaching study.

Numbers follow the AyroX synthetic-data drone study (README at repo root):
fly start -> goal 70-150 m away at 5 m above ground, 8-28 extra obstacles per flight,
wind 0.2-1.1 m/s, sensor noise 4-11 cm, randomized drone response.
"""

from dataclasses import dataclass, field

# --- Task -------------------------------------------------------------------
FLIGHT_ALT = 5.0  # m above ground
GOAL_DIST = (70.0, 150.0)  # m, start -> goal straight-line distance
GOAL_RADIUS = 2.0  # m, success when within this distance of the goal
CRUISE_SPEED = 3.0  # m/s, expert cruise speed
TIMEOUT_SLACK = 2.5  # timeout = slack * dist / cruise + 15 s
MIN_CLEARANCE = 0.25  # m, closer than this to any obstacle surface counts as a crash

# --- World --------------------------------------------------------------------
CAMPUS_HALF = 130.0  # m, campus spans [-half, half] in x and y
BASE_BUILDINGS = 28
BASE_TREES = 45
BASE_POWER_LINES = 2  # each line = several poles with cables between them

# --- Rates --------------------------------------------------------------------
PHYSICS_HZ = 100
CONTROL_HZ = 10  # policy / logging rate (velocity setpoints, like cflib's commander)

# --- Sensing -----------------------------------------------------------------
NUM_RAYS = 16  # horizontal range ring (stand-in for a multi-ranger / ToF array)
RAY_MAX = 20.0  # m


@dataclass
class Randomization:
    """Per-flight sampling ranges. A (lo, hi) tuple is sampled uniformly."""

    extra_obstacles: tuple = (8, 28)
    wind_speed: tuple = (0.2, 1.1)  # m/s mean wind
    gust_std: tuple = (0.05, 0.3)  # m/s Ornstein-Uhlenbeck gust strength
    sensor_noise: tuple = (0.04, 0.11)  # m, position / range noise std
    mass_scale: tuple = (0.9, 1.1)
    thrust_gain: tuple = (0.88, 1.12)  # actual / commanded thrust
    motor_tau: tuple = (0.015, 0.04)  # s, first-order motor lag
    ctrl_latency_steps: tuple = (0, 2)  # control periods of setpoint delay


# "clean" = the textbook simulation: base campus only, no wind, no noise, nominal drone.
CLEAN = Randomization(
    extra_obstacles=(0, 0),
    wind_speed=(0.0, 0.0),
    gust_std=(0.0, 0.0),
    sensor_noise=(0.0, 0.0),
    mass_scale=(1.0, 1.0),
    thrust_gain=(1.0, 1.0),
    motor_tau=(0.025, 0.025),
    ctrl_latency_steps=(0, 0),
)

VARIED = Randomization()

# Stress test: beyond the training ranges ("NUST, harsher than training").
HARSH = Randomization(
    extra_obstacles=(28, 40),
    wind_speed=(1.1, 1.8),
    gust_std=(0.3, 0.5),
    sensor_noise=(0.11, 0.16),
    mass_scale=(0.85, 1.15),
    thrust_gain=(0.82, 1.18),
    motor_tau=(0.04, 0.06),
    ctrl_latency_steps=(1, 3),
)

MODES = {"clean": CLEAN, "varied": VARIED, "harsh": HARSH}


# --- Drones ------------------------------------------------------------------------
# usd: Nucleus path (cloud assets) or a path relative to this folder. body: the rigid body that
# gets the thrust/torque; links: every rigid body of the drone (their masses add up).
DRONES = {
    "crazyflie": {
        "usd": "nucleus:/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        "root": "/World/cf",
        "body": "/World/cf/body",
        "links": ["/World/cf/body"] + [f"/World/cf/m{i}_prop" for i in range(1, 5)],
        "radius": 0.08,
        "chase": (0.5, 0.2),
    },
    # AyroX's hybrid drone from the Gazebo package (built by tools/build_twins.py): 1.5 kg, 0.36 m frame
    "ak_hybrid": {
        "usd": "assets/ak_hybrid_drone.usd",
        "root": "/World/drone",
        "body": "/World/drone",
        "links": ["/World/drone"],
        "radius": 0.26,
        "chase": (2.2, 0.9),
    },
}

# --- Worlds ----------------------------------------------------------------------------
# "procedural" = the generated mini-campus (campus.py). Twins come from tools/build_twins.py +
# tools/build_maps.py: assets/twins/<name>.usd and <name>_map.npz.
TWINS = ("nust", "uet_taxila")


@dataclass
class Split:
    """A named set of flights: which world and drone, which randomization, which seed range."""

    campus_seed: int
    mode: str
    seed_offset: int = 0
    world: str = "procedural"
    drone: str = "crazyflie"
    notes: str = field(default="")


# Train and test never share a flight seed. "campus_b" stands in for the unseen UET Taxila twin.
SPLITS = {
    "train_clean": Split(campus_seed=1, mode="clean", seed_offset=0),
    "train_varied": Split(campus_seed=1, mode="varied", seed_offset=0),
    "test_new_layouts": Split(campus_seed=1, mode="varied", seed_offset=1_000_000),
    "test_unseen_campus": Split(campus_seed=2, mode="varied", seed_offset=2_000_000),
    "test_harsh": Split(campus_seed=1, mode="harsh", seed_offset=3_000_000),
    # The study's setup: AyroX hybrid drone in the real campus twins. Train on NUST H-12 only;
    # UET Taxila is never trained on.
    "nust_train_clean": Split(campus_seed=101, mode="clean", seed_offset=0, world="nust", drone="ak_hybrid"),
    "nust_train_varied": Split(campus_seed=101, mode="varied", seed_offset=0, world="nust", drone="ak_hybrid"),
    "nust_test_new_layouts": Split(campus_seed=101, mode="varied", seed_offset=1_000_000, world="nust", drone="ak_hybrid"),
    "taxila_test_unseen": Split(campus_seed=102, mode="varied", seed_offset=2_000_000, world="uet_taxila", drone="ak_hybrid"),
    "nust_test_harsh": Split(campus_seed=101, mode="harsh", seed_offset=3_000_000, world="nust", drone="ak_hybrid"),
}
