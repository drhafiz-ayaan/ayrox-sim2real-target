"""Convert the Gazebo campus twins and the AK hybrid drone into Isaac Sim USD assets.

    ~/Documents/isaacsim/python.sh tools/build_twins.py --src ~/Documents/ak_hybrid_drone_src/ak_hybrid_drone

Input: the `ak_hybrid_drone` ROS/Gazebo package (worlds/*_campus_hybrid_drone.sdf + models/).
Output, under sim2real_target/assets/ (not committed, rebuild from the package):
  twins/<campus>/<model>/<tile>.usd   every DAE tile, converted (metres, Z-up)
  twins/<campus>.usd                  the whole campus, with static triangle-mesh colliders
  twins/<campus>_tris.npz             world-space triangles (terrain / obstacles) for tools/build_maps.py
  ak_hybrid_drone.usd                 the drone as one rigid body (mass/inertia from the SDF)
"""

import argparse
import asyncio
import os
import re
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.expanduser("~/Documents/ak_hybrid_drone_src/ak_hybrid_drone"))
ap.add_argument("--campuses", default="nust,uet_taxila")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402

app_utils.enable_extension("omni.kit.asset_converter")
app.update()

import numpy as np  # noqa: E402
import omni.kit.asset_converter as ac  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
# Which twin models get colliders, and which count as obstacles for the planning map.
COLLIDE = {"terrain", "buildings", "trees", "power_lines"}
OBSTACLE = {"buildings", "trees", "power_lines"}
DRONE_SDF = "hybrid_drone_backtrack"  # the drone placed in nust_campus_hybrid_drone.sdf


def run(coro):
    fut = asyncio.ensure_future(coro)
    while not fut.done():
        app.update()
    return fut.result()


async def _convert(src, dst):
    ctx = ac.AssetConverterContext()
    ctx.ignore_animations = ctx.ignore_cameras = ctx.ignore_lights = True
    ctx.use_meter_as_world_unit = True
    task = ac.get_instance().create_converter_task(src, dst, None, ctx)
    return await task.wait_until_finished(), task.get_error_message()


def convert(src, dst):
    """DAE/STL -> USD. The converter tags the stage Y-up but keeps the source's Z-up geometry."""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok, err = run(_convert(src, dst))
    if not ok:
        raise RuntimeError(f"convert failed: {src}: {err}")
    stage = Usd.Stage.Open(dst)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.Save()


def model_meshes(model_dir):
    """Unique mesh files referenced by a Gazebo model.sdf."""
    text = open(os.path.join(model_dir, "model.sdf")).read()
    uris = dict.fromkeys(re.findall(r"<uri>model://[^/]+/(.+?)</uri>", text))
    return [os.path.join(model_dir, u) for u in uris]


def world_models(world_sdf):
    return [m for m in dict.fromkeys(re.findall(r"model://([a-z_0-9]+)", open(world_sdf).read())) if m != DRONE_SDF]


def mesh_triangles(prim, xf_cache):
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if not len(pts) or not len(counts):
        return np.zeros((0, 3, 3), np.float32)
    m = np.array(xf_cache.GetLocalToWorldTransform(prim)).T  # row-major Gf -> column vectors
    pts = (np.c_[pts, np.ones(len(pts))] @ m.T)[:, :3]
    tris, start = [], 0
    for c in counts:  # fan-triangulate polygons
        f = idx[start : start + c]
        tris.extend((f[0], f[k], f[k + 1]) for k in range(1, c - 1))
        start += c
    return pts[np.asarray(tris)].astype(np.float32)


def build_campus(campus):
    world = os.path.join(args.src, "worlds", f"{campus}_campus_hybrid_drone.sdf")
    out_dir = os.path.join(ASSETS, "twins", campus)
    stage_path = os.path.join(ASSETS, "twins", f"{campus}.usd")
    stage = Usd.Stage.CreateNew(stage_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Twin")
    stage.SetDefaultPrim(root.GetPrim())

    for model in world_models(world):
        kind = model[len(campus) + 1 :]  # nust_buildings -> buildings
        meshes = model_meshes(os.path.join(args.src, "models", model))
        print(f"[{campus}] {model}: {len(meshes)} tiles", flush=True)
        group = UsdGeom.Xform.Define(stage, f"/Twin/{kind}")
        for src in meshes:
            name = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(out_dir, kind, name + ".usd")
            convert(src, dst)
            tile = UsdGeom.Xform.Define(stage, f"{group.GetPath()}/{name}")
            tile.GetPrim().GetReferences().AddReference(os.path.relpath(dst, os.path.dirname(stage_path)))
            tile.GetPrim().SetInstanceable(False)

    # Static triangle-mesh colliders, authored as overrides on the referenced meshes.
    xf = UsdGeom.XformCache()
    tris = {"terrain": [], "obstacles": []}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        kind = str(prim.GetPath()).split("/")[2]
        if kind in COLLIDE:
            UsdPhysics.CollisionAPI.Apply(prim)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
        if kind == "terrain":
            tris["terrain"].append(mesh_triangles(prim, xf))
        elif kind in OBSTACLE:
            tris["obstacles"].append(mesh_triangles(prim, xf))
    stage.Save()
    np.savez_compressed(os.path.join(ASSETS, "twins", f"{campus}_tris.npz"), **{k: np.concatenate(v) for k, v in tris.items()})
    print(f"[{campus}] -> {stage_path} ({sum(len(t) for t in tris['terrain'])} terrain, {sum(len(t) for t in tris['obstacles'])} obstacle triangles)", flush=True)


def build_drone():
    """One rigid body: the SDF's base_link mass/inertia, visual meshes for the frame and 4 props."""
    model = os.path.join(args.src, "models", DRONE_SDF)
    sdf = ET.parse(os.path.join(model, "model.sdf")).getroot()
    base = sdf.find(".//link[@name='base_link']")
    mass = float(base.find("inertial/mass").text)
    inertia = [float(base.find(f"inertial/inertia/{k}").text) for k in ("ixx", "iyy", "izz")]
    mesh_dir = os.path.join(ASSETS, "drone")
    convert(os.path.join(model, "meshes", "base_link.STL"), os.path.join(mesh_dir, "base_link.usd"))
    for p in ("iris_prop_ccw", "iris_prop_cw"):
        convert(os.path.join(model, "meshes", p + ".dae"), os.path.join(mesh_dir, p + ".usd"))

    path = os.path.join(ASSETS, "ak_hybrid_drone.usd")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/Drone")
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mapi = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mapi.CreateMassAttr(mass)
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

    vis = UsdGeom.Xform.Define(stage, "/Drone/frame")
    vis.GetPrim().GetReferences().AddReference("drone/base_link.usd")
    # Prop hubs from the SDF joint poses (relative to base_link).
    for joint in sdf.findall(".//joint"):
        if not joint.get("name", "").startswith("prop"):
            continue
        x, y, z, *_ = map(float, joint.find("pose").text.split())
        child = joint.find("child").text
        mesh = "iris_prop_ccw" if child in ("prop_fr", "prop_bl") else "iris_prop_cw"
        prop = UsdGeom.Xform.Define(stage, f"/Drone/{child}")
        prop.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        prop.GetPrim().GetReferences().AddReference(f"drone/{mesh}.usd")

    # Collider: a box around the frame mesh (the SDF's own boxes are for Gazebo's landing gear).
    bb = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(vis.GetPrim()).GetRange()
    lo, hi = np.array(bb.GetMin()), np.array(bb.GetMax())
    col = UsdGeom.Cube.Define(stage, "/Drone/collision")
    col.CreateSizeAttr(1.0)
    col.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    col.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    stage.Save()
    print(f"[drone] -> {path}: mass {mass} kg, inertia {inertia}, collision box {np.round(hi - lo, 3)} m", flush=True)


build_drone()
for campus in args.campuses.split(","):
    build_campus(campus)
app.close()
