"""Articulation animation: poses the rendered model with the kinematic solver and writes a GIF.

Every mesh belongs to one rigid body of tools/kinematics.py; a pose moves it by
F_pose(body) * F_spawn(body)^-1.  Hydraulic cylinder barrels / rods are re-aimed between
their (moved) pins, exactly like BeamNG's props do in game.  Hoses that span a joint are hidden.

Run:  python -m tools.render.animate [gif [frames] [width] | still]
"""
import math
import os
import sys

import bpy
import mathutils
import numpy as np

from tools import kinematics as K
from tools.render import preview as P
from tools.spec import machine as M

BUILD = P.BUILD
OUT = os.path.join(BUILD, "anim")

BODY_OF = {
    "fd25_ss_": "ss", "fd25_cab_": "ss",
    "fd25_boom": "boom", "fd25_stick": "stick", "fd25_sidelink": "sidelink", "fd25_hlink": "hlink",
    "fd25_att_bracket": "att", "fd25_att_rotator": "att",
    "fd25_att_ring": "head", "fd25_att_head": "head", "fd25_att_fork": "head", "fd25_att_jaw": "jaw",
    "fd25_clamp": "clamp",
}
HIDE = ("fd25_hose_boomfoot", "fd25_hose_boomstick", "fd25_hose_stickatt")
# hydraulic cylinder: (body carrying the barrel / base pin, body carrying the rod end)
CYL_BODIES = {"boom": ("ss", "boom"), "stick": ("boom", "stick"), "tilt": ("stick", "sidelink"),
              "jaw": ("head", "jaw"), "clamp": ("uc", "clamp")}

D = math.radians
# key poses (degrees for the planar joints, radians via D() for slew / rotator); equal time per segment.
# The roof of the wreck is gripped at key GRIP, carried, and released at key RELEASE.
READY = dict(boom=56, stick_rel=-128, att_rel=-8, jaw=18)
KEYS = [
    dict(READY),                                                                        # 0 ready above the wreck
    dict(READY, jaw=44, clamp=-3),                                                      # 1 open jaw, clamp down
    dict(boom=47, stick_rel=-115, att_rel=-30, jaw=44, clamp=-3),                       # 2 fork under the roof edge
    dict(boom=47, stick_rel=-115, att_rel=-30, jaw=6, clamp=-3),                        # 3 bite
    dict(boom=56, stick_rel=-126, att_rel=-12, jaw=6, clamp=-3),                        # 4 tear the roof up
    dict(boom=56, stick_rel=-126, att_rel=-12, jaw=6, rot=D(60), clamp=-3),             # 5 twist it off
    dict(boom=56, stick_rel=-126, att_rel=-12, jaw=6, rot=D(60), slew=D(45), clamp=-3),  # 6 swing to the pile
    dict(boom=56, stick_rel=-126, att_rel=-12, jaw=44, rot=D(60), slew=D(45), clamp=-3),  # 7 drop it
    dict(READY),                                                                        # 8 back to ready
]
GRIP, RELEASE = 3, 6


def seq_pos(t):
    return min(max(t, 0.0), 1.0) * (len(KEYS) - 1)


def pose_at(t):
    """t in [0, 1] over the whole sequence -> pose dict (radians) with eased segments."""
    base = K.default_pose()
    keys = []
    for k in KEYS:
        p = dict(base)
        for name, v in k.items():
            p[name] = math.radians(v) if name in ("boom", "stick_rel", "att_rel", "jaw", "clamp") else v
        keys.append(p)
    n = len(keys) - 1
    x = min(max(t, 0.0), 1.0) * n
    i = min(int(x), n - 1)
    f = x - i
    f = f * f * (3 - 2 * f)
    return {name: keys[i][name] + (keys[i + 1][name] - keys[i][name]) * f for name in base}


def body_of(name):
    for prefix, body in BODY_OF.items():
        if name == prefix or name.startswith(prefix):
            return body
    return "uc"


def add_car():
    from tools.model.build_blend import part_to_object
    from tools.render import scene_props as SP
    col = bpy.context.scene.collection
    mats = SP.setup_car_materials(bpy)
    out = {}
    for name, part in (("scene_car_body", SP.car_body()), ("scene_car_roof", SP.car_roof())):
        o = part_to_object(name, part, col)
        for i, m in enumerate(o.data.materials):
            o.data.materials[i] = mats[m.name] if m.name in mats else m
        for poly in o.data.polygons:
            poly.use_smooth = False
        out[name] = o
    return out


def setup_scene():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(BUILD, "fd25.blend"))
    P.setup_materials(textured=True)
    P.setup_world(bpy.context.scene)
    objs = {}
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name.startswith("Plane"):
            continue
        if o.hide_render and "__inst" not in o.name:
            continue
        if any(o.name.startswith(h) for h in HIDE):
            o.hide_render = True
            continue
        objs[o.name] = (o, np.array(o.matrix_world))
    return objs


def apply_pose(objs, pose, spawn):
    fs, fp = spawn["frames"], pose["frames"]
    T = {b: fp[b] @ np.linalg.inv(fs[b]) for b in fs}
    T["uc"] = np.eye(4)
    pins = {}
    for name, (o, M0) in objs.items():
        if name.startswith("fd25_cyl_"):
            continue
        base = name.split("__inst")[0]
        o.matrix_world = mathutils.Matrix((T[body_of(base)] @ M0).tolist())
    # cylinders: move each pin with its body, then aim barrel <-> rod at each other
    groups = {}
    for name, (o, M0) in objs.items():
        if not name.startswith("fd25_cyl_"):
            continue
        kind, part = name.split("_")[2], name.split("_")[3].split("__")[0]
        k = name.split("__inst")[1] if "__inst" in name else "0"
        groups.setdefault((kind, k), {})[part] = (o, M0)
    for (kind, k), g in groups.items():
        if "barrel" not in g or "rod" not in g:
            continue
        bb, rb = CYL_BODIES[kind]
        (ob, Mb), (orod, Mr) = g["barrel"], g["rod"]
        pb = (T[bb] @ Mb)[:, 3][:3]
        pr = (T[rb] @ Mr)[:, 3][:3]
        for o, M0, origin, target, body in ((ob, Mb, pb, pr, bb), (orod, Mr, pr, pb, rb)):
            R = (T[body] @ M0)[:3, :3]
            x = target - origin
            x /= np.linalg.norm(x)
            y = R[:, 1] - x * (R[:, 1] @ x)
            y /= np.linalg.norm(y)
            z = np.cross(x, y)
            F = np.eye(4)
            F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = x, y, z, origin
            o.matrix_world = mathutils.Matrix(F.tolist())


def roof_matrix(t, roof0):
    """World matrix of the wreck's roof at sequence time t (gripped, carried, then dropped)."""
    x = seq_pos(t)
    n = len(KEYS) - 1
    if x <= GRIP:
        return roof0
    jaw = lambda tt: K.solve(pose_at(tt))["frames"]["jaw"]
    J0 = jaw(GRIP / n)
    carry = lambda tt: jaw(tt) @ np.linalg.inv(J0) @ roof0
    if x <= RELEASE:
        return carry(t)
    Mr = carry(RELEASE / n)
    # falls flat onto the ground below the release point (keeps its yaw)
    c = Mr[:3, :3] @ ROOF_CENTRE + Mr[:3, 3]
    yaw = math.atan2(Mr[1, 0], Mr[0, 0])
    G = np.eye(4)
    G[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    G[:3, 3] = np.array([c[0], c[1], 0.03]) - G[:3, :3] @ ROOF_CENTRE
    f = min(1.0, (x - RELEASE) / 1.0) ** 2
    A, B = mathutils.Matrix(Mr.tolist()), mathutils.Matrix(G.tolist())
    return np.array(A.lerp(B, f))


ROOF_CENTRE = None


def render_sequence(frames, width, samples, out_dir, times=None, cam_view=None):
    global ROOF_CENTRE
    os.makedirs(out_dir, exist_ok=True)
    spawn = K.solve()
    objs = setup_scene()
    car = add_car()
    roof = car["scene_car_roof"]
    verts = np.array([v.co[:] for v in roof.data.vertices])
    ROOF_CENTRE = verts.mean(0)
    roof0 = np.eye(4)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = width, width * 9 // 16
    cam = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cam)
    scene.collection.objects.link(co)
    scene.camera = co
    pos, target, lens = cam_view or ((-10.5, -14.5, 5.6), (0.6, -3.6, 2.8), 31)
    co.location = pos
    P.look_at(co, target)
    cam.lens = lens
    times = times if times is not None else [i / (frames - 1) for i in range(frames)]
    paths = []
    for i, t in enumerate(times):
        apply_pose(objs, K.solve(pose_at(t)), spawn)
        roof.matrix_world = mathutils.Matrix(roof_matrix(t, roof0).tolist())
        path = os.path.join(out_dir, f"f{i:03d}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        paths.append(path)
        print(f"frame {i + 1}/{len(times)}", flush=True)
    return paths


def main(mode="gif", frames=96, width=640):
    frames, width = int(frames), int(width)
    low = min(min(K.solve(pose_at(i / 80))["pts"][p][2] for p in ("claw_tip_fixed", "claw_tip_jaw")) for i in range(81))
    print(f"lowest claw tip in the sequence: {low:.2f} m")
    if mode == "still":
        # the moment the roof comes off, as a high-quality still for the README
        from PIL import Image
        n = len(KEYS) - 1
        paths = render_sequence(1, 1920, 96, os.path.join(BUILD, "anim_still"), times=[4.6 / n],
                                cam_view=((-8.6, -12.0, 4.4), (0.2, -4.6, 2.4), 36))
        os.makedirs(ROOT_DOCS, exist_ok=True)
        Image.open(paths[0]).convert("RGB").save(os.path.join(ROOT_DOCS, "action.jpg"), quality=90)
        return
    paths = render_sequence(frames, width, 10, OUT)
    make_gif(paths, os.path.join(ROOT_DOCS, "articulation.gif"))


ROOT_DOCS = os.path.join(P.ROOT, "docs", "images")


def make_gif(paths, out, fps=12):
    from PIL import Image
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames = [Image.open(p).convert("RGB") for p in paths]
    pal = frames[len(frames) // 2].quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
