"""Quick / final preview renders of build/fd25.blend with Cycles (CPU).

python -m tools.render.preview [quick|final] [outdir]
Uses baked textures if build/textures_preview.json exists, else flat colours.
"""
import json
import math
import os
import sys

import bpy
import mathutils
import numpy as np

from tools.spec import machine as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")

FLAT = {
    "fd25_paint_upper": ((0.92, 0.60, 0.04), 0.0, 0.45),
    "fd25_paint_arm": ((0.92, 0.60, 0.04), 0.0, 0.45),
    "fd25_cyl_paint": ((0.92, 0.60, 0.04), 0.0, 0.45),
    "fd25_attachment": ((0.10, 0.10, 0.11), 0.3, 0.55),
    "fd25_undercarriage": ((0.06, 0.06, 0.06), 0.2, 0.6),
    "fd25_track": ((0.09, 0.08, 0.07), 0.8, 0.7),
    "fd25_steel": ((0.45, 0.45, 0.46), 1.0, 0.45),
    "fd25_steel_dark": ((0.12, 0.12, 0.12), 0.6, 0.55),
    "fd25_chrome": ((0.85, 0.86, 0.88), 1.0, 0.08),
    "fd25_rubber": ((0.03, 0.03, 0.03), 0.0, 0.7),
    "fd25_black": ((0.04, 0.04, 0.04), 0.0, 0.5),
    "fd25_grating": ((0.25, 0.25, 0.25), 0.8, 0.6),
    "fd25_glass": ((0.2, 0.25, 0.25), 0.0, 0.05),
    "fd25_glass_int": ((0.2, 0.25, 0.25), 0.0, 0.05),
    "fd25_interior": ((0.15, 0.15, 0.16), 0.0, 0.7),
    "fd25_lamp": ((0.9, 0.9, 0.85), 0.0, 0.1),
    "fd25_beacon": ((1.0, 0.45, 0.02), 0.0, 0.2),
    "fd25_decals": ((1.0, 1.0, 1.0), 0.0, 0.5),
}


def paint_layers(nt, bsdf, files, paint_rgb):
    """In-game style layered paint: under (primer/steel/rust) -> paint (tint x colour, opacity) -> dirt."""
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"

    def img(key, color=True):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(files[key][1], check_existing=True)
        n.image.colorspace_settings.name = "sRGB" if color else "Non-Color"
        nt.links.new(uvn.outputs["UV"], n.inputs["Vector"])
        return n.outputs["Color"]

    def mix(a, b, fac, blend="MIX"):
        m = nt.nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        m.blend_type = blend
        for src, dst in ((fac, m.inputs[0]), (a, m.inputs[6]), (b, m.inputs[7])):
            if isinstance(src, (int, float)):
                dst.default_value = src
            elif isinstance(src, tuple):
                dst.default_value = src
            else:
                nt.links.new(src, dst)
        return m.outputs[2]

    def bw(sock):
        n = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(sock, n.inputs[0])
        return n.outputs[0]

    lin = tuple(c ** 2.2 for c in paint_rgb) + (1.0,)
    paint_col = mix(img("paint"), lin, 1.0, "MULTIPLY")
    po, do = bw(img("paint_o", False)), bw(img("dirt_o", False))
    base = mix(img("under"), paint_col, po)
    base = mix(base, img("dirt"), do)
    nt.links.new(base, bsdf.inputs["Base Color"])
    rough = mix(img("rough", False), img("paint_r", False), po)
    rough = mix(rough, (0.93, 0.93, 0.93, 1.0), do)
    nt.links.new(bw(rough), bsdf.inputs["Roughness"])
    met = mix(img("metal", False), (0, 0, 0, 1), po)
    met = mix(met, (0, 0, 0, 1), do)
    nt.links.new(bw(met), bsdf.inputs["Metallic"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.uv_map = "UVMap"
    nt.links.new(img("normal", False), nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])


def setup_materials(textured, paint=None):
    info, index = {}, {}
    paint_rgb = M.PAINTS[paint or M.DEFAULT_PAINT]
    if textured and os.path.exists(os.path.join(BUILD, "textures_preview.json")):
        with open(os.path.join(BUILD, "textures_preview.json")) as f:
            info = json.load(f)
        with open(os.path.join(BUILD, "texture_index.json")) as f:
            index = json.load(f)
    FLAT["fd25_paint_upper"] = FLAT["fd25_paint_arm"] = FLAT["fd25_cyl_paint"] = (paint_rgb, 0.0, 0.45)
    for mat in bpy.data.materials:
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        col, met, rough = FLAT.get(mat.name, ((0.5, 0.5, 0.5), 0.0, 0.5))
        bsdf.inputs["Base Color"].default_value = (*col, 1)
        bsdf.inputs["Metallic"].default_value = met
        bsdf.inputs["Roughness"].default_value = rough
        if mat.name.startswith("fd25_glass"):
            bsdf.inputs["Alpha"].default_value = 0.25
            mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else None
        if mat.name in ("fd25_lamp",):
            bsdf.inputs["Emission Color"].default_value = (1, 0.95, 0.8, 1)
            bsdf.inputs["Emission Strength"].default_value = 2.0
        if index and mat.name in ("fd25_paint_upper", "fd25_paint_arm"):
            paint_layers(nt, bsdf, index["P"][mat.name.split("_")[-1]], paint_rgb)
            continue
        if index and mat.name == "fd25_cyl_paint":
            bsdf.inputs["Base Color"].default_value = tuple(c ** 2.2 * 0.9 for c in paint_rgb) + (1.0,)
            bsdf.inputs["Roughness"].default_value = 0.42
            continue
        t = info.get(mat.name)
        if t:
            uvn = nt.nodes.new("ShaderNodeUVMap")
            uvn.uv_map = "UVMap"

            def img(path, color=True):
                n = nt.nodes.new("ShaderNodeTexImage")
                n.image = bpy.data.images.load(path, check_existing=True)
                n.image.colorspace_settings.name = "sRGB" if color else "Non-Color"
                nt.links.new(uvn.outputs["UV"], n.inputs["Vector"])
                return n
            if t.get("base"):
                b = img(t["base"])
                nt.links.new(b.outputs["Color"], bsdf.inputs["Base Color"])
            if t.get("rough"):
                r = img(t["rough"], False)
                nt.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
            if t.get("metal"):
                m = img(t["metal"], False)
                nt.links.new(m.outputs["Color"], bsdf.inputs["Metallic"])
            if t.get("normal"):
                nmap = nt.nodes.new("ShaderNodeNormalMap")
                nmap.uv_map = "UVMap"
                n = img(t["normal"], False)
                nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
                nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
            if t.get("alpha"):
                a = img(t["alpha"], False)
                nt.links.new(a.outputs["Color"], bsdf.inputs["Alpha"])


def setup_world(scene, strength=1.0):
    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(38)
        sky.sun_rotation = math.radians(150)
        sky.sun_disc = False
    except Exception:
        pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.16 * strength
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.4 * strength
    sun.angle = math.radians(2.5)
    sun.color = (1.0, 0.96, 0.9)
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(52), math.radians(8), math.radians(150))
    scene.collection.objects.link(so)
    # colour management: AgX with a bit more contrast, slightly under-exposed like a photo
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = -0.35
    # worn concrete yard floor
    bpy.ops.mesh.primitive_plane_add(size=600, location=(0, 0, 0))
    g = bpy.context.active_object
    gm = bpy.data.materials.new("ground")
    gm.use_nodes = True
    gnt = gm.node_tree
    b = gnt.nodes["Principled BSDF"]
    noise = gnt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 8.0
    ramp = gnt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.10, 0.095, 0.085, 1)
    ramp.color_ramp.elements[1].color = (0.24, 0.225, 0.20, 1)
    gnt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    gnt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.92
    g.data.materials.append(gm)


def look_at(cam_obj, target):
    d = mathutils.Vector(target) - cam_obj.location
    cam_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def render_views(views, outdir, samples=24, res=(1280, 720), prefix=""):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.image_settings.file_format = "PNG"
    cam = bpy.data.cameras.new("cam")
    co = bpy.data.objects.new("cam", cam)
    scene.collection.objects.link(co)
    scene.camera = co
    outs = []
    for name, (pos, target, lens) in views.items():
        co.location = pos
        look_at(co, target)
        cam.lens = lens
        path = os.path.join(outdir, f"{prefix}{name}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        outs.append(path)
    return outs


VIEWS = {
    "hero": ((-11.5, -15.0, 6.0), (0.0, -1.9, 2.75), 42),
    "side": ((14.0, -2.0, 3.0), (0.0, -2.2, 2.0), 30),
    "cab_pov": (tuple(np.array(M.CAB.eye) + np.array([0.0, -0.05, 0.0])), (-0.2, -5.4, 0.9), 22),
    "claw": ((2.2, -7.6, 2.0), (-0.15, -5.4, 1.3), 40),
    "rear": ((7.0, 9.0, 4.0), (0.0, 0.5, 1.5), 30),
}


THUMB_VIEW = ((-11.5, -15.0, 6.0), (0.0, -1.9, 2.75), 41)


def apply_config(cfg_parts):
    """Show / hide / move render objects so the scene matches a .pc configuration."""
    objs = bpy.data.objects
    if cfg_parts.get("fd25_cab") == "fd25_cab_elevated":
        for n in ("fd25_cab_shell", "fd25_cab_glass", "fd25_cab_interior", "fd25_cab_guard"):
            if n in objs:
                objs[n].location.z += M.CAB.riser_elevated
        if "fd25_cab_riser" in objs:
            objs["fd25_cab_riser"].hide_render = False
    if cfg_parts.get("fd25_clamparm") == "fd25_clamparm_none":
        for o in objs:
            if o.name == "fd25_clamp" or o.name.startswith("fd25_cyl_clamp"):
                o.hide_render = True


def render_thumbs(outdir):
    """One thumbnail per configuration (<config>.jpg) plus default.jpg, written into the vehicle folder."""
    from PIL import Image
    from tools.meta.vehicle_info import CONFIGS
    vdir = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)
    outs = []
    for i, (key, cfg) in enumerate(CONFIGS.items()):
        bpy.ops.wm.open_mainfile(filepath=os.path.join(BUILD, "fd25.blend"))
        apply_config(cfg["parts"])
        setup_materials(textured=True, paint=cfg["paint"])
        setup_world(bpy.context.scene)
        png = render_views({key: THUMB_VIEW}, outdir, 64, (1280, 720), prefix="thumb_")[0]
        im = Image.open(png).convert("RGB").resize((800, 450), Image.LANCZOS)
        im.save(os.path.join(vdir, f"{key}.jpg"), quality=90)
        if i == 0:
            im.save(os.path.join(vdir, "default.jpg"), quality=90)
        outs.append(png)
    return outs


def main(mode="quick", outdir=None):
    outdir = outdir or os.path.join(BUILD, "renders")
    os.makedirs(outdir, exist_ok=True)
    if mode == "thumbs":
        return render_thumbs(outdir)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(BUILD, "fd25.blend"))
    setup_materials(textured=(mode != "quick"))
    setup_world(bpy.context.scene)
    samples = {"quick": 16, "test": 12}.get(mode, 96)
    res = {"quick": (960, 540), "test": (640, 360)}.get(mode, (1920, 1080))
    if mode == "test":
        return render_views({"hero": VIEWS["hero"]}, outdir, samples, res, prefix="t_")
    return render_views(VIEWS, outdir, samples, res, prefix=("q_" if mode == "quick" else ""))


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
