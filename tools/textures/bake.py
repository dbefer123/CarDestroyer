"""Bake per-atlas helper maps from build/fd25.blend with Cycles.

For every unique-UV atlas (materials listed in build_blend.ATLAS) bakes:
  ao   - ambient occlusion
  edge - convex/concave edge mask from the Bevel shader (1 - dot(N, N_bevel))
  pos  - world position (spawn pose)          (float)
  nrm  - world normal                         (float)
and stores them as float32 numpy arrays in build/bake/<atlas>.npz (+ coverage mask).

Run:  python -m tools.textures.bake [scale] [atlas ...]
"""
import math
import os
import sys
import time

import bpy
import numpy as np

from tools.model.build_blend import ATLAS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(BUILD, "bake")

RES = {"upper": 2048, "arm": 2048, "att": 2048, "uc": 1024, "int": 1024}


def new_image(name, res, float_buffer=True):
    img = bpy.data.images.get(name)
    if img:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, res, res, alpha=True, float_buffer=float_buffer)
    img.generated_color = (0, 0, 0, 0)
    return img


def prepare_materials(target_mat_names, img, mode, dummy):
    """Give every material an active image node: atlas materials -> img, others -> dummy.
    mode selects the emission shader used for EMIT bakes."""
    for mat in bpy.data.materials:
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        if mode == "ao":
            sh = nt.nodes.new("ShaderNodeBsdfDiffuse")
            nt.links.new(sh.outputs[0], out.inputs["Surface"])
        else:
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            nt.links.new(em.outputs[0], out.inputs["Surface"])
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            if mode == "pos":
                nt.links.new(geo.outputs["Position"], em.inputs["Color"])
            elif mode == "nrm":
                nt.links.new(geo.outputs["Normal"], em.inputs["Color"])
            elif mode == "edge":
                bev = nt.nodes.new("ShaderNodeBevel")
                bev.samples = 8
                bev.inputs["Radius"].default_value = 0.018
                dot = nt.nodes.new("ShaderNodeVectorMath")
                dot.operation = "DOT_PRODUCT"
                nt.links.new(bev.outputs["Normal"], dot.inputs[0])
                nt.links.new(geo.outputs["Normal"], dot.inputs[1])
                # edge = 1 - dot  (0 on flat faces, grows at edges); concavity sign via pointiness
                inv = nt.nodes.new("ShaderNodeMath")
                inv.operation = "SUBTRACT"
                inv.inputs[0].default_value = 1.0
                nt.links.new(dot.outputs["Value"], inv.inputs[1])
                comb = nt.nodes.new("ShaderNodeCombineXYZ")
                nt.links.new(inv.outputs[0], comb.inputs["X"])
                nt.links.new(geo.outputs["Pointiness"], comb.inputs["Y"])
                nt.links.new(comb.outputs[0], em.inputs["Color"])
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img if mat.name in target_mat_names else dummy
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "UVMap"
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        nt.nodes.active = tex
        tex.select = True


def bake_atlas(atlas, res):
    mats = {m for m, a in ATLAS.items() if a == atlas}
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    members = [o for o in bpy.data.objects if o.type == "MESH" and o.users_collection
               and o.users_collection[0].name == "export"
               and any(m and m.name in mats for m in o.data.materials)]
    dummy = new_image("dummy", 32)
    results = {}
    for mode, samples, btype in (("ao", 48, "AO"), ("edge", 8, "EMIT"), ("pos", 1, "EMIT"), ("nrm", 1, "EMIT")):
        img = new_image(f"{atlas}_{mode}", res)
        prepare_materials(mats, img, mode, dummy)
        scene.cycles.samples = samples
        bpy.ops.object.select_all(action="DESELECT")
        for o in members:
            o.select_set(True)
        bpy.context.view_layer.objects.active = members[0]
        t0 = time.time()
        bpy.ops.object.bake(type=btype, margin=0, use_clear=True, target="IMAGE_TEXTURES")
        px = np.array(img.pixels[:], dtype=np.float32).reshape(res, res, 4)
        results[mode] = px
        print(f"  baked {atlas}/{mode} {res}px in {time.time() - t0:.1f}s")
    cov = (results["pos"][..., 3] > 0.5).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"{atlas}.npz"),
                        ao=results["ao"][..., 0], edge=results["edge"][..., 0], point=results["edge"][..., 1],
                        pos=results["pos"][..., :3], nrm=results["nrm"][..., :3], cov=cov)


def strip_decals():
    """Decal quads float a few mm above the paint; they must not occlude it in the AO bake."""
    import bmesh
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        names = [m.name if m else "" for m in o.data.materials]
        if "fd25_decals" not in names:
            continue
        idx = names.index("fd25_decals")
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index == idx], context="FACES")
        bm.to_mesh(o.data)
        bm.free()


def main(*args):
    scale = 1.0
    only = []
    for a in args:
        try:
            scale = float(a)
        except ValueError:
            only.append(a)
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(BUILD, "fd25.blend"))
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("bakeworld")
    scene.world.light_settings.distance = 0.5   # local AO for texturing
    # render-instance props (tracks, cylinders) take part in occlusion
    for o in bpy.data.objects:
        o.hide_render = False
    # the elevated-cab riser shares the upper atlas but must not occlude the standard cab: bake it far away
    bpy.data.objects["fd25_cab_riser"].location.x += 40.0
    strip_decals()
    for atlas, res in RES.items():
        if only and atlas not in only:
            continue
        bake_atlas(atlas, int(res * float(scale)))


if __name__ == "__main__":
    main(*sys.argv[1:])
