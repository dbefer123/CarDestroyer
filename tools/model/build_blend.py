"""Build all FD-25 meshes in Blender, unwrap, and export Collada files for BeamNG.

Run:  python -m tools.model.build_blend
Outputs:
  mod/vehicles/ferrovore_fd25/*.dae
  build/fd25.blend            (scene with everything placed at spawn, for baking / rendering)
  build/meshes.json           (mesh -> dae, triangle counts, materials)
"""
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np

from tools.spec import machine as M
from tools.model import parts_uc, parts_ss, parts_arm, parts_att, parts_cyl
from tools.model.geo import merge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)
BUILD = os.path.join(ROOT, "build")

ATLAS = {  # material -> atlas id (unique UV0 unwrap, baked masks)
    "fd25_paint_upper": "upper",
    "fd25_paint_arm": "arm",
    "fd25_attachment": "att",
    "fd25_undercarriage": "uc",
    "fd25_interior": "int",
}
TILE_SCALE = {  # UV0 scale (repeats per metre) for tiling materials
    "fd25_steel": 2.0, "fd25_steel_dark": 2.0, "fd25_chrome": 1.0, "fd25_rubber": 3.0, "fd25_black": 2.0,
    "fd25_grating": 2.0, "fd25_glass": 0.5, "fd25_glass_int": 0.5, "fd25_cyl_paint": 1.5, "fd25_lamp": 4.0,
    "fd25_beacon": 4.0, "fd25_track": 1.6,
}

DAE_GROUPS = {
    "fd25_undercarriage.dae": ["fd25_uc_frame", "fd25_uc_rollers", "fd25_track_bottom", "fd25_track_top",
                               "fd25_track_front", "fd25_track_rear", "fd25_sprocket", "fd25_idler"],
    "fd25_superstructure.dae": ["fd25_ss_frame", "fd25_ss_counterweight", "fd25_ss_hood", "fd25_ss_tanks",
                                "fd25_ss_details", "fd25_ss_boomfoot", "fd25_ss_decals"],
    "fd25_cab.dae": ["fd25_cab_shell", "fd25_cab_glass", "fd25_cab_interior", "fd25_cab_guard", "fd25_cab_riser"],
    "fd25_arm.dae": ["fd25_boom", "fd25_boom_hoses", "fd25_hose_boomfoot", "fd25_stick", "fd25_hose_boomstick",
                     "fd25_sidelink", "fd25_hlink"],
    "fd25_attachment.dae": ["fd25_att_bracket", "fd25_att_rotator", "fd25_att_ring", "fd25_att_head",
                            "fd25_att_fork", "fd25_att_jaw", "fd25_hose_stickatt", "fd25_clamp"],
    "fd25_cylinders.dae": [],  # filled with every fd25_cyl_* mesh
}


def load_json(name):
    with open(os.path.join(BUILD, name)) as f:
        return json.load(f)


def hydro_min_lengths():
    """Map frozenset(node pair) -> inExtent (minimum cylinder length) from the generated jbeam."""
    from tools.validate.jbeam_load import assemble
    out = {}
    for cfgname in (None, {"fd25_cab": "fd25_cab_elevated"}):
        V = assemble(cfgname)
        for h in V["hydros"]:
            out[frozenset((h["id1"], h["id2"]))] = float(h["inExtent"])
    return out


def collect_meshes():
    nodes = load_json("nodes.json")
    props = load_json("props.json")
    world = {}
    world.update(parts_uc.undercarriage_meshes())
    world.update(parts_ss.superstructure_meshes())
    world.update(parts_ss.cab_meshes())
    world.update(parts_arm.arm_meshes())
    world.update(parts_att.attachment_meshes())
    world.update(parts_att.clamp_meshes())
    # props: representative frame = first prop instance for each mesh
    frames = {}
    for pm in props:
        if pm["mesh"] in ("POINTLIGHT", "SPOTLIGHT"):
            continue
        frames.setdefault(pm["mesh"], []).append(parts_cyl.prop_frame(nodes, pm["ref"], pm["idx"], pm["idy"]))
    local = {}
    for name, part in parts_uc.track_meshes().items():
        F = frames[name][0]
        local[name] = part.transform(np.linalg.inv(F))
    local.update(parts_uc.prop_meshes_local())
    local.update(parts_cyl.cylinder_meshes(nodes, props, hydro_min_lengths()))
    return world, local, frames


# ----------------------------------------------------------------------------- blender helpers
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    return mat


def part_to_object(name, part, collection):
    mesh = bpy.data.meshes.new(name)
    verts = [tuple(v) for v in part.v]
    faces = [tuple(int(i) for i in f) for f in part.f]
    mesh.from_pydata(verts, [], faces)
    mesh.validate(clean_customdata=False)
    mats = sorted(set(part.m))
    for m in mats:
        mesh.materials.append(get_material(m))
    idx = {m: i for i, m in enumerate(mats)}
    # validate() may drop degenerate faces; map by polygon order when counts still match
    if len(mesh.polygons) == len(part.f):
        mesh.polygons.foreach_set("material_index", [idx[m] for m in part.m])
    uv0 = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.new(name="UVDetail")
    # explicit UVs from the generator
    if len(mesh.polygons) == len(part.f):
        has = [u is not None for u in part.uv]
        if any(has):
            for poly, uvs in zip(mesh.polygons, part.uv):
                if uvs is None:
                    continue
                for li, (u, v) in zip(poly.loop_indices, uvs):
                    uv0.data[li].uv = (u, v)
            mesh["explicit_uv"] = [1 if h else 0 for h in has]
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def smooth_by_angle(obj, angle=math.radians(38)):
    mesh = obj.data
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=angle)


def box_uvs(obj, layer, scale_fn, only_polys=None, world=True):
    mesh = obj.data
    mw = obj.matrix_world if world else None
    uvl = mesh.uv_layers[layer]
    mats = [m.name if m else "" for m in mesh.materials]
    for poly in mesh.polygons:
        if only_polys is not None and not only_polys(poly, mats):
            continue
        n = poly.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        sc = scale_fn(mats[poly.material_index] if poly.material_index < len(mats) else "")
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            if mw is not None:
                co = mw @ co
            uvl.data[li].uv = (co[a] * sc, co[b] * sc)


def atlas_unwrap(objs, atlas_mats):
    """Smart-project + pack the faces using atlas_mats across all objs into one UV0 atlas."""
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
        o.data.uv_layers.active = o.data.uv_layers["UVMap"]
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    for o in objs:
        bm = bmesh.from_edit_mesh(o.data)
        mats = [m.name if m else "" for m in o.data.materials]
        for f in bm.faces:
            f.select_set(f.material_index < len(mats) and mats[f.material_index] in atlas_mats)
        bmesh.update_edit_mesh(o.data)
    bpy.ops.uv.smart_project(angle_limit=math.radians(52), island_margin=0.0025, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.select_all(action="SELECT")
    try:
        bpy.ops.uv.pack_islands(rotate=True, margin=0.003, shape_method="CONCAVE")
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")


def export_dae(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.collada_export(filepath=path, selected=True, apply_modifiers=True, triangulate=True,
                              include_children=False, use_texture_copies=False, active_uv_only=False,
                              export_mesh_type_selection="render")


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def main():
    world, local, frames = collect_meshes()
    reset_scene()
    col_export = bpy.data.collections.new("export")
    col_show = bpy.data.collections.new("render_instances")
    bpy.context.scene.collection.children.link(col_export)
    bpy.context.scene.collection.children.link(col_show)

    objs = {}
    for name, part in world.items():
        objs[name] = part_to_object(name, part, col_export)
    for name, part in local.items():
        objs[name] = part_to_object(name, part, col_export)

    # smooth shading
    for o in objs.values():
        smooth_by_angle(o)

    # UV0: atlas unwrap per atlas id (world meshes only; props use tiling materials)
    for atlas in sorted(set(ATLAS.values())):
        mats = {m for m, a in ATLAS.items() if a == atlas}
        members = [o for o in objs.values() if any(m.name in mats for m in o.data.materials if m)]
        atlas_unwrap(members, mats)
        print(f"atlas {atlas}: {len(members)} objects")

    # UV0 for tiling materials (box projection, world metres) unless explicit; UV1 = world detail coords
    def tiling(poly, mats):
        name = mats[poly.material_index] if poly.material_index < len(mats) else ""
        return name not in ATLAS

    for name, o in objs.items():
        explicit = o.data.get("explicit_uv")
        if explicit is not None:
            ex = list(explicit)
            box_uvs(o, "UVMap", lambda m: TILE_SCALE.get(m, 1.0),
                    only_polys=lambda poly, mats: tiling(poly, mats) and not ex[poly.index], world=name in world)
        else:
            box_uvs(o, "UVMap", lambda m: TILE_SCALE.get(m, 1.0), only_polys=tiling, world=name in world)
        box_uvs(o, "UVDetail", lambda m: 1.0, world=True)

    # export
    groups = dict(DAE_GROUPS)
    groups["fd25_cylinders.dae"] = sorted(n for n in local if n.startswith("fd25_cyl_"))
    meta = {}
    os.makedirs(VDIR, exist_ok=True)
    for fn, names in groups.items():
        sel = [objs[n] for n in names if n in objs]
        missing = [n for n in names if n not in objs]
        if missing:
            print("WARNING missing meshes", missing)
        export_dae(sel, os.path.join(VDIR, fn))
        for o in sel:
            meta[o.name] = dict(dae=fn, tris=tri_count(o), materials=[m.name for m in o.data.materials if m])
    total = sum(v["tris"] for v in meta.values())
    print(f"exported {len(meta)} meshes, {total} triangles")

    # render instances: place props at every instance frame (not exported)
    for mesh_name, flist in frames.items():
        if mesh_name not in objs:
            continue
        for k, F in enumerate(flist):
            inst = bpy.data.objects.new(f"{mesh_name}__inst{k}", objs[mesh_name].data)
            inst.matrix_world = [list(r) for r in F.T]
            inst.matrix_world = __import__("mathutils").Matrix([list(r) for r in F])
            col_show.objects.link(inst)
        objs[mesh_name].hide_render = True
        objs[mesh_name].hide_viewport = True
    # the elevated-cab riser is not part of the default configuration
    objs["fd25_cab_riser"].hide_render = True

    with open(os.path.join(BUILD, "meshes.json"), "w") as f:
        json.dump(meta, f, indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(BUILD, "fd25.blend"))
    return meta


if __name__ == "__main__":
    main()
