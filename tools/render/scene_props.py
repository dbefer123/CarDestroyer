"""Render-only scene props (not part of the mod): a stripped sedan wreck for the dismantling shots.

The car lies across the front of the machine, roof edge within the shear's reach; the roof
panel is a separate object so the animation can tear it off.
"""
import math

import numpy as np

from tools.model.geo import Part, merge, rounded_box, cylinder, sweep

CAR_C = np.array([-0.18, -5.95])      # car centre (x, y): length along X, across the machine's front
ROOF_Z = 1.37


def car_body():
    cx, cy = CAR_C
    parts = []
    L, W = 4.40, 1.74
    # lower body: sill/floor, hood and boot decks with slightly lower ends
    parts.append(rounded_box((L, W, 0.50), (cx, cy, 0.62), 0.10, 3, "scene_car_paint"))
    parts.append(rounded_box((1.15, W - 0.06, 0.12), (cx + 1.55, cy, 0.90), 0.05, 2, "scene_car_paint"))   # hood
    parts.append(rounded_box((0.95, W - 0.06, 0.14), (cx - 1.70, cy, 0.92), 0.05, 2, "scene_car_paint"))   # boot
    parts.append(rounded_box((2.05, W - 0.10, 0.10), (cx - 0.15, cy, 0.90), 0.03, 2, "scene_car_dark"))    # tub
    # bumpers
    for sx in (-1, 1):
        parts.append(rounded_box((0.12, W - 0.04, 0.18), (cx + sx * (L / 2 + 0.03), cy, 0.52), 0.05, 2, "scene_car_dark"))
    # wheels (rubber + rim)
    for x in (cx + 1.38, cx - 1.42):
        for y in (cy + W / 2 - 0.10, cy - W / 2 + 0.10):
            parts.append(cylinder(0.31, 0.21, 28, axis="y", center=(x, y, 0.31), mat="scene_tire", bevel=0.03))
            parts.append(cylinder(0.19, 0.225, 20, axis="y", center=(x, y, 0.31), mat="scene_rim", bevel=0.01))
    # cabin frame: pillars from the belt line up to the roof (glass and doors already removed)
    belt, roofz = 0.96, ROOF_Z - 0.03
    for side in (-1, 1):
        yb = cy + side * (W / 2 - 0.06)
        yr = cy + side * 0.64
        for xb, xr in ((cx + 1.05, cx + 0.62), (cx - 0.10, cx - 0.12), (cx - 1.25, cx - 0.98)):
            parts.append(sweep([(xb, yb, belt), (xr, yr, roofz)], 0.035, 8, mat="scene_car_paint"))
        parts.append(sweep([(cx + 0.62, yr, roofz), (cx - 0.98, yr, roofz)], 0.03, 8, mat="scene_car_paint"))
    # seats and dashboard inside
    for x in (cx + 0.25, cx - 0.70):
        for y in (cy + 0.38, cy - 0.38):
            parts.append(rounded_box((0.50, 0.50, 0.16), (x, y, 0.98), 0.05, 2, "scene_car_seat"))
            parts.append(rounded_box((0.12, 0.48, 0.34), (x - 0.24, y, 1.15), 0.05, 2, "scene_car_seat"))
    parts.append(rounded_box((0.30, W - 0.20, 0.22), (cx + 0.80, cy, 1.08), 0.05, 2, "scene_car_dark"))
    return merge(parts)


def car_roof():
    cx, cy = CAR_C
    roof = rounded_box((1.70, 1.34, 0.035), (cx - 0.18, cy, ROOF_Z), 0.015, 2, "scene_car_paint")
    # slight crown: lift the centre
    v = roof.v.copy()
    r = ((v[:, 0] - (cx - 0.18)) / 0.85) ** 2 + ((v[:, 1] - cy) / 0.67) ** 2
    v[:, 2] += 0.03 * np.clip(1 - r, 0, 1)
    roof.v = v
    return roof


def setup_car_materials(bpy):
    def principled(name, col, metal, rough, rust=None):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
        b.inputs["Metallic"].default_value = metal
        b.inputs["Roughness"].default_value = rough
        if rust is None:
            b.inputs["Base Color"].default_value = (*col, 1)
            return m
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 3.0
        noise.inputs["Detail"].default_value = 10.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.55
        ramp.color_ramp.elements[0].color = (*col, 1)
        ramp.color_ramp.elements[1].position = 0.68
        ramp.color_ramp.elements[1].color = (*rust, 1)
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        return m
    return {
        "scene_car_paint": principled("scene_car_paint", (0.30, 0.035, 0.03), 0.25, 0.55, rust=(0.16, 0.08, 0.04)),
        "scene_car_seat": principled("scene_car_seat", (0.09, 0.08, 0.07), 0.0, 0.9),
        "scene_car_dark": principled("scene_car_dark", (0.025, 0.025, 0.027), 0.0, 0.75),
        "scene_tire": principled("scene_tire", (0.018, 0.018, 0.018), 0.0, 0.85),
        "scene_rim": principled("scene_rim", (0.45, 0.45, 0.46), 0.85, 0.35),
    }
