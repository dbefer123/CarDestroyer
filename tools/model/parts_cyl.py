"""Hydraulic cylinder props (barrel + rod), modelled in prop-local frames.

Prop frame (BeamNG convention, see tools/jbeam/meshes.py): origin = idRef node,
X = towards idX, Y = towards idY (orthogonalised), Z = X x Y.  The barrel's
origin sits on the base pin and extends along +X; the rod's origin sits on the
rod-end pin and extends along +X towards the base.
"""
import math

import numpy as np

from tools.spec import machine as M
from tools.model.geo import Part, merge, cylinder, rounded_box, box, frame_z, mat4, sweep, hexbolt, tube

STEEL = "fd25_steel"
CHROME = "fd25_chrome"
CYLP = "fd25_cyl_paint"
RUB = "fd25_rubber"


def prop_frame(nodes, ref, idx, idy):
    o = np.array(nodes[ref]["pos"])
    x = np.array(nodes[idx]["pos"]) - o
    x /= np.linalg.norm(x)
    y = np.array(nodes[idy]["pos"]) - o
    y -= x * (y @ x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = x, y, z, o
    return F


def lateral_local(F):
    """World lateral (+X, slew 0) expressed in the prop frame, orthogonal to the cylinder axis."""
    R = F[:3, :3]
    lat = R.T @ np.array([1.0, 0.0, 0.0])
    lat[0] = 0.0
    n = np.linalg.norm(lat)
    return lat / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])


def eye(lat, r, width, mat=CYLP, pin=True):
    M_ = frame_z((0, 0, 0), lat, (1, 0, 0))
    parts = [cylinder(r, width, 24, mat=mat, bevel=0.006).transform(M_)]
    if pin:
        parts.append(cylinder(r * 0.45, width + 0.02, 12, mat=STEEL).transform(M_))
    return merge(parts)


def barrel(spec, Lmin, lat):
    rb = spec.bore / 2 + 0.025
    Lb = Lmin - 0.08
    r_eye = max(0.055, spec.rod / 2 + 0.02)
    w_eye = 0.10 + spec.bore * 0.3
    parts = [eye(lat, r_eye, w_eye)]
    x0 = r_eye * 0.85
    parts.append(cylinder(rb + 0.012, 0.06, 32, axis="x", center=(x0 + 0.03, 0, 0), mat=CYLP, bevel=0.008))
    parts.append(cylinder(rb, Lb - x0 - 0.10, 48, axis="x", center=((x0 + Lb - 0.10) / 2 + 0.03, 0, 0), mat=CYLP))
    parts.append(cylinder(rb + 0.015, 0.09, 32, axis="x", center=(Lb - 0.05, 0, 0), mat=CYLP, bevel=0.01))
    # gland nut + wiper
    parts.append(cylinder(spec.rod / 2 + 0.018, 0.03, 24, axis="x", center=(Lb + 0.012, 0, 0), mat=STEEL, bevel=0.004))
    # ports with hose stubs (on the side facing +Z local)
    up = np.cross(np.array([1.0, 0, 0]), lat)
    for xp in (x0 + 0.12, Lb - 0.16):
        c = np.array([xp, 0, 0]) + up * (rb + 0.02)
        parts.append(rounded_box((0.07, 0.07, 0.05), (0, 0, 0), 0.01, 1, CYLP).transform(frame_z(c, up, (1, 0, 0))))
        hose = [c + up * 0.03, c + up * 0.10 + np.array([0.05, 0, 0]), c + up * 0.16 + np.array([0.12, 0, 0])]
        parts.append(sweep(hose, 0.018, 8, mat=RUB))
    return merge(parts)


def rod(spec, Lmin, lat):
    rr = spec.rod / 2
    Lr = Lmin - 0.12
    r_eye = max(0.055, rr + 0.02)
    w_eye = 0.10 + spec.bore * 0.3
    parts = [eye(lat, r_eye, w_eye, mat=STEEL)]
    x0 = r_eye * 0.7
    parts.append(cylinder(rr, Lr - x0, 40, axis="x", center=((x0 + Lr) / 2, 0, 0), mat=CHROME, bevel=0.004))
    parts.append(cylinder(rr + 0.02, 0.08, 24, axis="x", center=(x0 + 0.02, 0, 0), mat=STEEL, bevel=0.006))
    return merge(parts)


def cylinder_meshes(nodes, props, hydros_by_pair):
    """Return {mesh: Part(local)} for every fd25_cyl_* mesh, using the first prop instance as reference."""
    out = {}
    for pm in props:
        mesh = pm["mesh"]
        if not mesh.startswith("fd25_cyl_") or mesh in out:
            continue
        kind = mesh.split("_")[2]
        spec = M.CYL[kind]
        F = prop_frame(nodes, pm["ref"], pm["idx"], pm["idy"])
        lat = lateral_local(F)
        key = frozenset((pm["ref"], pm["idx"]))
        Lmin = hydros_by_pair.get(key)
        if Lmin is None:
            raise KeyError(f"no hydro for {pm['ref']}-{pm['idx']}")
        out[mesh] = barrel(spec, Lmin, lat) if mesh.endswith("barrel") else rod(spec, Lmin, lat)
    return out
