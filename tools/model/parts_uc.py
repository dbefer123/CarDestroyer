"""Undercarriage meshes: frames, centre body, slew ring, rollers, track pieces, sprockets, idlers.

All geometry is produced in world (spawn) coordinates.  Prop meshes (track
runs/arcs, sprockets, idlers) are also built in world space; the exporter
converts them into their prop-local frames.
"""
import math

import numpy as np

from tools.spec import machine as M
from tools.jbeam import meshes as MESH
from tools.model.geo import (Part, merge, box, rounded_box, cylinder, tube, prism, loft_rings, rounded_rect,
                             frame_z, mat4, rot_x, rot_y, rot_z, bolt_circle, sweep, catmull, hexbolt, place,
                             boss, plate_on_frame, frame_from, arc_points, offset_polyline)

U = M.UC
UCM = "fd25_undercarriage"
STEEL = "fd25_steel"
TRACK = "fd25_track"
RUB = "fd25_rubber"


def track_frame(sx):
    """One track frame (box beam with sloped ends, guards, motor housing, idler yoke)."""
    xc = sx * U.track_x
    parts = []
    # main box: loft of rounded rectangles along y
    ys = [U.frame_y0 - 0.05, U.frame_y0 + 0.25, U.frame_y1 - 0.25, U.frame_y1 + 0.05]
    hs = [0.26, 0.40, 0.40, 0.30]
    rings = []
    for y, h in zip(ys, hs):
        zc = (U.frame_z0 + U.frame_z1) / 2 + (0.40 - h) * 0.15
        rr = rounded_rect(U.frame_w, h, 0.03, 2)
        rings.append(np.c_[xc + rr[:, 0], np.full(len(rr), y), zc + rr[:, 1]])
    parts.append(loft_rings(rings, mat=UCM))
    # top cover plate & track guard rails
    parts.append(box((U.frame_w + 0.10, U.frame_y1 - U.frame_y0 - 0.2, 0.025),
                     (xc, (U.frame_y0 + U.frame_y1) / 2, U.frame_z1 + 0.012), UCM, chamfer=0.008))
    # lower track guides (rock guards) along bottom
    for gx in (-0.13, 0.13):
        parts.append(box((0.03, 2.6, 0.08), (xc + gx, 0.0, U.frame_z0 - 0.03), UCM, chamfer=0.006))
    # idler yoke / recoil housing at the front
    parts.append(box((0.26, 0.42, 0.24), (xc, U.idler_y + 0.32, U.idler_z), UCM, chamfer=0.02))
    # final drive (travel motor) housing at the rear
    fd = cylinder(0.27, 0.30, 36, axis="x", center=(xc + sx * 0.05, U.sprocket_y, U.sprocket_z + 0.013),
                  mat=UCM, bevel=0.02)
    parts.append(fd)
    parts.append(bolt_circle((xc + sx * 0.205, U.sprocket_y, U.sprocket_z + 0.013), (sx, 0, 0), 0.20, 14,
                             size=0.028, mat=STEEL))
    parts.append(cylinder(0.12, 0.10, 24, axis="x", center=(xc + sx * 0.25, U.sprocket_y, U.sprocket_z + 0.013),
                          mat=UCM, bevel=0.01))
    # bottom rollers (7), carrier rollers (2)
    rollers = []
    span = np.linspace(-1.30, 1.30, U.n_bottom_rollers)
    for y in span:
        rollers.append(roller((xc, y, U.bottom_run_z + 0.065 + U.bottom_roller_r), U.bottom_roller_r))
        # roller bracket
        parts.append(box((0.30, 0.16, 0.10), (xc, y, U.frame_z0 - 0.02), UCM, chamfer=0.01))
    for y in (-0.70, 0.70):
        rollers.append(roller((xc, y, 0.815 - 0.065 - U.carrier_roller_r), U.carrier_roller_r, width=0.16))
        parts.append(box((0.10, 0.12, 0.20), (xc - sx * 0.12, y, 0.66), UCM, chamfer=0.01))
    return merge(parts), merge(rollers)


def roller(center, r, width=0.22):
    c = np.asarray(center)
    parts = [cylinder(r, width * 0.6, 24, axis="x", center=c, mat=STEEL, bevel=0.006)]
    for fx in (-1, 1):
        parts.append(cylinder(r + 0.025, 0.03, 24, axis="x", center=c + np.array([fx * width * 0.36, 0, 0]),
                              mat=STEEL, bevel=0.005))
        parts.append(cylinder(r * 0.55, 0.05, 16, axis="x", center=c + np.array([fx * (width * 0.5 + 0.02), 0, 0]),
                              mat=UCM, bevel=0.005))
    return merge(parts)


def centre_body():
    parts = []
    # X-frame: two diagonal box legs + central drum
    for sy in (-1, 1):
        for sx in (-1, 1):
            a = np.array([sx * 0.32, sy * 0.30, 0.0])
            b = np.array([sx * 1.03, sy * 0.95, 0.0])
            d = b - a
            L = np.linalg.norm(d)
            ang = math.atan2(d[1], d[0])
            leg = rounded_box((L, 0.42, 0.40), (0, 0, 0), 0.035, 2, UCM)
            parts.append(leg.transform(mat4(rot_z(ang), (a + b) / 2 + np.array([0, 0, 0.66]))))
    parts.append(cylinder(0.66, 0.46, 48, center=(0, 0, 0.68), mat=UCM, bevel=0.02))
    # slew ring (outer race + bolts + gear teeth hint)
    parts.append(tube(0.74, 0.50, 0.10, 64, center=(0, 0, 0.955), mat=STEEL))
    parts.append(bolt_circle((0, 0, 1.005), (0, 0, 1), 0.70, 36, size=0.03, mat=STEEL))
    # swivel joint (centre)
    parts.append(cylinder(0.10, 0.36, 20, center=(0, 0, 0.98), mat=STEEL, bevel=0.01))
    # belly guard
    parts.append(box((1.10, 1.70, 0.03), (0, 0, U.cb_z0 + 0.02), UCM, chamfer=0.008))
    return merge(parts)


def clamp_mounts():
    Cl = M.CLAMP
    parts = []
    for sx in (-1, 1):
        x = sx * Cl.pivot_x
        y = -Cl.pivot[0]
        z = Cl.pivot[1]
        # two side plates per pivot forming a clevis
        for dx in (-0.09, 0.09):
            poly = [(-0.95 + 0.0, -0.32), (0.10, -0.25), (0.20, 0.0), (0.10, 0.20), (-0.75, 0.40), (-1.05, 0.30)]
            M_ = frame_from(np.array([x + dx, y, z]), (0, 1, 0), (0, 0, 1))
            # plate in the (Y,Z) plane: frame X=+Y world, Y=+Z world, Z = +X? (right-handed)
            pl = prism(np.array(poly), 0.035, mat=UCM)
            parts.append(pl.transform(frame_from(np.array([x + dx, y, z]), (0, 1, 0), (0, 0, 1))))
        parts.append(boss((x, y, z), (1, 0, 0), 0.10, 0.26, r_in=0.05, mat=UCM))
        # cylinder base clevis
        bx, by, bz = sx * Cl.cyl_x, -Cl.cyl_base[0], Cl.cyl_base[1]
        parts.append(boss((bx, by, bz), (1, 0, 0), 0.075, 0.20, r_in=0.04, mat=UCM))
        parts.append(box((0.22, 0.30, 0.18), (bx, by + 0.12, bz - 0.12), UCM, chamfer=0.015))
    # front cross plate joining the mounts to the centre body
    parts.append(rounded_box((1.50, 0.70, 0.50), (0, -1.30, 0.60), 0.03, 2, UCM))
    return merge(parts)


def undercarriage_meshes():
    frames, rollers = [], []
    for sx in (1, -1):
        f, r = track_frame(sx)
        frames.append(f)
        rollers.append(r)
    frame = merge(frames + [centre_body(), clamp_mounts()])
    return {"fd25_uc_frame": frame, "fd25_uc_rollers": merge(rollers)}


# ----------------------------------------------------------------------------- tracks
def shoe():
    """Single track shoe + link segment.  Local: X lateral, Y along travel, Z outward; origin on pitch line."""
    p = U.shoe_pitch
    w = U.shoe_width
    t = U.shoe_plate_t
    z_in = 0.050  # plate inner face above pitch line
    parts = []
    # plate with slightly bevelled edges
    plate = rounded_box((w, p * 0.97, t), (0, 0, z_in + t / 2), 0.004, 1, TRACK)
    parts.append(plate)
    # single grouser bar
    gpoly = np.array([(-0.022, 0.0), (0.022, 0.0), (0.010, U.grouser_h), (-0.010, U.grouser_h)])
    gr = prism(gpoly, w * 0.98, mat=TRACK).transform(mat4(rot_y(math.pi / 2) @ rot_x(0)))
    # prism extrudes along Z; rotate so extrusion is along X and the profile lies in (Y,Z)
    gr = prism(gpoly, w * 0.98, mat=TRACK)
    gv = gr.v.copy()
    gr.v = np.c_[gv[:, 2], gv[:, 0] + p * 0.18, gv[:, 1] + z_in + t]
    parts.append(gr)
    # bolts (4) on the outer face
    for bx in (-0.13, 0.13):
        for by in (-0.04, 0.03):
            b = hexbolt(0.026, 0.012, mat=STEEL)
            parts.append(b.translate((bx, by - 0.02, z_in + t)))
    # links (two rails) and pin bosses
    for lx in (-0.11, 0.11):
        poly = np.array([(-p / 2 - 0.02, -0.035), (p / 2 + 0.02, -0.035), (p / 2 + 0.02, 0.030),
                         (0.03, z_in), (-0.03, z_in), (-p / 2 - 0.02, 0.030)])
        link = prism(poly, 0.045, mat=TRACK)
        lv = link.v.copy()
        link.v = np.c_[lv[:, 2] + lx, lv[:, 0], lv[:, 1]]
        parts.append(link)
    parts.append(cylinder(0.026, 0.30, 12, axis="x", center=(0, p / 2, 0.0), mat=STEEL))
    # shared local UVs: every placed copy samples the same single-shoe texture
    return merge(parts).box_uv(scale=1.6, offset=(0.5, 0.5))


def track_path_world(sx):
    """Return list of (position, tangent, outward normal) at pitch spacing for each track piece."""
    xc = sx * U.track_x
    p = U.shoe_pitch
    ri = U.idler_r
    rs = MESH.SPROCKET_PITCH_R
    ci = np.array([U.idler_y, U.idler_z])
    cs = np.array([U.sprocket_y, U.bottom_run_z + rs])
    pieces = {}
    # bottom run: from below idler centre to below sprocket centre, z = bottom_run_z
    n_b = int((cs[0] - ci[0]) / p) + 1
    pieces["bottom"] = [((xc, ci[0] + k * p, U.bottom_run_z), (0, 1, 0), (0, 0, -1)) for k in range(n_b)]
    # top run: from sprocket top towards idler top
    top_s = np.array([cs[0], cs[1] + rs])
    top_i = np.array([ci[0], ci[1] + ri])
    d = top_i - top_s
    L = np.linalg.norm(d)
    tdir = d / L
    nrm = np.array([-tdir[1], tdir[0]])
    if nrm[1] < 0:
        nrm = -nrm
    n_t = int(L / p) + 1
    pieces["top"] = []
    for k in range(n_t):
        q = top_s + tdir * k * p
        pieces["top"].append(((xc, q[0], q[1]), (0, tdir[0], tdir[1]), (0, nrm[0], nrm[1])))
    # front arc around the idler: from top (90deg) over the front (180) to bottom (270)
    n_f = int(math.pi * ri / p) + 1
    pieces["front"] = []
    for k in range(n_f + 1):
        a = math.pi / 2 + k * p / ri
        q = ci + ri * np.array([math.cos(a), math.sin(a)])
        tangent = np.array([-math.sin(a), math.cos(a)])  # direction of travel along the arc (CCW in (y,z))
        pieces["front"].append(((xc, q[0], q[1]), (0, tangent[0], tangent[1]), (0, math.cos(a), math.sin(a))))
    # rear arc around the sprocket: from bottom (-90) over the rear (0) to top (90)
    n_r = int(math.pi * rs / p) + 1
    pieces["rear"] = []
    for k in range(n_r + 1):
        a = -math.pi / 2 + k * p / rs
        q = cs + rs * np.array([math.cos(a), math.sin(a)])
        tangent = np.array([-math.sin(a), math.cos(a)])
        pieces["rear"].append(((xc, q[0], q[1]), (0, tangent[0], tangent[1]), (0, math.cos(a), math.sin(a))))
    return pieces


def place_shoes(samples, sh):
    parts = []
    for pos, tan, nrm in samples:
        y = np.asarray(tan, float)
        z = np.asarray(nrm, float)
        x = np.cross(y, z)
        Mx = np.eye(4)
        Mx[:3, 0], Mx[:3, 1], Mx[:3, 2], Mx[:3, 3] = x, y, z, pos
        parts.append(sh.transform(Mx))
    return merge(parts)


def track_meshes():
    """World-space track pieces for the LEFT track (props reuse them on both sides via local frames)."""
    sh = shoe()
    pieces = track_path_world(1)
    return {
        "fd25_track_bottom": place_shoes(pieces["bottom"], sh),
        "fd25_track_top": place_shoes(pieces["top"], sh),
        "fd25_track_front": place_shoes(pieces["front"], sh),
        "fd25_track_rear": place_shoes(pieces["rear"], sh),
    }


def sprocket_mesh():
    """13-tooth sprocket around the +X axis, centred at origin (local)."""
    rs = MESH.SPROCKET_PITCH_R
    n = MESH.SPROCKET_TEETH
    parts = []
    # tooth profile polygon in (y, z)
    pts = []
    for k in range(n):
        a0 = 2 * math.pi * k / n
        for a, r in ((a0 - 0.10, rs - 0.06), (a0 - 0.05, rs + 0.02), (a0, rs + 0.055), (a0 + 0.05, rs + 0.02),
                     (a0 + 0.10, rs - 0.06), (a0 + math.pi / n, rs - 0.075)):
            pts.append((r * math.cos(a), r * math.sin(a)))
    poly = np.array(pts)
    for dx in (-0.05, 0.05):
        disc = prism(poly, 0.04, mat=STEEL)
        dv = disc.v.copy()
        disc.v = np.c_[dv[:, 2] + dx, dv[:, 0], dv[:, 1]]
        parts.append(disc)
    parts.append(cylinder(0.24, 0.16, 36, axis="x", mat="fd25_steel_dark", bevel=0.012))
    parts.append(bolt_circle((0.08, 0, 0), (1, 0, 0), 0.18, 12, size=0.026, mat=STEEL))
    return merge(parts)


def idler_mesh():
    r = U.idler_r - 0.055
    parts = [cylinder(r, 0.14, 40, axis="x", mat=STEEL, bevel=0.01)]
    for dx in (-0.09, 0.09):
        parts.append(cylinder(r + 0.035, 0.035, 40, axis="x", center=(dx, 0, 0), mat=STEEL, bevel=0.006))
    parts.append(cylinder(0.11, 0.26, 20, axis="x", mat="fd25_steel_dark", bevel=0.01))
    # lightening holes as dark recessed discs
    for k in range(6):
        a = 2 * math.pi * k / 6
        c = (0.072, 0.17 * math.cos(a), 0.17 * math.sin(a))
        parts.append(cylinder(0.045, 0.01, 16, axis="x", center=c, mat="fd25_steel_dark"))
        c2 = (-0.072, 0.17 * math.cos(a), 0.17 * math.sin(a))
        parts.append(cylinder(0.045, 0.01, 16, axis="x", center=c2, mat="fd25_steel_dark"))
    return merge(parts)


def prop_meshes_local():
    """Meshes that are modelled directly in their prop-local frame (X = axle)."""
    return {"fd25_sprocket": sprocket_mesh(), "fd25_idler": idler_mesh()}
