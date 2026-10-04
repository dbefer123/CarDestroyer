"""Boom, stick, linkage and hydraulic line meshes (built in body-local frames, placed by the spawn pose)."""
import math

import numpy as np

from tools.spec import machine as M
from tools import kinematics as K
from tools.model.geo import (Part, merge, box, rounded_box, cylinder, prism, loft_rings, rounded_rect, mat4,
                             rot_x, rot_y, rot_z, sweep, catmull, hexbolt, bolt_circle, boss, frame_z,
                             offset_polyline, arc_points, weld_bead)

A = M.ARM
PAINT = "fd25_paint_arm"
STEEL = "fd25_steel"
RUB = "fd25_rubber"
BLACK = "fd25_black"
LAMP = "fd25_lamp"

SPAWN = K.solve()
FR = SPAWN["frames"]


def local_box_section(stations, mat=PAINT, radius=0.035, seg=3):
    """Loft of rounded rectangles in a body-local (s, t, x) frame.
    stations: list of (s, t_centre, depth, width)."""
    rings = []
    for s, tc, d, w in stations:
        rr = rounded_rect(d, w, radius, seg)  # (t, x)
        rings.append(np.c_[np.full(len(rr), s), tc + rr[:, 0], rr[:, 1]])
    return loft_rings(rings, mat=mat)


def side_plates(poly_st, x_center, thickness, mat=PAINT):
    """Plate in the (s, t) plane at lateral x."""
    p = prism(np.asarray(poly_st, float), thickness, mat=mat)
    return p.translate((0, 0, x_center))


def lboss(s, t, x, r, width, r_pin=None, mat=PAINT, segments=44):
    """Boss around the lateral axis in body-local coordinates."""
    parts = [cylinder(r, width, segments, axis="z", center=(s, t, x), mat=mat, bevel=0.008)]
    if r_pin:
        parts.append(cylinder(r_pin, width + 0.04, 16, axis="z", center=(s, t, x), mat=STEEL, bevel=0.005))
        for sx in (-1, 1):
            # pin retainer plate + bolts
            parts.append(box((0.10, 0.10, 0.012), (s, t + r_pin + 0.03, x + sx * (width / 2 + 0.006)), STEEL, 0.003))
    return merge(parts)


# ----------------------------------------------------------------------------- boom
def boom_center_t(s):
    L, ks, kt = A.boom_len, A.boom_knee_s * A.boom_len, A.boom_knee_t
    s = min(max(s, 0.0), L)
    # smooth knee: blend two straight segments with a short arc
    lin = kt * s / ks if s <= ks else kt * (L - s) / (L - ks)
    blend = 0.45
    if abs(s - ks) < blend:
        a = kt * (ks - blend) / ks
        b = kt * (L - ks - blend) / (L - ks)
        u = (s - (ks - blend)) / (2 * blend)
        # quadratic bezier through a, kt (control), b
        ctrl = kt + 0.06
        return (1 - u) ** 2 * a + 2 * (1 - u) * u * ctrl + u * u * b
    return lin


def boom_depth(s):
    L, ks = A.boom_len, A.boom_knee_s * A.boom_len
    if s <= ks:
        return A.boom_depth_foot + (A.boom_depth_knee - A.boom_depth_foot) * s / ks
    return A.boom_depth_knee + (A.boom_depth_nose - A.boom_depth_knee) * (s - ks) / (L - ks)


def boom_width(s):
    return A.boom_width - 0.06 * s / A.boom_len


def boom_local():
    L = A.boom_len
    parts = []
    st = []
    for s in np.linspace(0.42, 5.36, 40):
        st.append((s, boom_center_t(s), boom_depth(s), boom_width(s)))
    parts.append(local_box_section(st))
    # foot: side cheeks + big boss
    for sx in (-1, 1):
        poly = [(0.0, -0.30), (0.62, -0.20), (0.62, 0.36), (0.20, 0.34), (-0.18, 0.18), (-0.24, -0.05), (-0.12, -0.26)]
        parts.append(side_plates(poly, sx * 0.27, 0.035))
    parts.append(lboss(0.0, 0.0, 0.0, 0.25, 0.62, r_pin=0.09))
    # nose clevis plates around the boom-stick pin
    for sx in (-1, 1):
        poly = [(5.20, -0.22), (5.62, -0.24), (5.88, -0.10), (5.92, 0.08), (5.72, 0.24), (5.32, 0.28), (5.10, 0.20)]
        parts.append(side_plates(poly, sx * 0.258, 0.045))
        parts.append(lboss(L, 0.0, sx * 0.258, 0.23, 0.07))
    parts.append(cylinder(0.075, 0.68, 18, axis="z", center=(L, 0, 0), mat=STEEL, bevel=0.005))
    # boom cylinder rod lugs (underside), stick cylinder base bracket (top of knee)
    s_l, t_l = A.boomcyl_rod_on_boom
    for sx in (-1, 1):
        poly = [(s_l - 0.40, boom_center_t(s_l - 0.40) - boom_depth(s_l - 0.40) / 2 + 0.02),
                (s_l + 0.40, boom_center_t(s_l + 0.40) - boom_depth(s_l + 0.40) / 2 + 0.02),
                (s_l + 0.12, t_l - 0.05), (s_l - 0.12, t_l - 0.05)]
        for dx in (-0.07, 0.07):
            parts.append(side_plates(poly, sx * A.boomcyl_x + dx, 0.03))
        parts.append(lboss(s_l, t_l, sx * A.boomcyl_x, 0.085, 0.18, r_pin=0.045))
    s_b, t_b = A.stickcyl_base_on_boom
    top_t = boom_center_t(s_b) + boom_depth(s_b) / 2
    for dx in (-0.10, 0.10):
        poly = [(s_b - 0.42, top_t - 0.02), (s_b + 0.38, top_t - 0.02), (s_b + 0.10, t_b + 0.08), (s_b - 0.16, t_b + 0.08)]
        parts.append(side_plates(poly, dx, 0.035))
    parts.append(lboss(s_b, t_b, 0.0, 0.09, 0.24, r_pin=0.05))
    # longitudinal stiffener ribs and weld beads along the box corners
    for sx in (-1, 1):
        path = [(s, boom_center_t(s) + boom_depth(s) / 2, sx * boom_width(s) / 2) for s in np.linspace(0.5, 5.3, 12)]
        parts.append(weld_bead(path, 0.008, mat=PAINT))
        path = [(s, boom_center_t(s) - boom_depth(s) / 2, sx * boom_width(s) / 2) for s in np.linspace(0.5, 5.3, 12)]
        parts.append(weld_bead(path, 0.008, mat=PAINT))
    # lifting eye and grease nipples
    parts.append(cylinder(0.05, 0.03, 16, axis="z", center=(2.2, boom_center_t(2.2) + boom_depth(2.2) / 2 + 0.05, 0.0), mat=PAINT))
    return merge(parts)


def boom_lamp_local():
    s = 2.10
    t = boom_center_t(s) + boom_depth(s) / 2
    parts = [box((0.10, 0.06, 0.06), (s, t + 0.03, 0.0), BLACK, 0.008),
             rounded_box((0.10, 0.16, 0.22), (s - 0.02, t + 0.15, 0.0), 0.02, 2, BLACK)]
    lens = box((0.012, 0.12, 0.18), (s - 0.076, t + 0.15, 0.0), LAMP)
    parts.append(lens)
    return merge(parts)


def boom_hoses_local():
    """Steel lines along the boom (top) with clamps, from foot to nose."""
    L = A.boom_len
    parts = []
    for i, dx in enumerate((-0.07, -0.025, 0.025, 0.07)):
        path = []
        for s in np.linspace(0.75, 5.20, 24):
            path.append((s, boom_center_t(s) + boom_depth(s) / 2 + 0.035, dx))
        parts.append(sweep(path, 0.017, 8, mat=STEEL))
    for s in np.linspace(1.0, 5.0, 6):
        t = boom_center_t(s) + boom_depth(s) / 2
        parts.append(box((0.05, 0.05, 0.22), (s, t + 0.03, 0.0), BLACK, 0.006))
    return merge(parts)


# ----------------------------------------------------------------------------- stick
def stick_local():
    parts = []
    st = []
    for s in np.linspace(0.18, 2.18, 14):
        f = (s - 0.18) / 2.0
        st.append((s, 0.04 * (1 - f), 0.46 - 0.08 * f, A.stick_width - 0.06 * f))
    parts.append(local_box_section(st))
    # tail plates: from the boss region up/back to the stick-cylinder lug
    s_t, t_t = A.stick_tail
    s_b, t_b = A.tiltcyl_base_on_stick
    for sx in (-1, 1):
        poly = [(0.30, -0.22), (0.30, 0.28), (s_b + 0.12, t_b + 0.02), (s_t + 0.10, t_t + 0.12),
                (s_t - 0.10, t_t + 0.02), (-0.30, 0.0), (-0.12, -0.22)]
        parts.append(side_plates(poly, sx * 0.20, 0.04))
    parts.append(box((0.6, 0.06, 0.36), (0.0, 0.30, 0.0), PAINT, 0.01).transform(mat4(rot_z(0.6), (-0.15, 0.10, 0))))
    parts.append(lboss(0.0, 0.0, 0.0, 0.20, 0.42, r_pin=0.085))
    parts.append(lboss(s_t, t_t, 0.0, 0.09, 0.26, r_pin=0.05))
    parts.append(lboss(s_b, t_b, 0.0, 0.085, 0.22, r_pin=0.045))
    # tilt-cylinder base bracket plates
    for dx in (-0.08, 0.08):
        poly = [(s_b - 0.25, 0.20), (s_b + 0.35, 0.20), (s_b + 0.10, t_b + 0.06), (s_b - 0.10, t_b + 0.06)]
        parts.append(side_plates(poly, dx, 0.03))
    # nose boss (attachment pin) and side link pivot bosses
    parts.append(lboss(A.stick_len, 0.0, 0.0, 0.17, 0.40, r_pin=0.075))
    ks, kt = A.linkpivot_on_stick
    parts.append(lboss(ks, kt, 0.0, 0.13, 0.42, r_pin=0.065))
    # nose cheek plates
    for sx in (-1, 1):
        poly = [(1.85, -0.18), (2.30, -0.17), (2.52, -0.05), (2.50, 0.08), (2.12, 0.22), (1.80, 0.20)]
        parts.append(side_plates(poly, sx * 0.18, 0.04))
    # wear strip on the underside
    parts.append(box((1.6, 0.025, 0.30), (1.15, -0.215, 0.0), STEEL, 0.004))
    return merge(parts)


def stick_hoses_local():
    parts = []
    for dx in (-0.05, 0.05):
        path = [(s, 0.27 - 0.08 * (s / 2.2), dx) for s in np.linspace(0.30, 2.05, 10)]
        parts.append(sweep(path, 0.016, 8, mat=STEEL))
    return merge(parts)


# ----------------------------------------------------------------------------- linkage
def sidelink_local():
    """Two dog-bone plates from K (origin) along +s to P (s = sidelink_len)."""
    L = A.sidelink_len
    parts = []
    for sx in (-1, 1):
        outline = np.vstack([arc_points((0, 0), 0.10, math.pi / 2, 3 * math.pi / 2, 10),
                             arc_points((L, 0), 0.09, -math.pi / 2, math.pi / 2, 10)])
        p = prism(outline, 0.04, mat=PAINT).translate((0, 0, sx * 0.25))
        parts.append(p)
        parts.append(cylinder(0.075, 0.07, 20, axis="z", center=(0, 0, sx * 0.25), mat=PAINT, bevel=0.006))
        parts.append(cylinder(0.07, 0.07, 20, axis="z", center=(L, 0, sx * 0.25), mat=PAINT, bevel=0.006))
    parts.append(cylinder(0.055, 0.62, 16, axis="z", center=(L, 0, 0), mat=STEEL, bevel=0.005))
    return merge(parts)


def hlink_local():
    """H-link from P (origin) along +s to C."""
    L = A.hlink_len
    parts = []
    for sx in (-1, 1):
        outline = np.vstack([arc_points((0, 0), 0.10, math.pi / 2, 3 * math.pi / 2, 10),
                             arc_points((L, 0), 0.10, -math.pi / 2, math.pi / 2, 10)])
        parts.append(prism(outline, 0.045, mat=PAINT).translate((0, 0, sx * 0.21)))
    parts.append(rounded_box((0.22, 0.10, 0.38), (L * 0.5, 0.0, 0.0), 0.02, 2, PAINT))
    parts.append(cylinder(0.08, 0.36, 20, axis="z", center=(L, 0, 0), mat=PAINT, bevel=0.008))
    parts.append(cylinder(0.055, 0.50, 16, axis="z", center=(L, 0, 0), mat=STEEL, bevel=0.005))
    return merge(parts)


# ----------------------------------------------------------------------------- joints hoses (flex)
def hose_boomfoot_world():
    """Hose bundle from the superstructure valve bank into the boom foot (bends with the joint)."""
    W = SPAWN["W"]
    parts = []
    for i, dx in enumerate((-0.06, 0.0, 0.06)):
        p0 = np.array([A.xa + dx, -0.40, 1.70])
        p1 = np.array([A.xa + dx, -0.55, 2.15])
        p2 = W("boom", 0.55, boom_center_t(0.55) + boom_depth(0.55) / 2 + 0.06, dx)
        p3 = W("boom", 0.85, boom_center_t(0.85) + boom_depth(0.85) / 2 + 0.04, dx)
        parts.append(sweep(catmull([p0, p1, p2, p3], 6), 0.024, 10, mat=RUB))
    return merge(parts)


def hose_boomstick_world():
    W = SPAWN["W"]
    parts = []
    L = A.boom_len
    for dx in (-0.07, -0.025, 0.025, 0.07):
        p0 = W("boom", 5.15, boom_center_t(5.15) + boom_depth(5.15) / 2 + 0.035, dx)
        p1 = W("boom", 5.55, 0.45, dx)
        p2 = W("stick", -0.10, 0.45, dx * 0.8)
        p3 = W("stick", 0.35, 0.30, dx * 0.8)
        parts.append(sweep(catmull([p0, p1, p2, p3], 6), 0.026, 10, mat=RUB))
    return merge(parts)


def hose_stickatt_world():
    """Hoses from the stick nose down to the rotator swivel (stick + bracket groups)."""
    W = SPAWN["W"]
    parts = []
    for dx in (-0.05, 0.05):
        p0 = W("stick", 2.05, 0.21, dx)
        p1 = W("stick", 2.30, 0.30, dx * 2)
        p2 = W("att", 0.05, 0.10, dx * 3)
        p3 = W("att", 0.30, 0.20, dx * 3)
        parts.append(sweep(catmull([p0, p1, p2, p3], 6), 0.022, 10, mat=RUB))
    return merge(parts)


def arm_meshes():
    return {
        "fd25_boom": merge([boom_local(), boom_lamp_local()]).transform(FR["boom"]),
        "fd25_boom_hoses": boom_hoses_local().transform(FR["boom"]),
        "fd25_hose_boomfoot": hose_boomfoot_world(),
        "fd25_stick": merge([stick_local(), stick_hoses_local()]).transform(FR["stick"]),
        "fd25_hose_boomstick": hose_boomstick_world(),
        "fd25_sidelink": sidelink_local().transform(FR["sidelink"]),
        "fd25_hlink": hlink_local().transform(FR["hlink"]),
        "fd25_hose_stickatt": hose_stickatt_world(),
    }
