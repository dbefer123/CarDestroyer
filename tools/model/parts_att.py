"""Car-dismantling shear (bracket, rotator, head, fork jaw, moving jaw) and the front clamp arm."""
import math

import numpy as np

from tools.spec import machine as M
from tools import kinematics as K
from tools.model.geo import (Part, merge, box, rounded_box, cylinder, tube, prism, loft_rings, rounded_rect, mat4,
                             rot_x, rot_y, rot_z, sweep, catmull, hexbolt, bolt_circle, offset_polyline,
                             arc_points, weld_bead)

T = M.ATT
Cl = M.CLAMP
ATT = "fd25_attachment"
STEEL = "fd25_steel"
RUB = "fd25_rubber"
SPAWN = K.solve()
FR = SPAWN["frames"]

# attachment-local coordinates (a, b, x): a down the attachment axis, b forward, x lateral


def ring_axis_cyl(a0, a1, r, segments=40, mat=ATT, bevel=0.01):
    """Cylinder whose axis is the local a-axis (local X)."""
    return cylinder(r, a1 - a0, segments, axis="x", center=((a0 + a1) / 2, 0, 0), mat=mat, bevel=bevel)


def bracket_local():
    half = T.pin_spacing / 2
    parts = []
    # side plates with pin bosses
    for sx in (-1, 1):
        poly = [(-0.16, -half - 0.06), (-0.16, half + 0.06), (0.05, half + 0.17), (0.22, half + 0.10),
                (0.22, -half - 0.10), (0.05, -half - 0.17)]
        parts.append(prism(np.asarray(poly), 0.05, mat=ATT).translate((0, 0, sx * 0.235)))
        for b in (-half, half):
            parts.append(cylinder(0.11, 0.07, 28, axis="z", center=(0.0, b, sx * 0.235), mat=ATT, bevel=0.008))
            parts.append(cylinder(0.065, 0.10, 16, axis="z", center=(0.0, b, sx * 0.235), mat=STEEL, bevel=0.005))
    # top and base plates
    parts.append(rounded_box((0.05, 0.62, 0.44), (-0.12, 0.0, 0.0), 0.015, 2, ATT))
    parts.append(rounded_box((0.05, 0.70, 0.64), (0.20, 0.0, 0.0), 0.015, 2, ATT))
    # gussets
    for b in (-0.30, 0.0, 0.30):
        parts.append(box((0.16, 0.03, 0.40), (0.11, b, 0.0), ATT, 0.006))
    parts.append(bolt_circle((0.235, 0, 0), (1, 0, 0), 0.26, 12, size=0.03, mat=STEEL))
    return merge(parts)


def rotator_local():
    """Stator housing (fixed to bracket)."""
    a0, a1 = T.rotator_a[0], T.ring_a[0]
    parts = [ring_axis_cyl(a0 + 0.02, a1, T.rotator_r, 48, bevel=0.015),
             ring_axis_cyl(a0, a0 + 0.04, T.rotator_r + 0.02, 48, bevel=0.006)]
    # hydraulic motor bump + swivel ports
    parts.append(rounded_box((0.16, 0.18, 0.18), ((a0 + a1) / 2, T.rotator_r + 0.06, 0.0), 0.03, 2, ATT))
    for k, b in enumerate((-0.12, 0.0, 0.12)):
        parts.append(cylinder(0.022, 0.08, 12, axis="y", center=(a0 + 0.08, b, T.rotator_r + 0.02), mat=STEEL))
    parts.append(bolt_circle((a1 - 0.005, 0, 0), (1, 0, 0), T.rotator_r - 0.04, 16, size=0.026, mat=STEEL))
    return merge(parts)


def ring_local():
    """Blue rotating ring (part of the rotating head)."""
    a0, a1 = T.ring_a
    parts = [ring_axis_cyl(a0, a1, T.rotator_r + 0.015, 56, mat="fd25_attachment", bevel=0.01)]
    parts.append(bolt_circle((a0 + 0.002, 0, 0), (-1, 0, 0), T.rotator_r - 0.02, 20, size=0.024, mat=STEEL))
    return merge(parts)


def head_local():
    a0, a1 = T.head_a
    parts = []
    hw, hd = T.head_hw, T.head_hd
    # main body: loft from the round rotator flange to the rectangular head box
    rings = []
    for a, w, d in ((a0, 0.30, 0.30), (a0 + 0.10, hw, hd), (a1 - 0.05, hw, hd), (a1, hw - 0.02, hd - 0.02)):
        rr = rounded_rect(2 * d, 2 * w, 0.05, 3)  # (b, x)
        rings.append(np.c_[np.full(len(rr), a), rr[:, 0], rr[:, 1]])
    parts.append(loft_rings(rings, mat=ATT))
    # side reinforcement plates with jaw pivot bosses
    pa, pb = T.jaw_pivot
    for sx in (-1, 1):
        poly = [(a1 - 0.40, -hd - 0.02), (a1 + 0.02, -hd - 0.04), (pa + 0.18, pb - 0.08), (pa + 0.12, pb + 0.18),
                (a1 - 0.20, hd + 0.02), (a1 - 0.45, hd)]
        parts.append(prism(np.asarray(poly), 0.035, mat=ATT).translate((0, 0, sx * (hw + 0.018))))
        parts.append(cylinder(0.13, 0.08, 32, axis="z", center=(pa, pb, sx * (hw + 0.04)), mat=ATT, bevel=0.008))
        parts.append(cylinder(0.07, 0.10, 18, axis="z", center=(pa, pb, sx * (hw + 0.06)), mat=STEEL, bevel=0.005))
        for k in range(6):
            ang = 2 * math.pi * k / 6
            b_ = hexbolt(0.024, 0.012, STEEL).transform(mat4(np.eye(3) * np.array([1, 1, sx]),
                                                               (pa + 0.10 * math.cos(ang), pb + 0.10 * math.sin(ang), sx * (hw + 0.08))))
            parts.append(b_)
    # jaw cylinder trunnion block and inspection window
    ja, jb = T.jawcyl_base
    parts.append(rounded_box((0.14, 0.12, 0.30), (ja, jb, 0.0), 0.02, 2, ATT))
    parts.append(box((0.30, 0.012, 0.22), ((a0 + a1) / 2 + 0.05, hd + 0.004, 0.0), "fd25_black", 0.003))
    # lifting eye
    parts.append(tube(0.06, 0.035, 0.03, 20, axis="z", center=(a0 + 0.20, -hd - 0.05, 0.0), mat=ATT))
    return merge(parts)


def jaw_line_poly(line, depth0, taper, hook):
    """Plate outline (a, b) following a centre line with tapering depth and a hooked tip."""
    pts = np.asarray(line, float)
    hw = np.linspace(depth0 / 2, depth0 / 2 * (1 - taper), len(pts))
    poly = offset_polyline(pts, hw)
    return poly


def teeth_strip(line, offset_sign, depth0, taper, x_half, n_teeth, height=0.035, mat=STEEL):
    """Serrated teeth along the gripping edge of a plate."""
    pts = np.asarray(line, float)
    parts = []
    L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    for k in range(n_teeth):
        u = (k + 0.5) / n_teeth * L[-1] * 0.85 + L[-1] * 0.05
        i = min(np.searchsorted(L, u), len(pts) - 1)
        i0 = max(i - 1, 0)
        f = (u - L[i0]) / max(L[i] - L[i0], 1e-9)
        c = pts[i0] + (pts[i] - pts[i0]) * f
        t = pts[i] - pts[i0]
        t /= np.linalg.norm(t) + 1e-12
        n = np.array([-t[1], t[0]]) * offset_sign
        hw = depth0 / 2 * (1 - taper * u / L[-1])
        base = c + n * hw
        tri = np.array([base - t * 0.035, base + t * 0.035, base + n * height + t * 0.01])
        tooth = prism(tri, 2 * x_half * 0.9, mat=mat)
        parts.append(tooth)
    return merge(parts)


def fork_local():
    parts = []
    xp = T.fork_gap + T.fork_plate_t / 2
    line = list(T.fork_line)
    for sx in (-1, 1):
        poly = jaw_line_poly(line, T.fork_depth, 0.55, True)
        plate = prism(poly, T.fork_plate_t, mat=ATT).translate((0, 0, sx * xp))
        parts.append(plate)
        # gripping teeth on the inner (-b) edge, wear edge plate
        parts.append(teeth_strip(line, -1, T.fork_depth, 0.55, T.fork_plate_t / 2, 9).translate((0, 0, sx * xp)))
        # hook tip reinforcement
        ta, tb = line[-1]
        parts.append(rounded_box((0.10, 0.10, T.fork_plate_t + 0.02), (ta - 0.03, tb - 0.02, sx * xp), 0.02, 2, STEEL))
    # spacer blocks between the fork plates (root) and cross tube
    a0, b0 = line[0]
    parts.append(rounded_box((0.16, T.fork_depth, 2 * xp + T.fork_plate_t), (a0 - 0.02, b0 + 0.02, 0.0), 0.02, 2, ATT))
    parts.append(cylinder(0.04, 2 * xp + 0.02, 16, axis="z", center=(line[3][0], line[3][1] + 0.06, 0.0), mat=STEEL))
    return merge(parts)


def jaw_local_closed():
    """Moving jaw at jaw angle 0 (closed), head-local (a, b, x)."""
    parts = []
    line = list(T.jaw_line)
    poly = jaw_line_poly(line, T.jaw_depth, 0.60, True)
    parts.append(prism(poly, 2 * T.jaw_plate_half, mat=ATT))
    parts.append(teeth_strip(line, +1, T.jaw_depth, 0.60, T.jaw_plate_half, 8))
    # hook tip and blade insert
    ta, tb = line[-1]
    parts.append(rounded_box((0.11, 0.09, 2 * T.jaw_plate_half + 0.02), (ta - 0.02, tb + 0.01, 0.0), 0.02, 2, STEEL))
    # pivot hub and lever arm with the cylinder eye
    pa, pb = T.jaw_pivot
    la, lb = T.jaw_lever
    parts.append(cylinder(0.12, 2 * T.jaw_plate_half + 0.06, 32, axis="z", center=(pa, pb, 0.0), mat=ATT, bevel=0.008))
    lever = offset_polyline([(pa, pb), (la, lb)], 0.075)
    parts.append(prism(lever, 2 * T.jaw_plate_half + 0.02, mat=ATT))
    parts.append(cylinder(0.075, 2 * T.jaw_plate_half + 0.06, 24, axis="z", center=(la, lb, 0.0), mat=ATT, bevel=0.006))
    return merge(parts)


def jaw_pose_matrix():
    """Rotation about the jaw pivot by the spawn jaw opening (same convention as kinematics)."""
    g = -SPAWN["pose"]["jaw"]
    cg, sg = math.cos(g), math.sin(g)
    pa, pb = T.jaw_pivot
    J = np.eye(4)
    J[0, 0], J[0, 1], J[1, 0], J[1, 1] = cg, -sg, sg, cg
    J[0, 3] = pa - (cg * pa - sg * pb)
    J[1, 3] = pb - (sg * pa + cg * pb)
    return J


# ----------------------------------------------------------------------------- clamp arm
def clamp_local():
    parts = []
    L = Cl.arm_len
    for sx in (-1, 1):
        st = []
        for s in np.linspace(0.05, L, 10):
            xc = sx * (Cl.pivot_x + (Cl.fork_half - Cl.pivot_x) * s / L)
            st.append((s, 0.0, 0.32, 0.17, xc))
        rings = []
        for s, tc, d, w, xc in st:
            rr = rounded_rect(d, w, 0.03, 2)
            rings.append(np.c_[np.full(len(rr), s), tc + rr[:, 0], xc + rr[:, 1]])
        parts.append(loft_rings(rings, mat=ATT))
        # pivot boss
        parts.append(cylinder(0.13, 0.24, 32, axis="z", center=(0.0, 0.0, sx * Cl.pivot_x), mat=ATT, bevel=0.01))
        parts.append(cylinder(0.055, 0.30, 16, axis="z", center=(0.0, 0.0, sx * Cl.pivot_x), mat=STEEL))
        # cylinder rod lug
        ls, lt = Cl.cyl_rod_on_arm
        for dx in (-0.07, 0.07):
            poly = [(ls - 0.30, 0.15), (ls + 0.30, 0.15), (ls + 0.08, lt + 0.06), (ls - 0.08, lt + 0.06)]
            parts.append(prism(np.asarray(poly), 0.03, mat=ATT).translate((0, 0, sx * Cl.cyl_x + dx)))
        parts.append(cylinder(0.075, 0.18, 24, axis="z", center=(ls, lt, sx * Cl.cyl_x), mat=ATT, bevel=0.006))
        # finger (down & hooked back) with teeth
        x0 = sx * Cl.fork_half
        fpoly = [(L - 0.12, 0.10), (L + 0.20, 0.08), (L + 0.22, -0.20), (L + 0.14, -0.16 - Cl.finger_len + 0.02),
                 (L + 0.02, -0.16 - Cl.finger_len), (L - 0.10, -0.16 - Cl.finger_len + 0.12), (L + 0.04, -0.30),
                 (L - 0.12, -0.14)]
        for dx in (-0.045, 0.045):
            parts.append(prism(np.asarray(fpoly), 0.035, mat=ATT).translate((0, 0, x0 + dx)))
        parts.append(rounded_box((0.10, 0.08, 0.13), (L + 0.08, -0.16 - Cl.finger_len + 0.03, x0), 0.02, 2, STEEL))
    # crossbar (tube) and mid cross tube
    parts.append(cylinder(0.11, 2 * Cl.fork_half + 0.1, 32, axis="z", center=(L, 0.0, 0.0), mat=ATT, bevel=0.01))
    parts.append(cylinder(0.07, 2 * Cl.pivot_x + 0.25, 24, axis="z", center=(1.0, 0.0, 0.0), mat=ATT, bevel=0.008))
    # rubber pads under the crossbar to protect car bodies? (steel teeth instead, as in the photos)
    for k in range(5):
        x = -0.48 + k * 0.24
        tooth = prism(np.array([(L - 0.04, -0.10), (L + 0.04, -0.10), (L + 0.0, -0.18)]), 0.05, mat=STEEL)
        parts.append(tooth.translate((0, 0, x)))
    return merge(parts)


def attachment_meshes():
    J = jaw_pose_matrix()
    return {
        "fd25_att_bracket": bracket_local().transform(FR["att"]),
        "fd25_att_rotator": rotator_local().transform(FR["att"]),
        "fd25_att_ring": ring_local().transform(FR["head"]),
        "fd25_att_head": head_local().transform(FR["head"]),
        "fd25_att_fork": fork_local().transform(FR["head"]),
        "fd25_att_jaw": jaw_local_closed().transform(FR["head"] @ J),
    }


def clamp_meshes():
    return {"fd25_clamp": clamp_local().transform(FR["clamp"])}
