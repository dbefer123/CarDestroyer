"""Superstructure and cab meshes (world coordinates at spawn, slew = 0)."""
import math

import numpy as np

from tools.spec import machine as M
from tools.model.geo import (Part, merge, box, rounded_box, cylinder, tube, prism, loft_rings, rounded_rect, mat4,
                             rot_x, rot_y, rot_z, sweep, catmull, hexbolt, bolt_circle, boss, frame_from, frame_z,
                             offset_polyline, arc_points)

S = M.SS
C = M.CAB
A = M.ARM
PU = "fd25_paint_upper"
STEEL = "fd25_steel"
BLACK = "fd25_black"
RUB = "fd25_rubber"
GRATE = "fd25_grating"
GLASS = "fd25_glass"
INT = "fd25_interior"
LAMP = "fd25_lamp"
BEACON = "fd25_beacon"
DECAL = "fd25_decals"


def deck_outline(inset=0.0):
    hw = S.half_w - inset
    y0 = S.front_y + inset
    y1 = S.cw_y0 + 0.02
    ch = 0.12
    return np.array([(hw - ch, y0), (hw, y0 + ch), (hw, y1), (-hw, y1), (-hw, y0 + ch), (-hw + ch, y0)])


def lofted_outline(outline_fn, levels, mat):
    """Stack outlines at (z, inset) levels and bridge them (rounded/chamfered edges)."""
    rings = []
    for z, ins in levels:
        o = outline_fn(ins)
        rings.append(np.c_[o, np.full(len(o), z)])
    return loft_rings(rings, mat=mat)


def cw_outline(inset=0.0, n_arc=28):
    R = S.tail_radius - inset
    hw = 1.30 - inset
    y_arc = math.sqrt(max(R * R - hw * hw, 0.01))
    a0 = math.atan2(y_arc, hw)
    a1 = math.pi - a0
    arc = [(R * math.cos(a), R * math.sin(a)) for a in np.linspace(a0, a1, n_arc)]
    front = S.cw_y0 + inset
    return np.array([(hw, front)] + arc + [(-hw, front)])[::-1]


def deck():
    parts = []
    parts.append(lofted_outline(deck_outline, [(S.frame_z0, 0.04), (S.frame_z0 + 0.03, 0.0),
                                               (S.deck_z - 0.02, 0.0), (S.deck_z, 0.015)], PU))
    # anti-slip walkway plates (left of the cab rear, right side in front of the tanks)
    parts.append(box((0.62, 0.95, 0.012), (-0.10, 0.10, S.deck_z + 0.006), GRATE))
    parts.append(box((0.30, 1.10, 0.012), (1.20, 1.10, S.deck_z + 0.006), GRATE))
    # side skirts / under covers
    for sx in (-1, 1):
        parts.append(box((0.03, 2.9, 0.10), (sx * (S.half_w - 0.015), 0.35, S.frame_z0 - 0.04), PU, 0.006))
    # slew ring upper race and pinion cover
    parts.append(tube(0.72, 0.52, 0.05, 64, center=(0, 0, S.frame_z0 - 0.02), mat=STEEL))
    parts.append(cylinder(0.12, 0.12, 20, center=(0.45, 0.45, S.frame_z0 - 0.05), mat=PU))
    return merge(parts)


def counterweight():
    parts = []
    levels = [(S.cw_z0, 0.05), (S.cw_z0 + 0.04, 0.0), (S.cw_z1 - 0.10, 0.0), (S.cw_z1 - 0.03, 0.04), (S.cw_z1, 0.10)]
    parts.append(lofted_outline(cw_outline, levels, PU))
    # lifting eyes on top
    for sx in (-1, 1):
        parts.append(tube(0.07, 0.04, 0.04, 20, axis="y", center=(sx * 0.70, 2.30, S.cw_z1 + 0.05), mat=PU))
    # rear work light housing (centre) and camera
    parts.append(rounded_box((0.26, 0.10, 0.12), (0.0, S.tail_radius - 0.02, S.cw_z1 + 0.06), 0.02, 2, BLACK))
    parts.append(box((0.20, 0.012, 0.08), (0.0, S.tail_radius + 0.033, S.cw_z1 + 0.06), LAMP))
    parts.append(rounded_box((0.10, 0.10, 0.08), (0.35, S.tail_radius - 0.06, S.cw_z1 + 0.04), 0.015, 2, BLACK))
    return merge(parts)


def hood():
    parts = []
    y0, y1, ztop = S.hood_y0, S.hood_y1, S.hood_z
    hw = 1.30

    def outline(ins):
        return np.array([(hw - ins, y0 + ins), (hw - ins, y1 - ins), (-hw + ins, y1 - ins), (-hw + ins, y0 + ins)])
    parts.append(lofted_outline(outline, [(S.deck_z, 0.0), (ztop - 0.12, 0.0), (ztop - 0.03, 0.05), (ztop, 0.14)], PU))
    # louvre grilles on the left side and on the top
    for k in range(9):
        z = S.deck_z + 0.30 + k * 0.08
        parts.append(box((0.04, 0.70, 0.022), (hw + 0.01, 1.32, z), BLACK).transform(np.eye(4)))
    for k in range(7):
        y = 0.95 + k * 0.12
        parts.append(box((0.90, 0.05, 0.02), (-0.10, y, ztop + 0.006), BLACK))
    # door seams (dark thin strips) and handles
    for y in (0.75, 1.33, 1.91):
        parts.append(box((0.012, 0.012, 1.0), (hw + 0.006, y, S.deck_z + 0.62), BLACK))
    for y in (1.05, 1.62):
        parts.append(rounded_box((0.03, 0.10, 0.03), (hw + 0.02, y, S.deck_z + 0.75), 0.008, 1, STEEL))
    # hinge pins along the top edge (left side doors) and a radiator grille on the right rear
    for y in (0.90, 1.18, 1.48, 1.76):
        parts.append(cylinder(0.018, 0.10, 12, axis="y", center=(hw + 0.01, y, ztop - 0.13), mat=STEEL))
    gy0, gy1 = 1.20, 1.90
    parts.append(box((0.02, gy1 - gy0 + 0.06, 0.78), (-hw - 0.01, (gy0 + gy1) / 2, S.deck_z + 0.62), BLACK, 0.004))
    for k in range(14):
        z = S.deck_z + 0.27 + k * 0.052
        parts.append(box((0.03, gy1 - gy0, 0.016), (-hw - 0.025, (gy0 + gy1) / 2, z), STEEL))
    # exhaust stack with rain cap, air pre-cleaner
    ex, ey, ez = S.exhaust
    parts.append(cylinder(0.065, ez - ztop, 18, center=(ex, ey, (ez + ztop) / 2), mat=BLACK, bevel=0.004))
    parts.append(cylinder(0.08, 0.06, 18, center=(ex, ey, ztop + 0.03), mat=BLACK))
    parts.append(box((0.16, 0.16, 0.012), (ex, ey + 0.02, ez + 0.02), BLACK))
    parts.append(cylinder(0.11, 0.30, 24, center=(0.55, 1.05, ztop + 0.15), mat=BLACK, bevel=0.02))
    parts.append(cylinder(0.13, 0.05, 24, center=(0.55, 1.05, ztop + 0.32), mat=BLACK, bevel=0.01))
    # handrail on the hood (right side)
    rail = [(-1.20, 0.80, ztop + 0.0), (-1.20, 0.82, ztop + 0.18), (-1.20, 1.85, ztop + 0.18), (-1.20, 1.87, ztop)]
    parts.append(sweep(catmull(rail, 5), 0.018, 8, mat=PU))
    return merge(parts)


def tanks():
    parts = []
    x0, x1, y0, y1, zt = S.tank_x0, S.tank_x1, S.tank_y0, S.tank_y1, S.tank_z

    def outline(ins):
        return np.array([(x1 - ins, y0 + ins), (x1 - ins, y1 - ins), (x0 + ins, y1 - ins), (x0 + ins, y0 + ins)])
    parts.append(lofted_outline(outline, [(S.deck_z, 0.0), (zt - 0.06, 0.0), (zt, 0.05)], PU))
    parts.append(box((x1 - x0 - 0.14, y1 - y0 - 0.14, 0.012), ((x0 + x1) / 2, (y0 + y1) / 2, zt + 0.006), GRATE))
    # caps, sight glass, breather
    parts.append(cylinder(0.075, 0.06, 20, center=(-1.10, 0.30, zt + 0.03), mat=STEEL, bevel=0.008))
    parts.append(cylinder(0.06, 0.05, 20, center=(-0.75, -0.80, zt + 0.025), mat=STEEL, bevel=0.008))
    parts.append(box((0.012, 0.06, 0.25), (x0 - 0.006, 0.20, S.deck_z + 0.55), GLASS))
    # side door seam and handle (tool box at the front)
    parts.append(box((0.012, 0.012, 0.70), (x0 - 0.006, -0.40, S.deck_z + 0.45), BLACK))
    parts.append(rounded_box((0.03, 0.10, 0.03), (x0 - 0.02, -0.60, S.deck_z + 0.55), 0.008, 1, STEEL))
    # steps at the right front corner
    for k, z in enumerate((S.deck_z - 0.38, S.deck_z - 0.75)):
        parts.append(box((0.32, 0.20, 0.03), (-1.20, -1.05 + 0.05 * k, z), GRATE))
        parts.append(box((0.02, 0.20, 0.40), (-1.36, -1.05, z + 0.18), PU))
    # handrails around the tank top
    rail = [(-1.33, y0 + 0.05, zt), (-1.33, y0 + 0.07, zt + 0.55), (-1.33, y1 - 0.20, zt + 0.55), (-1.33, y1 - 0.18, zt)]
    parts.append(sweep(catmull(rail, 5), 0.018, 8, mat=PU))
    return merge(parts)


def boomfoot():
    parts = []
    xa = A.xa
    for sx in (-1, 1):
        x = xa + sx * 0.40
        poly = np.array([(-1.27, S.deck_z), (-0.30, S.deck_z), (-0.38, 1.70), (-0.62, 2.12), (-0.98, 2.14),
                         (-1.16, 1.95), (-1.27, 1.45)])
        # plate polygon lives in (y, z); build in a frame with X = +Y world, Y = +Z world, Z = X world
        Mf = np.array([[0, 0, 1, x], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
        parts.append(prism(poly, 0.05, mat=PU).transform(Mf))
        parts.append(cylinder(0.21, 0.11, 32, axis="x", center=(xa + sx * 0.33, -A.boom_foot[0], A.boom_foot[1]), mat=PU,
                              bevel=0.01))
        # boom cylinder base clevis
        bx = xa + sx * A.boomcyl_x
        parts.append(cylinder(0.10, 0.20, 24, axis="x", center=(bx, -A.boomcyl_base[0], A.boomcyl_base[1]), mat=PU, bevel=0.008))
        parts.append(cylinder(0.05, 0.24, 14, axis="x", center=(bx, -A.boomcyl_base[0], A.boomcyl_base[1]), mat=STEEL))
        parts.append(box((0.22, 0.26, 0.10), (bx, -A.boomcyl_base[0] + 0.05, A.boomcyl_base[1] - 0.10), PU, 0.01))
    parts.append(cylinder(0.08, 0.95, 18, axis="x", center=(xa, -A.boom_foot[0], A.boom_foot[1]), mat=STEEL))
    # control valve cover between cab and tanks (behind the boom foot)
    parts.append(rounded_box((0.86, 0.95, 0.58), (-0.06, 0.20, S.deck_z + 0.29), 0.04, 2, PU))
    return merge(parts)


def details():
    parts = []
    # mirrors: right front (on tank rail) and left front (cab pillar)
    for (x, y, z) in ((-1.34, -1.10, 2.70), (1.42, -1.15, 2.55)):
        parts.append(sweep([(x, y + 0.05, z - 0.55), (x, y + 0.02, z - 0.10), (x, y, z)], 0.012, 6, mat=BLACK))
        parts.append(rounded_box((0.04, 0.20, 0.30), (x, y - 0.02, z + 0.10), 0.015, 2, BLACK))
        parts.append(box((0.006, 0.17, 0.26), (x - 0.02 if x > 0 else x + 0.02, y - 0.03, z + 0.10), GLASS))
    # grease lines along the deck
    parts.append(sweep([(-0.40, -0.30, 1.20), (-0.35, -0.70, 1.22), (A.xa - 0.30, -0.85, 1.70)], 0.01, 6, mat=STEEL))
    # horn under the front deck
    parts.append(cylinder(0.05, 0.10, 16, axis="y", center=(0.95, -1.18, 1.05), mat=BLACK))
    return merge(parts)


def decals_ss():
    """Decal planes (explicit UVs into the decal atlas)."""
    parts = []
    eps = 0.004
    # counterweight sides: brand
    for sx in (-1, 1):
        x = sx * (1.30 + eps)
        quad = np.array([(x, 1.93, 1.55), (x, 2.31, 1.55), (x, 2.31, 1.90), (x, 1.93, 1.90)])
        f = (0, 1, 2, 3) if sx > 0 else (3, 2, 1, 0)
        # left side (+X): a viewer's right is +Y, so U runs with +Y
        uv = [(0.0, 0.75), (1.0, 0.75), (1.0, 1.0), (0.0, 1.0)]
        p = Part(quad, [tuple(range(4))], [DECAL], [uv])
        if sx < 0:
            p = Part(quad, [(3, 2, 1, 0)], [DECAL], [[(1.0, 1.0), (0.0, 1.0), (0.0, 0.75), (1.0, 0.75)]])
        parts.append(p)
    # counterweight rear: hazard stripes band following the arc
    R = S.tail_radius + eps
    z0, z1 = 1.20, 1.42
    angs = np.linspace(math.radians(62), math.radians(118), 12)
    for i in range(len(angs) - 1):
        a, b = angs[i], angs[i + 1]
        q = np.array([(R * math.cos(b), R * math.sin(b), z0), (R * math.cos(a), R * math.sin(a), z0),
                      (R * math.cos(a), R * math.sin(a), z1), (R * math.cos(b), R * math.sin(b), z1)])
        u0, u1 = i / (len(angs) - 1), (i + 1) / (len(angs) - 1)
        parts.append(Part(q, [(0, 1, 2, 3)], [DECAL], [[(u0, 0.50), (u1, 0.50), (u1, 0.60), (u0, 0.60)]]))
    # model badge on the counterweight rear top centre
    q = np.array([(0.35, R, 1.70), (-0.35, R, 1.70), (-0.35, R, 1.95), (0.35, R, 1.95)])
    parts.append(Part(q, [(0, 1, 2, 3)], [DECAL], [[(0.0, 0.62), (0.5, 0.62), (0.5, 0.74), (0.0, 0.74)]]))
    return merge(parts)


def superstructure_meshes():
    return {
        "fd25_ss_frame": deck(),
        "fd25_ss_counterweight": counterweight(),
        "fd25_ss_hood": hood(),
        "fd25_ss_tanks": tanks(),
        "fd25_ss_details": details(),
        "fd25_ss_boomfoot": boomfoot(),
        "fd25_ss_decals": decals_ss(),
    }


# ============================================================================= cab
def cab_shell():
    parts = []
    x0, x1, y0, y1, z0, z1 = C.x0, C.x1, C.y0, C.y1, C.z0 + 0.04, C.z1
    t = 0.05  # pillar size
    # floor pan / base box
    parts.append(rounded_box((x1 - x0, y1 - y0, 0.14), ((x0 + x1) / 2, (y0 + y1) / 2, z0 + 0.07), 0.02, 2, PU))
    # roof
    parts.append(rounded_box((x1 - x0 + 0.04, y1 - y0 + 0.06, 0.10), ((x0 + x1) / 2, (y0 + y1) / 2 + 0.01, z1 - 0.05),
                             0.04, 3, PU))
    # pillars: front (slightly raked), mid (door rear), rear
    front_top_y = y0 + 0.10
    for x in (x0 + t / 2, x1 - t / 2):
        parts.append(sweep([(x, y0 + t / 2, z0 + 0.12), (x, front_top_y, z1 - 0.08)], t * 0.6, 6, mat=PU))
        parts.append(box((t, t, z1 - z0 - 0.18), (x, y1 - t / 2, (z0 + z1) / 2), PU, 0.01))
        parts.append(box((t, t, z1 - z0 - 0.18), (x, -0.15, (z0 + z1) / 2), PU, 0.01))
    # rear wall panel and lower side panels
    parts.append(box((x1 - x0, 0.03, z1 - z0 - 0.2), ((x0 + x1) / 2, y1 - 0.015, (z0 + z1) / 2), PU, 0.008))
    for x in (x0 + 0.015, x1 - 0.015):
        parts.append(box((0.03, y1 - y0 - 0.06, 0.40), (x, (y0 + y1) / 2, z0 + 0.33), PU, 0.008))
    # door handle & hinges (left side = +X)
    parts.append(rounded_box((0.03, 0.14, 0.035), (x1 + 0.015, -0.30, z0 + 1.0), 0.008, 1, STEEL))
    for z in (z0 + 0.45, z0 + 1.35):
        parts.append(cylinder(0.022, 0.08, 10, center=(x1 + 0.01, -1.10, z), mat=BLACK))
    # wiper
    parts.append(box((0.012, 0.012, 0.70), (0.90, y0 - 0.01, z0 + 1.05), BLACK).transform(np.eye(4)))
    # roof work lamps (2) and beacon base
    for x in (x0 + 0.12, x1 - 0.10):
        parts.append(rounded_box((0.16, 0.10, 0.13), (x, y0 + 0.02, z1 - 0.02), 0.02, 2, BLACK))
        parts.append(box((0.13, 0.012, 0.10), (x, y0 - 0.034, z1 - 0.02), LAMP))
    parts.append(cylinder(0.07, 0.04, 20, center=((x0 + x1) / 2, y1 - 0.25, z1 + 0.02), mat=BLACK))
    parts.append(cylinder(0.06, 0.13, 20, center=((x0 + x1) / 2, y1 - 0.25, z1 + 0.10), mat=BEACON, radius2=0.045))
    # cab decal: model name on the rear side panel
    quad = np.array([(x1 + 0.004, 0.40, z0 + 0.25), (x1 + 0.004, 0.02, z0 + 0.25), (x1 + 0.004, 0.02, z0 + 0.43),
                     (x1 + 0.004, 0.40, z0 + 0.43)])
    # facing +X: U must increase towards +Y to read left-to-right
    parts.append(Part(quad, [(3, 2, 1, 0)], [DECAL], [[(1.0, 0.74), (0.5, 0.74), (0.5, 0.62), (1.0, 0.62)]]))
    return merge(parts)


def cab_glass():
    parts = []
    x0, x1, y0, y1, z0, z1 = C.x0, C.x1, C.y0, C.y1, C.z0 + 0.04, C.z1
    e = 0.01
    # windshield (front), raked: quad
    q = np.array([(x1 - 0.05, y0 + 0.015, z0 + 0.20), (x0 + 0.05, y0 + 0.015, z0 + 0.20),
                  (x0 + 0.05, y0 + 0.10, z1 - 0.10), (x1 - 0.05, y0 + 0.10, z1 - 0.10)])
    parts.append(Part(q, [(0, 1, 2, 3)], [GLASS]))
    # side glasses (left door + right side), rear glass, roof window
    for x, flip in ((x1 + e, False), (x0 - e, True)):
        q = np.array([(x, y0 + 0.12, z0 + 0.45), (x, y1 - 0.06, z0 + 0.45), (x, y1 - 0.06, z1 - 0.12), (x, y0 + 0.14, z1 - 0.12)])
        f = (0, 1, 2, 3) if not flip else (3, 2, 1, 0)
        parts.append(Part(q, [f], [GLASS]))
    q = np.array([(x0 + 0.06, y1 + e, z0 + 0.70), (x1 - 0.06, y1 + e, z0 + 0.70), (x1 - 0.06, y1 + e, z1 - 0.12),
                  (x0 + 0.06, y1 + e, z1 - 0.12)])
    parts.append(Part(q, [(0, 1, 2, 3)], [GLASS]))
    q = np.array([(x0 + 0.15, y0 + 0.25, z1 + 0.003), (x1 - 0.15, y0 + 0.25, z1 + 0.003), (x1 - 0.15, y0 + 0.85, z1 + 0.003),
                  (x0 + 0.15, y0 + 0.85, z1 + 0.003)])
    parts.append(Part(q, [(0, 1, 2, 3)], [GLASS]))
    # rubber seals around every pane
    seals = []
    for g in parts:
        q = g.v
        loop = [q[0], q[1], q[2], q[3], q[0]]
        seals.append(sweep(loop, 0.011, 6, mat=RUB, cap=False))
    # give the glass planes a back face too (visible from inside)
    out = merge(parts)
    back = Part(out.v.copy(), [tuple(reversed(f)) for f in out.f], ["fd25_glass_int"] * len(out.f))
    return merge([out, back] + seals)


def cab_guard():
    """Front debris guard: slim perimeter frame with three vertical bars (keeps the view clear)."""
    parts = []
    x0, x1, y0, z0, z1 = C.x0, C.x1, C.y0, C.z0 + 0.04, C.z1
    yg = y0 - 0.10
    frame = [(x0 + 0.02, yg, z0 + 0.25), (x0 + 0.02, yg + 0.06, z1 - 0.05), (x1 - 0.02, yg + 0.06, z1 - 0.05),
             (x1 - 0.02, yg, z0 + 0.25)]
    for i in range(4):
        parts.append(sweep([frame[i], frame[(i + 1) % 4]], 0.018, 10, mat=BLACK))
    for k in range(3):
        x = x0 + 0.20 + k * (x1 - x0 - 0.40) / 2
        parts.append(sweep([(x, yg, z0 + 0.25), (x, yg + 0.06, z1 - 0.05)], 0.009, 8, mat=BLACK))
    parts.append(sweep([(x0 + 0.02, yg + 0.015, z0 + 0.75), (x1 - 0.02, yg + 0.015, z0 + 0.75)], 0.009, 8, mat=BLACK))
    for x in (x0 + 0.02, x1 - 0.02):
        for z in (z0 + 0.25, z1 - 0.05):
            parts.append(rounded_box((0.05, 0.12, 0.05), (x, yg + 0.06, z), 0.01, 2, BLACK))
            parts.append(hexbolt(0.024, 0.012, STEEL).transform(frame_z((x, yg + 0.12, z), (0, 1, 0))))
    return merge(parts)


def cab_interior():
    parts = []
    cx = 0.92
    zf = C.z0 + 0.18  # floor top
    # floor mat
    parts.append(box((C.x1 - C.x0 - 0.10, C.y1 - C.y0 - 0.10, 0.02), (cx, (C.y0 + C.y1) / 2, zf), INT))
    # seat: cushion, backrest, headrest, suspension base
    parts.append(rounded_box((0.30, 0.30, 0.25), (cx, 0.05, zf + 0.13), 0.02, 2, BLACK))
    parts.append(rounded_box((0.50, 0.50, 0.12), (cx, -0.02, zf + 0.33), 0.05, 3, INT))
    parts.append(rounded_box((0.48, 0.12, 0.62), (cx, 0.24, zf + 0.68), 0.05, 3, INT))
    parts.append(rounded_box((0.26, 0.08, 0.18), (cx, 0.27, zf + 1.07), 0.04, 3, INT))
    # consoles + joysticks
    for sx in (-1, 1):
        x = cx + sx * 0.36
        parts.append(rounded_box((0.16, 0.55, 0.36), (x, 0.0, zf + 0.30), 0.03, 2, INT))
        parts.append(cylinder(0.025, 0.10, 12, center=(x, -0.16, zf + 0.53), mat=BLACK))
        parts.append(rounded_box((0.06, 0.06, 0.14), (x, -0.16, zf + 0.64), 0.025, 3, BLACK))
        parts.append(rounded_box((0.14, 0.22, 0.03), (x, 0.06, zf + 0.49), 0.01, 1, INT))
    # travel pedals/levers in front of the seat
    for dx in (-0.07, 0.07):
        parts.append(box((0.08, 0.18, 0.02), (cx + dx, -0.80, zf + 0.10), BLACK).transform(np.eye(4)))
        parts.append(cylinder(0.012, 0.42, 8, center=(cx + dx, -0.72, zf + 0.30), mat=STEEL))
        parts.append(rounded_box((0.03, 0.03, 0.10), (cx + dx, -0.72, zf + 0.52), 0.012, 2, BLACK))
    # monitor on the front right pillar
    parts.append(rounded_box((0.06, 0.22, 0.16), (C.x1 - 0.10, -1.05, zf + 1.10), 0.01, 2, BLACK))
    parts.append(box((0.006, 0.18, 0.12), (C.x1 - 0.135, -1.05, zf + 1.10), GLASS))
    # AC unit behind seat and roof liner
    parts.append(rounded_box((0.80, 0.22, 0.40), (cx, C.y1 - 0.15, zf + 0.25), 0.03, 2, INT))
    parts.append(box((C.x1 - C.x0 - 0.08, C.y1 - C.y0 - 0.08, 0.02), (cx, (C.y0 + C.y1) / 2, C.z1 - 0.11), INT))
    return merge(parts)


def cab_riser():
    parts = []
    R = C.riser_elevated
    z0, z1 = M.SS.deck_z, C.z0 + 0.04 + R
    cx, cy = (C.x0 + C.x1) / 2, (C.y0 + C.y1) / 2
    parts.append(rounded_box((C.x1 - C.x0 - 0.10, C.y1 - C.y0 - 0.25, z1 - z0), (cx, cy, (z0 + z1) / 2), 0.04, 2, PU))
    # ladder on the left side
    for dy in (-0.15, 0.15):
        parts.append(sweep([(C.x1 + 0.08, -0.55 + dy, z0), (C.x1 + 0.08, -0.55 + dy, z1 + 0.6)], 0.018, 8, mat=PU))
    for k in range(5):
        z = z0 + 0.2 + k * 0.22
        parts.append(sweep([(C.x1 + 0.08, -0.70, z), (C.x1 + 0.08, -0.40, z)], 0.014, 6, mat=STEEL))
    return merge(parts)


def cab_meshes():
    return {
        "fd25_cab_shell": cab_shell(),
        "fd25_cab_glass": cab_glass(),
        "fd25_cab_interior": cab_interior(),
        "fd25_cab_guard": cab_guard(),
        "fd25_cab_riser": cab_riser(),
    }
