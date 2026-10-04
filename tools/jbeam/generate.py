"""Generate all FD-25 JBeam parts from the machine spec.

Run:  python -m tools.jbeam.generate  [outdir]

Writes the .jbeam files into  mod/vehicles/ferrovore_fd25/  and node/prop
metadata into  build/  (consumed by the Blender modeller and validators).
"""
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

from tools.spec import machine as M
from tools import kinematics as K
from tools.jbeam.builder import Vehicle, write_jbeam, FLT_MAX, r4
from tools.jbeam import meshes as MESH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)
BUILD = os.path.join(ROOT, "build")

SPAWN = K.solve()
W = SPAWN["W"]

# structural beam classes ----------------------------------------------------
STIFF = dict(beamSpring=4.0e7, beamDamp=4000, beamDeform=FLT_MAX, beamStrength=FLT_MAX)
MEDIUM = dict(beamSpring=1.6e7, beamDamp=1800, beamDeform=FLT_MAX, beamStrength=FLT_MAX)
LIGHT = dict(beamSpring=6.0e6, beamDamp=600, beamDeform=FLT_MAX, beamStrength=FLT_MAX)
JOINT = dict(beamSpring=5.0e7, beamDamp=5000, beamDeform=FLT_MAX, beamStrength=FLT_MAX)
WHEEL = dict(beamSpring=4.0e6, beamDamp=300, beamDeform=FLT_MAX, beamStrength=FLT_MAX)


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


class Body:
    """Group of nodes belonging to one rigid body (mass distributed later)."""

    def __init__(self, part, name, group, mass, **nodeopts):
        self.part, self.name, self.group, self.mass = part, name, group, mass
        self.nodes = []
        self.fixed_weight = {}
        self.nodeopts = nodeopts

    def n(self, name, pos, weight=None, groups=None, **opts):
        o = dict(self.nodeopts)
        o.update(opts)
        g = groups if groups is not None else [self.group]
        o["group"] = g if len(g) > 1 else g[0]
        self.part.node(name, pos, body=self.name, **o)
        self.nodes.append(name)
        if weight is not None:
            self.fixed_weight[name] = weight
        return name

    def finalize_mass(self, veh):
        free = [n for n in self.nodes if n not in self.fixed_weight]
        fixed = sum(self.fixed_weight.values())
        rest = max(self.mass - fixed, 1.0)
        w = rest / max(len(free), 1)
        for n in self.nodes:
            veh.nodes[n].opts["nodeWeight"] = r4(self.fixed_weight.get(n, w))


def ring_beams(part, ring, **o):
    k = len(ring)
    for i in range(k):
        part.beam(ring[i], ring[(i + 1) % k], **o)
    if k == 4:
        part.beam(ring[0], ring[2], **o)
        part.beam(ring[1], ring[3], **o)


def truss(part, rings, chords=True, **o):
    """Rigid box truss through consecutive 4-node cross sections."""
    for r in rings:
        ring_beams(part, r, **o)
    for r0, r1 in zip(rings, rings[1:]):
        for i in range(4):
            j = (i + 1) % 4
            part.beam(r0[i], r1[i], **o)
            part.beam(r0[i], r1[j], **o)
            part.beam(r0[j], r1[i], **o)
        part.beam(r0[0], r1[2], **o)
        part.beam(r0[1], r1[3], **o)
        part.beam(r0[2], r1[0], **o)
        part.beam(r0[3], r1[1], **o)
    # long chords along every corner line: large EA = k*L -> bending stiffness at low node budget
    if chords:
        n = len(rings)
        for i in range(n):
            for j in range(i + 2, n):
                for k in range(4):
                    part.beam(rings[i][k], rings[j][k], **o)


def connect(part, a, targets, **o):
    for t in targets:
        if t != a:
            part.beam(a, t, **o)


def nearest(veh, p, candidates, k):
    p = np.asarray(p)
    c = sorted(candidates, key=lambda n: np.linalg.norm(veh.pos(n) - p))
    return c[:k]


def tri_outward(part, veh, a, b, c, centre, **o):
    pa, pb, pc = veh.pos(a), veh.pos(b), veh.pos(c)
    nrm = np.cross(pb - pa, pc - pa)
    if np.dot(nrm, (pa + pb + pc) / 3 - centre) < 0:
        b, c = c, b
    part.tri(a, b, c, **o)


def skin_rings(part, veh, rings, caps=(True, True), **o):
    """Collision skin over a list of 4-node rings, wound outward."""
    centre = np.mean([veh.pos(n) for r in rings for n in r], axis=0)
    for r0, r1 in zip(rings, rings[1:]):
        for i in range(4):
            j = (i + 1) % 4
            tri_outward(part, veh, r0[i], r0[j], r1[j], centre, **o)
            tri_outward(part, veh, r0[i], r1[j], r1[i], centre, **o)
    if caps[0]:
        r = rings[0]
        tri_outward(part, veh, r[0], r[1], r[2], centre, **o)
        tri_outward(part, veh, r[0], r[2], r[3], centre, **o)
    if caps[1]:
        r = rings[-1]
        tri_outward(part, veh, r[0], r[1], r[2], centre, **o)
        tri_outward(part, veh, r[0], r[2], r[3], centre, **o)


def pw(u, w, x, slew=0.0):
    """Superstructure plane (u, w) + lateral x -> world xyz at spawn."""
    return K.plane_to_world((u, w), x, slew)


# =============================================================================
def build():
    veh = Vehicle()
    bodies = []
    props_meta = []

    # ------------------------------------------------------------------ parts
    p_main = veh.part("ferrovore_fd25_undercarriage", slotType="main",
                      info_name="Ferrovore FD-25 Undercarriage", value=380000)
    p_ss = veh.part("fd25_superstructure", slotType="fd25_superstructure",
                    info_name="FD-25 Superstructure")
    p_cab = veh.part("fd25_cab_standard", slotType="fd25_cab", info_name="Standard Cab (FOPS guard)")
    p_cabe = veh.part("fd25_cab_elevated", slotType="fd25_cab", info_name="Elevated Cab (+0.95 m riser)")
    p_boom = veh.part("fd25_boom_mono", slotType="fd25_boom", info_name="5.7 m Mono Boom")
    p_stick = veh.part("fd25_stick_hd", slotType="fd25_stick", info_name="2.4 m Heavy-Duty Stick")
    p_att = veh.part("fd25_dismantling_shear", slotType="fd25_attachment",
                     info_name="Rotating Car Dismantling Shear")
    p_clamp = veh.part("fd25_clamparm", slotType="fd25_clamparm", info_name="Hydraulic Car Clamp Arm")
    p_noclamp = veh.part("fd25_clamparm_none", slotType="fd25_clamparm", info_name="No Clamp Arm")
    p_hyd = veh.part("fd25_hydraulics_standard", slotType="fd25_hydraulics", info_name="Standard Hydraulics")
    p_hyd_fast = veh.part("fd25_hydraulics_arcade", slotType="fd25_hydraulics", info_name="Arcade Hydraulics (fast)")

    p_main.slot("fd25_superstructure", "fd25_superstructure", "Superstructure", coreSlot=True)
    p_main.slot("fd25_clamparm", "fd25_clamparm", "Front Clamp Arm")
    p_main.slot("fd25_hydraulics", "fd25_hydraulics_standard", "Hydraulic Control System", coreSlot=True)
    p_ss.slot("fd25_cab", "fd25_cab_standard", "Cab", coreSlot=True)
    p_ss.slot("fd25_boom", "fd25_boom_mono", "Boom", coreSlot=True)
    p_boom.slot("fd25_stick", "fd25_stick_hd", "Stick", coreSlot=True)
    p_stick.slot("fd25_attachment", "fd25_dismantling_shear", "Attachment", coreSlot=True)

    uc = build_undercarriage(veh, p_main, bodies, props_meta)
    ss = build_superstructure(veh, p_ss, bodies, props_meta)
    build_cab(veh, p_cab, bodies, props_meta, elevated=False)
    build_cab(veh, p_cabe, bodies, props_meta, elevated=True)
    build_boom(veh, p_boom, bodies, props_meta)
    build_stick(veh, p_stick, bodies, props_meta)
    build_attachment(veh, p_att, bodies, props_meta)
    build_clamp(veh, p_clamp, bodies, props_meta)
    build_hydraulics_parts(veh, p_hyd, p_hyd_fast)

    for b in bodies:
        b.finalize_mass(veh)
    nk, nd = veh.clamp_stability(M.PHYS_DT)
    print(f"stability clamp: {nk} springs and {nd} dampers reduced to the 2 kHz limit")

    # flexbodies from mesh registry
    for mesh, info in MESH.FLEXBODIES.items():
        veh.parts[info["part"]].flexbody(mesh, info["groups"], **info.get("opts", {}))
    for mesh, info in MESH.ELEVATED_CAB.items():
        veh.parts["fd25_cab_elevated"].flexbody(mesh, info["groups"], **info["opts"])
    for pm in props_meta:
        part = veh.parts[pm["part"]]
        part.prop(pm["func"], pm["mesh"], pm["ref"], pm["idx"], pm["idy"], rot=pm.get("rot", (0, 0, 0)),
                  trans=pm.get("trans", (0, 0, 0)), **pm.get("opts", {}))

    return veh, bodies, props_meta


# ============================================================ undercarriage
def build_undercarriage(veh, p, bodies, props_meta):
    U = M.UC
    body = Body(p, "uc", "fd_uc", U.mass - 8 * U.road_wheel_mass)
    bodies.append(body)
    ys = [U.frame_y0, -0.79, 0.0, 0.79, U.frame_y1]
    frame = {}
    for S, sx in (("L", 1), ("R", -1)):
        xc = sx * U.track_x
        hw = U.frame_w / 2
        rings = []
        for i, y in enumerate(ys):
            ob = body.n(f"tf{S}{i}ob", (xc + sx * hw, y, U.frame_z0))
            ib = body.n(f"tf{S}{i}ib", (xc - sx * hw, y, U.frame_z0))
            it = body.n(f"tf{S}{i}it", (xc - sx * hw, y, U.frame_z1))
            ot = body.n(f"tf{S}{i}ot", (xc + sx * hw, y, U.frame_z1))
            rings.append([ob, ib, it, ot])
        truss(p, rings, **STIFF)
        frame[S] = rings
        skin_rings(p, veh, rings)

    # centre body
    cb = {}
    rings = []
    for j, y in enumerate((-U.cb_y, 0.0, U.cb_y)):
        r = []
        for (side, sx), z in (((("L", 1), U.cb_z0)), ((("R", -1), U.cb_z0)), ((("R", -1), U.cb_z1)), ((("L", 1), U.cb_z1))):
            nm = f"cb{side}{j}{'b' if z == U.cb_z0 else 't'}"
            if nm not in cb:
                cb[nm] = body.n(nm, (sx * U.cb_x, y, z))
            r.append(nm)
        rings.append(r)
    truss(p, rings, **STIFF)
    skin_rings(p, veh, rings)
    # tie centre body to track frames
    for S, sx in (("L", 1), ("R", -1)):
        fr = frame[S]
        for j, (ia, ib_) in enumerate(((0, 1), (1, 3), (3, 4))):
            for z in "bt":
                c = f"cb{S}{j}{z}"
                for i in range(ia, ib_ + 1):
                    connect(p, c, [fr[i][1], fr[i][2]], **STIFF)
    # ref nodes
    body.n("uc_ref", (0.0, 0.0, U.cb_z0), collision=False)
    body.n("uc_back", (0.0, U.cb_y, U.cb_z0), collision=False)
    body.n("uc_up", (0.0, 0.0, U.cb_z1), collision=False)
    for n in ("uc_ref", "uc_back", "uc_up"):
        connect(p, n, nearest(veh, veh.pos(n), list(cb), 6), **STIFF)
    connect(p, "uc_ref", ["uc_back", "uc_up"], **STIFF)
    connect(p, "uc_back", ["uc_up"], **STIFF)

    # slew stator ring + axis
    stator = []
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        stator.append(body.n(f"sst{k}", (U.slew_ring_r * math.cos(a), U.slew_ring_r * math.sin(a), U.slew_ring_z),
                             weight=85.0, collision=False))
    ring_beams(p, stator, **STIFF)
    for s in stator:
        connect(p, s, nearest(veh, veh.pos(s), [n for n in cb if n.endswith("t")] + ["uc_up"], 3), **STIFF)
        connect(p, s, nearest(veh, veh.pos(s), [n for n in cb if n.endswith("b")], 2), **STIFF)
    for k in range(8):
        p.beam(stator[k], stator[(k + 3) % 8], **STIFF)
    sl0 = body.n("sl0", (0.0, 0.0, M.SS.slew_axis_low), weight=320.0, collision=False,
                 groups=["fd_uc", "fd_ss"])
    sl1 = body.n("sl1", (0.0, 0.0, M.SS.slew_axis_high), weight=320.0, collision=False,
                 groups=["fd_uc", "fd_ss"])
    connect(p, sl0, stator + list(cb) + ["uc_ref", "uc_up"], **JOINT)
    tops = [frame[S][i][2] for S in "LR" for i in (1, 2, 3)]
    connect(p, sl1, stator + [n for n in cb if n.endswith("t")] + tops, **JOINT)

    # clamp-arm mount and cylinder bases
    Cl = M.CLAMP
    cm = []
    for S, sx in (("L", 1), ("R", -1)):
        a = body.n(f"cm{S}0", (sx * Cl.pivot_x, -1.62, 0.34))
        b = body.n(f"cm{S}1", (sx * Cl.pivot_x, -1.62, 0.86))
        cm += [a, b]
        connect(p, a, [f"cb{S}0b", f"cb{S}0t", frame[S][0][1], frame[S][0][2], frame[S][1][1]], **STIFF)
        connect(p, b, [f"cb{S}0b", f"cb{S}0t", frame[S][0][1], frame[S][0][2], frame[S][1][2]], **STIFF)
        p.beam(a, b, **STIFF)
    p.beam("cmL0", "cmR0", **STIFF); p.beam("cmL1", "cmR1", **STIFF)
    p.beam("cmL0", "cmR1", **STIFF); p.beam("cmL1", "cmR0", **STIFF)
    for S, sx in (("L", 1), ("R", -1)):
        piv = body.n(f"cp{S}", pw(Cl.pivot[0], Cl.pivot[1], sx * Cl.pivot_x), weight=60.0,
                     groups=["fd_uc", "fd_clamp"])
        connect(p, piv, [f"cm{S}0", f"cm{S}1", frame[S][0][1], frame[S][0][2], frame[S][0][0]], **JOINT)
        o = "R" if S == "L" else "L"
        connect(p, piv, [f"cm{o}0", f"cm{o}1"], **STIFF)
        base = body.n(f"ccb{S}", pw(Cl.cyl_base[0], Cl.cyl_base[1], sx * Cl.cyl_x),
                      weight=40.0 + M.CYL["clamp"].mass / 2, collision=False)
        connect(p, base, [f"cb{S}0b", f"cb{S}0t", f"cm{S}1", f"cm{S}0", f"cb{o}0t"], **JOINT)
    p.beam("cpL", "cpR", **STIFF)
    for a in ("ccbL", "ccbR"):
        connect(p, a, ["cpL", "cpR"], **JOINT)

    # sprocket / idler axle nodes (visual props + track ends)
    for S, sx in (("L", 1), ("R", -1)):
        xc = sx * U.track_x
        fr = frame[S]
        for kind, y, z, ring in (("idl", U.idler_y, U.idler_z, fr[0]), ("spr", U.sprocket_y, U.sprocket_z, fr[4])):
            c = body.n(f"{kind}{S}c", (xc, y, z), collision=False)
            pp = body.n(f"{kind}{S}p", (xc + 0.25, y, z), collision=False)  # +X side of the axis
            pm = body.n(f"{kind}{S}m", (xc - 0.25, y, z), collision=False)
            pu = body.n(f"{kind}{S}u", (xc, y, z + 0.40), collision=False)
            for n in (c, pp, pm, pu):
                connect(p, n, ring, **STIFF)
            p.beam(c, pu, **STIFF)
            other = fr[1] if kind == "idl" else fr[3]
            connect(p, c, other[:2], **STIFF)
            p.beam(c, pp, **STIFF); p.beam(c, pm, **STIFF); p.beam(pp, pm, **STIFF)

    # track envelope (collision skin / obstacle skids)
    env = {}
    for S, sx in (("L", 1), ("R", -1)):
        xc = sx * U.track_x
        hw = U.shoe_width / 2 - 0.01
        pts = OrderedDict([
            ("fn", (U.idler_y - U.idler_r - 0.04, U.idler_z)),
            ("t0", (-1.50, 0.90)), ("t1", (0.0, 0.90)), ("t2", (1.50, 0.90)),
            ("rn", (U.sprocket_y + U.sprocket_r + 0.04, U.sprocket_z)),
            ("b2", (1.50, 0.07)), ("b1", (0.0, 0.07)), ("b0", (-1.50, 0.07)),
        ])
        loop = []
        for key, (y, z) in pts.items():
            o_ = body.n(f"te{S}{key}o", (xc + sx * hw, y, z), weight=18.0, frictionCoef=0.55)
            i_ = body.n(f"te{S}{key}i", (xc - sx * hw, y, z), weight=18.0, frictionCoef=0.55)
            loop.append((o_, i_))
            cand = [n for r in frame[S] for n in r] + [f"idl{S}c", f"spr{S}c"]
            for nn in (o_, i_):
                connect(p, nn, nearest(veh, veh.pos(nn), cand, 4), **MEDIUM)
            p.beam(o_, i_, **MEDIUM)
        n_ = len(loop)
        for k in range(n_):
            a, b = loop[k], loop[(k + 1) % n_]
            p.beam(a[0], b[0], **MEDIUM); p.beam(a[1], b[1], **MEDIUM); p.beam(a[0], b[1], **MEDIUM)
        env[S] = loop
        # skin: outer & inner side faces (fan), and the running surfaces
        cen = np.mean([veh.pos(n) for pr in loop for n in pr], axis=0)
        for k in range(n_):
            a, b = loop[k], loop[(k + 1) % n_]
            tri_outward(p, veh, a[0], b[0], b[1], cen)
            tri_outward(p, veh, a[0], b[1], a[1], cen)
        for side_idx in (0, 1):
            ids = [pr[side_idx] for pr in loop]
            for k in range(1, n_ - 1):
                tri_outward(p, veh, ids[0], ids[k], ids[k + 1], cen)

    # hidden steel road wheels (rotators)
    rot_rows = [["name", "[group]:", "node1:", "node2:", "nodeArm:", "wheelDir"],
                OrderedDict([("radius", U.road_wheel_r), ("brakeTorque", 22000), ("parkingTorque", 26000),
                             ("brakeSpring", 400), ("rotatorType", "rotator"), ("speedo", True),
                             ("frictionCoef", U.road_wheel_friction), ("enableBrakeThermals", False),
                             ("brakeInputSplit", 1), ("useDefaultBrakeInput", False)])]
    wheel_names = {"L": [], "R": []}
    for S, sx in (("L", 1), ("R", -1)):
        xc = sx * U.track_x
        for i, y in enumerate(U.road_wheel_y):
            wname = f"RW_{S}{i + 1}"
            grp = f"fd_rw_{S}{i + 1}"
            ah = U.road_wheel_axle_half
            a_m = body.n(f"rwa{S}{i}m", (xc - ah, y, U.road_wheel_r), weight=65.0, collision=False)
            a_p = body.n(f"rwa{S}{i}p", (xc + ah, y, U.road_wheel_r), weight=65.0, collision=False)
            cand = [n for r in frame[S] for n in r]
            for a in (a_m, a_p):
                connect(p, a, nearest(veh, veh.pos(a), cand, 6), **STIFF)
            p.beam(a_m, a_p, **STIFF)
            wb = Body(p, f"rw{S}{i}", grp, U.road_wheel_mass,
                      frictionCoef=U.road_wheel_friction, nodeMaterial="|NM_METAL")
            bodies.append(wb)
            rays = U.road_wheel_rays
            rings = {}
            for side_, dx in (("a", -U.road_wheel_halfwidth), ("b", U.road_wheel_halfwidth)):
                rr = []
                for k in range(rays):
                    phi = 2 * math.pi * k / rays
                    pos = (xc + dx, y + U.road_wheel_r * math.sin(phi), U.road_wheel_r + U.road_wheel_r * math.cos(phi))
                    rr.append(wb.n(f"rw{S}{i}{side_}{k}", pos))
                rings[side_] = rr
            for side_ in "ab":
                rr = rings[side_]
                for k in range(rays):
                    p.beam(rr[k], rr[(k + 1) % rays], **WHEEL)
                    p.beam(rr[k], rr[(k + 2) % rays], **WHEEL)
                    p.beam(rr[k], a_m, **WHEEL)
                    p.beam(rr[k], a_p, **WHEEL)
            for k in range(rays):
                p.beam(rings["a"][k], rings["b"][k], **WHEEL)
                p.beam(rings["a"][k], rings["b"][(k + 1) % rays], **WHEEL)
            arm = frame[S][1][3] if i < 2 else frame[S][3][3]
            # node1 -> node2 always points +X so both sides share one convention
            rot_rows.append([wname, [grp], a_m, a_p, arm, 1,
                             {"brakeElectricsName": f"fd_brake{S}"}])
            wheel_names[S].append(wname)
    p.section("rotators", rot_rows)

    # powertrain: one electric travel motor per side -> locked diff tree -> 4 shafts
    gear = 62.0
    pt = [["type", "name", "inputName", "inputIndex"]]
    for S in "LR":
        pt.append(["electricMotor", f"travelMotor{S}", "dummy", 0])
        pt.append(["differential", f"trackDiff{S}", f"travelMotor{S}", 1, {"diffType": "locked", "uiName": f"Final drive {S}"}])
        pt.append(["differential", f"trackDiff{S}a", f"trackDiff{S}", 1, {"diffType": "locked"}])
        pt.append(["differential", f"trackDiff{S}b", f"trackDiff{S}", 2, {"diffType": "locked"}])
        for k, wname in enumerate(wheel_names[S]):
            parent = f"trackDiff{S}a" if k < 2 else f"trackDiff{S}b"
            pt.append(["shaft", f"trackAxle{S}{k + 1}", parent, 1 + (k % 2),
                       {"connectedWheel": wname, "gearRatio": gear, "friction": 3, "dynamicFriction": 0.02}])
    p.section("powertrain", pt)
    for S in "LR":
        fr = frame[S]
        p.section(f"travelMotor{S}", OrderedDict([
            ("torque", [["rpm", "torque"], [0, 650], [2400, 650], [3000, 520], [3400, 0]]),
            ("maxRPM", 3200), ("inertia", 0.35), ("friction", 4), ("dynamicFriction", 0.004),
            ("electricsThrottleName", f"fd_track{S}"),
            ("electricsThrottleFactorName", f"fd_trackFactor{S}"),
            ("isAffectedByIgnitionLevel", False),
            ("torqueReactionNodes:", [fr[3][0], fr[3][1], fr[4][3]]),
            ("electricalEfficiency", 0.9),
            ("uiName", f"Travel motor {S}"),
        ]))

    # undercarriage visuals: animated props
    for S in "LR":
        props_meta += MESH.track_props(S)
    return body


# ============================================================ superstructure
def build_superstructure(veh, p, bodies, props_meta):
    S_ = M.SS
    body = Body(p, "ss", "fd_ss", S_.mass)
    cwb = Body(p, "cw", "fd_ss", S_.counterweight_mass)
    bodies += [body, cwb]
    xs = (1.30, 0.40, -0.40, -1.30)
    ys = (S_.front_y + 0.02, -0.35, 0.70, 1.70)
    deck = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for lvl, z in (("b", S_.frame_z0), ("t", S_.deck_z)):
                deck[(i, j, lvl)] = body.n(f"sd{i}{j}{lvl}", (x, y, z))
    # deck truss: rings along y for each pair of adjacent x columns
    for i in range(3):
        rings = [[deck[(i, j, "b")], deck[(i + 1, j, "b")], deck[(i + 1, j, "t")], deck[(i, j, "t")]] for j in range(4)]
        truss(p, rings, **STIFF)
    allb = [deck[k] for k in deck if k[2] == "b"]
    allt = [deck[k] for k in deck if k[2] == "t"]
    # deck skin (top & sides) for collisions
    cen = np.array([0.0, 0.25, 1.08])
    for i in range(3):
        for j in range(3):
            a, b_, c, d = deck[(i, j, "t")], deck[(i + 1, j, "t")], deck[(i + 1, j + 1, "t")], deck[(i, j + 1, "t")]
            tri_outward(p, veh, a, b_, c, cen); tri_outward(p, veh, a, c, d, cen)
    # deep main frame (the two vertical side plates of the upper frame): top chord at z = 1.95
    mf = {}
    for i, x in ((1, 0.40), (2, -0.40)):
        for j, y in enumerate(ys):
            mf[(i, j)] = body.n(f"mf{i}{j}", (x, y, 1.95), weight=110.0)
    for i in (1, 2):
        col = [[deck[(i, j, "b")], deck[(i, j, "t")], mf[(i, j)]] for j in range(4)]
        for j in range(4):
            p.beam(col[j][0], col[j][2], **STIFF); p.beam(col[j][1], col[j][2], **STIFF)
            if j + 1 < 4:
                for a in col[j]:
                    for b in col[j + 1]:
                        p.beam(a, b, **STIFF)
        for j in range(4):
            for jj in range(j + 2, 4):
                p.beam(mf[(i, j)], mf[(i, jj)], **STIFF)
                p.beam(deck[(i, j, "b")], deck[(i, jj, "b")], **STIFF)
    for j in range(4):
        p.beam(mf[(1, j)], mf[(2, j)], **STIFF)
        p.beam(mf[(1, j)], deck[(2, j, "b")], **STIFF); p.beam(mf[(2, j)], deck[(1, j, "b")], **STIFF)
        if j + 1 < 4:
            p.beam(mf[(1, j)], mf[(2, j + 1)], **STIFF); p.beam(mf[(2, j)], mf[(1, j + 1)], **STIFF)
    allmf = list(mf.values())

    # rotor ring
    rotor = []
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        rotor.append(body.n(f"ssr{k}", (M.UC.slew_ring_r * math.cos(a), M.UC.slew_ring_r * math.sin(a), S_.frame_z0 + 0.02),
                            weight=85.0, collision=False))
    ring_beams(p, rotor, **STIFF)
    for k in range(8):
        p.beam(rotor[k], rotor[(k + 3) % 8], **STIFF)
    for r in rotor:
        connect(p, r, nearest(veh, veh.pos(r), allb, 3) + nearest(veh, veh.pos(r), allt, 2), **STIFF)
    connect(p, "sl0", rotor + nearest(veh, (0, 0, 1.0), allb, 4), **JOINT)
    connect(p, "sl1", rotor + nearest(veh, (0, 0, 1.16), allt, 6) + allmf, **JOINT)

    # boom foot tower and pins
    A = M.ARM
    xa = A.xa
    tower = []
    for side, sx in (("L", 1), ("R", -1)):
        x = xa + sx * 0.40
        t0 = body.n(f"bt{side}0", (x, -1.18, S_.deck_z + 0.02), weight=170.0)
        t1 = body.n(f"bt{side}1", (x, -0.40, S_.deck_z + 0.02), weight=170.0)
        t2 = body.n(f"bt{side}2", (x, -0.78, 2.10), weight=170.0)
        tower += [t0, t1, t2]
        connect(p, t0, nearest(veh, veh.pos(t0), allt + allb, 4), **STIFF)
        connect(p, t1, nearest(veh, veh.pos(t1), allt + allb, 4), **STIFF)
        connect(p, t2, [t0, t1] + nearest(veh, veh.pos(t2), allt, 3), **STIFF)
        p.beam(t0, t1, **STIFF)
    for a in ("0", "1", "2"):
        p.beam("btL" + a, "btR" + a, **STIFF)
    p.beam("btL0", "btR2", **STIFF); p.beam("btR0", "btL2", **STIFF)
    p.beam("btL1", "btR2", **STIFF); p.beam("btR1", "btL2", **STIFF)
    connect(p, "sl1", ["btL2", "btR2", "btL1", "btR1"], **STIFF)
    for t in tower:
        connect(p, t, nearest(veh, veh.pos(t), allmf, 3), **STIFF)
    for side, sx in (("L", 1), ("R", -1)):
        pin = body.n(f"bf{side}", pw(A.boom_foot[0], A.boom_foot[1], xa + sx * A.boom_pin_half),
                     weight=160.0, groups=["fd_ss", "fd_boom"])
        connect(p, pin, tower, **JOINT)
        base = body.n(f"bcb{side}", pw(A.boomcyl_base[0], A.boomcyl_base[1], xa + sx * A.boomcyl_x),
                      weight=60.0 + M.CYL["boom"].mass / 2, collision=False)
        connect(p, base, nearest(veh, veh.pos(base), allt + allb + tower, 6), **JOINT)
    p.beam("bfL", "bfR", **JOINT)
    p.beam("bcbL", "bcbR", **STIFF)
    # tie the boom-cylinder bases straight to the boom foot pins (rigid support triangle)
    for a in ("bcbL", "bcbR"):
        connect(p, a, ["bfL", "bfR"], **JOINT)

    # counterweight (separate heavy body, welded)
    cw_pts = [(1.20, S_.cw_y0), (0.75, 2.55), (0.0, S_.tail_radius), (-0.75, 2.55), (-1.20, S_.cw_y0)]
    cwn = []
    for k, (x, y) in enumerate(cw_pts):
        b_ = cwb.n(f"cw{k}b", (x, y, S_.cw_z0))
        t_ = cwb.n(f"cw{k}t", (x, y, S_.cw_z1))
        cwn.append((b_, t_))
    for k in range(len(cwn)):
        p.beam(*cwn[k], **STIFF)
        if k + 1 < len(cwn):
            a, b_ = cwn[k], cwn[k + 1]
            p.beam(a[0], b_[0], **STIFF); p.beam(a[1], b_[1], **STIFF)
            p.beam(a[0], b_[1], **STIFF); p.beam(a[1], b_[0], **STIFF)
    rear = [deck[(i, 3, lvl)] for i in range(4) for lvl in "bt"] + [deck[(i, 2, "t")] for i in range(4)]
    for pr in cwn:
        for n in pr:
            connect(p, n, nearest(veh, veh.pos(n), rear, 4), **STIFF)
    p.beam("cw0t", "cw4t", **STIFF); p.beam("cw0b", "cw4b", **STIFF); p.beam("cw1t", "cw3b", **STIFF)
    # counterweight skin
    cen = np.array([0.0, 2.1, 1.7])
    for k in range(len(cwn) - 1):
        a, b_ = cwn[k], cwn[k + 1]
        tri_outward(p, veh, a[0], b_[0], b_[1], cen); tri_outward(p, veh, a[0], b_[1], a[1], cen)
    ids_t = [pr[1] for pr in cwn]
    for k in range(1, len(ids_t) - 1):
        tri_outward(p, veh, ids_t[0], ids_t[k], ids_t[k + 1], cen)

    # engine hood and tanks (collision volume nodes)
    hood = []
    for k, (x, y) in enumerate(((1.25, S_.hood_y0), (-1.25, S_.hood_y0), (-1.25, S_.hood_y1), (1.25, S_.hood_y1))):
        hood.append(body.n(f"hd{k}", (x, y, S_.hood_z)))
    ring_beams(p, hood, **MEDIUM)
    for n in hood:
        connect(p, n, nearest(veh, veh.pos(n), allt + ids_t, 4) + nearest(veh, veh.pos(n), allmf, 2), **MEDIUM)
    for n in ids_t:
        connect(p, n, nearest(veh, veh.pos(n), allmf, 2), **STIFF)
    tank = []
    for k, (x, y) in enumerate(((S_.tank_x1, S_.tank_y0), (S_.tank_x0, S_.tank_y0), (S_.tank_x0, S_.tank_y1), (S_.tank_x1, S_.tank_y1))):
        tank.append(body.n(f"tk{k}", (x, y, S_.tank_z)))
    ring_beams(p, tank, **MEDIUM)
    for n in tank:
        connect(p, n, nearest(veh, veh.pos(n), allt, 4), **MEDIUM)
    connect(p, "sl1", [tank[0], tank[3], hood[0], hood[1]], **MEDIUM)
    cen = np.array([0.0, 0.6, 1.6])
    tri_outward(p, veh, hood[0], hood[1], hood[2], cen); tri_outward(p, veh, hood[0], hood[2], hood[3], cen)
    tri_outward(p, veh, tank[0], tank[1], tank[2], cen); tri_outward(p, veh, tank[0], tank[2], tank[3], cen)
    # sides of tank block
    tri_outward(p, veh, tank[1], deck[(3, 0, "t")], tank[2], np.array([-0.9, -0.3, 1.6]))
    tri_outward(p, veh, tank[2], deck[(3, 0, "t")], deck[(3, 2, "t")], np.array([-0.9, -0.3, 1.6]))

    # rear work light & exhaust
    body.n("ss_lamp_rear", (0.0, S_.tail_radius - 0.05, S_.cw_z1 + 0.10), weight=5.0, collision=False)
    connect(p, "ss_lamp_rear", ["cw2t", "cw1t", "cw3t"], **LIGHT)

    # slew torsion hydros (stator k  <->  rotor k)
    for k in (0, 3, 5):
        p.torsion_hydro(f"sst{k}", "sl0", "sl1", f"ssr{k}", spring=2.2e7, damp=1.5e4,
                        strength=FLT_MAX, deform=FLT_MAX,
                        inputSource="fd_slew", inLimit=-M.RANGE.slew[1], outLimit=M.RANGE.slew[1],
                        inRate=2.0, outRate=2.0, autoCenterRate=2.0, inputFactor=1)
    return body


# ============================================================ cab
def build_cab(veh, p, bodies, props_meta, elevated):
    C = M.CAB
    sfx = "e" if elevated else ""
    R = C.riser_elevated if elevated else 0.0
    body = Body(p, "cab" + sfx, "fd_cab", C.mass + (C.riser_mass if elevated else 0.0))
    bodies.append(body)
    z0 = C.z0 + 0.04 + R
    zs = (z0, 2.10 + R, C.z1 + R)
    nodes = {}
    for xi, x in (("o", C.x1), ("i", C.x0)):
        for yi, y in (("f", C.y0), ("r", C.y1)):
            for zi, z in zip("bmt", zs):
                nodes[(xi, yi, zi)] = body.n(f"cab{sfx}{xi}{yi}{zi}", (x, y, z))
    rings = []
    for zi in "bmt":
        rings.append([nodes[("o", "f", zi)], nodes[("i", "f", zi)], nodes[("i", "r", zi)], nodes[("o", "r", zi)]])
    truss(p, rings, **MEDIUM)
    skin_rings(p, veh, rings, caps=(False, True))
    deck_t = [f"sd{i}{j}t" for i in range(4) for j in range(4)]
    mfn = [f"mf{i}{j}" for i in (1, 2) for j in range(4)]
    if not elevated:
        for n in rings[0]:
            connect(p, n, nearest(veh, veh.pos(n), deck_t, 5), **STIFF)
        for n in rings[1]:
            connect(p, n, nearest(veh, veh.pos(n), mfn, 2), **STIFF)
    else:
        # riser box: 4 posts from the deck to the cab floor
        posts = []
        for k, (x, y) in enumerate(((C.x1 - 0.08, C.y0 + 0.25), (C.x0 + 0.08, C.y0 + 0.25),
                                    (C.x0 + 0.08, C.y1 - 0.20), (C.x1 - 0.08, C.y1 - 0.20))):
            posts.append(body.n(f"cabe_post{k}", (x, y, M.SS.deck_z + 0.04)))
        ring_beams(p, posts, **STIFF)
        for n in posts:
            connect(p, n, nearest(veh, veh.pos(n), deck_t, 5) + nearest(veh, veh.pos(n), mfn, 2), **STIFF)
        truss(p, [posts, rings[0]], **STIFF)
    # driver camera and lamps
    eye = np.array(C.eye) + np.array([0, 0, R])
    cams = [["type", "x", "y", "z", "fov", "id1:", "id2:", "id3:", "id4:", "id5:", "id6:"],
            {"nodeWeight": 1.5, "selfCollision": False, "collision": False, "beamSpring": 1.0e6, "beamDamp": 200},
            ["driver", r4(eye[0]), r4(eye[1]), r4(eye[2]), 72,
             nodes[("o", "f", "m")], nodes[("i", "f", "m")], nodes[("o", "r", "m")], nodes[("i", "r", "m")],
             nodes[("o", "f", "t")], nodes[("i", "r", "t")],
             # orientation follows the (slewing) cab: left = +X side, back = +Y side of ref
             {"idRef": nodes[("i", "f", "b")], "idBack": nodes[("i", "r", "b")], "idLeft": nodes[("o", "f", "b")]}],
            ]
    p.section("camerasInternal", cams)
    lamps = []
    for k, x in enumerate((C.x0 + 0.12, C.x1 - 0.10)):
        n = body.n(f"cab{sfx}_lamp{k}", (x, C.y0 - 0.05, C.z1 + R - 0.06), weight=4.0, collision=False)
        connect(p, n, nearest(veh, veh.pos(n), [nodes[k_] for k_ in nodes], 3), **LIGHT)
        lamps.append(n)
    bea = body.n(f"cab{sfx}_beacon", (0.5 * (C.x0 + C.x1), C.y1 - 0.25, C.z1 + R + 0.12), weight=3.0, collision=False)
    connect(p, bea, nearest(veh, veh.pos(bea), [nodes[k_] for k_ in nodes], 3), **LIGHT)
    props_meta += MESH.cab_props(p.name, sfx, nodes, lamps, bea)
    return body


# ============================================================ boom
def boom_center_t(s):
    A = M.ARM
    L, ks, kt = A.boom_len, A.boom_knee_s * A.boom_len, A.boom_knee_t
    return kt * s / ks if s <= ks else kt * (L - s) / (L - ks)


def boom_depth(s):
    A = M.ARM
    L, ks = A.boom_len, A.boom_knee_s * A.boom_len
    if s <= ks:
        return A.boom_depth_foot + (A.boom_depth_knee - A.boom_depth_foot) * s / ks
    return A.boom_depth_knee + (A.boom_depth_nose - A.boom_depth_knee) * (s - ks) / (L - ks)


def boom_width(s):
    return M.ARM.boom_width - 0.06 * s / M.ARM.boom_len


def build_boom(veh, p, bodies, props_meta):
    A = M.ARM
    body = Body(p, "boom", "fd_boom", A.boom_mass)
    bodies.append(body)
    stations = (0.32, 1.40, A.boom_knee_s * A.boom_len, 4.40, 5.36)
    rings = []
    for i, s in enumerate(stations):
        c, d, w = boom_center_t(s), boom_depth(s), boom_width(s) / 2
        r = [body.n(f"bm{i}bl", W("boom", s, c - d / 2, +w)), body.n(f"bm{i}br", W("boom", s, c - d / 2, -w)),
             body.n(f"bm{i}tr", W("boom", s, c + d / 2, -w)), body.n(f"bm{i}tl", W("boom", s, c + d / 2, +w))]
        rings.append(r)
    truss(p, rings, **STIFF)
    skin_rings(p, veh, rings)
    connect(p, "bfL", rings[0] + rings[1][:2], **JOINT)
    connect(p, "bfR", rings[0] + rings[1][:2], **JOINT)
    for side, sx in (("L", 1), ("R", -1)):
        n = body.n(f"bn{side}", W("boom", A.boom_len, 0.0, sx * 0.27), weight=110.0, groups=["fd_boom", "fd_stick"])
        connect(p, n, rings[4] + rings[3], **JOINT)
        lug = body.n(f"bcr{side}", W("boom", *A.boomcyl_rod_on_boom, x=sx * A.boomcyl_x),
                     weight=40.0 + M.CYL["boom"].mass / 2, collision=False)
        connect(p, lug, nearest(veh, veh.pos(lug), rings[1] + rings[2], 5) + ["bfL", "bfR"], **JOINT)
    p.beam("bnL", "bnR", **JOINT)
    for a in ("bcrL", "bcrR"):
        connect(p, a, ["bnL", "bnR"], **STIFF)
    scb = body.n("scb", W("boom", *A.stickcyl_base_on_boom), weight=40.0 + M.CYL["stick"].mass / 2, collision=False)
    connect(p, scb, nearest(veh, veh.pos(scb), rings[2] + rings[3] + rings[1], 6) + ["bfL", "bfR"], **JOINT)
    connect(p, "scb", ["bnL", "bnR"], **JOINT)
    lamp = body.n("boom_lamp", W("boom", 2.10, boom_center_t(2.10) + boom_depth(2.10) / 2 + 0.12, 0.0),
                  weight=4.0, collision=False)
    connect(p, lamp, nearest(veh, veh.pos(lamp), rings[1] + rings[2], 4), **LIGHT)

    # boom cylinders: hydro per side
    for side in "LR":
        add_cylinder_hydro(p, "boom", f"bcb{side}", f"bcr{side}")
    props_meta += MESH.cylinder_props("fd25_boom_mono", "boom", [("bcbL", "bcrL", "bcbR", "bcrR"),
                                                                ("bcbR", "bcrR", "bcbL", "bcrL")])
    props_meta += MESH.light_props("fd25_boom_mono", "boom_lamp", "bm2tl", "bm2tr", range_=34, bright=1.1)
    return body


# ============================================================ cylinders (hydros)
def cyl_limits(name):
    rep = K.cylinder_ranges()
    key = {"boom": "boom_l", "stick": "stick", "tilt": "tilt", "jaw": "jaw", "clamp": "clamp_l"}[name]
    return rep[key]


def add_cylinder_hydro(p, cname, a, b):
    c = M.CYL[cname]
    lim = cyl_limits(cname)
    area = math.pi * c.bore ** 2 / 4.0
    force_limit = c.relief * area * 1.05
    spawn = lim["spawn"]
    rate = max(0.25, 1.6 * c.max_speed / spawn)
    p.hydro(a, b, beamSpring=4.0e7 if cname != "jaw" else 3.0e7, beamDamp=4500,
            beamDeform=FLT_MAX, beamStrength=FLT_MAX,
            inputSource=f"fd_{c.channel}", inExtent=r4(lim["min"]), outExtent=r4(lim["max"]),
            inRate=r4(rate), outRate=r4(rate), autoCenterRate=r4(rate), inputFactor=1,
            forceLimit=r4(force_limit * c.count / max(c.count, 1)))


# ============================================================ stick + linkage
def build_stick(veh, p, bodies, props_meta):
    A = M.ARM
    body = Body(p, "stick", "fd_stick", A.stick_mass)
    side = Body(p, "sidelink", "fd_sidelink", A.linkage_mass * 0.55)
    hlink_mass = A.linkage_mass * 0.45
    bodies += [body, side]
    st = ((-0.45, 0.24, 0.52), (0.30, 0.04, 0.48), (1.20, 0.0, 0.44), (2.06, 0.0, 0.38))
    rings = []
    for i, (s, c, d) in enumerate(st):
        w = A.stick_width / 2 - 0.02 * i
        r = [body.n(f"st{i}bl", W("stick", s, c - d / 2, +w)), body.n(f"st{i}br", W("stick", s, c - d / 2, -w)),
             body.n(f"st{i}tr", W("stick", s, c + d / 2, -w)), body.n(f"st{i}tl", W("stick", s, c + d / 2, +w))]
        rings.append(r)
    truss(p, rings, **STIFF)
    skin_rings(p, veh, rings)
    for sname in ("bnL", "bnR"):
        connect(p, sname, rings[0] + rings[1], **JOINT)
    for sd, sx in (("L", 1), ("R", -1)):
        n = body.n(f"sn{sd}", W("stick", A.stick_len, 0.0, sx * M.ATT.pin_half), weight=70.0,
                   groups=["fd_stick", "fd_bracket"])
        connect(p, n, rings[3] + rings[2], **JOINT)
        k = body.n(f"lk{sd}", W("stick", *A.linkpivot_on_stick, x=sx * 0.25), weight=25.0,
                   groups=["fd_stick", "fd_sidelink"])
        connect(p, k, rings[3] + rings[2][2:], **JOINT)
    p.beam("snL", "snR", **JOINT); p.beam("lkL", "lkR", **JOINT)
    p.beam("snL", "lkR", **JOINT); p.beam("snR", "lkL", **JOINT)
    sct = body.n("sct", W("stick", *A.stick_tail), weight=40.0 + M.CYL["stick"].mass / 2, collision=False)
    connect(p, sct, rings[0] + rings[1][2:] + ["bnL", "bnR"], **JOINT)
    tcb = body.n("tcb", W("stick", *A.tiltcyl_base_on_stick), weight=30.0 + M.CYL["tilt"].mass / 2, collision=False)
    connect(p, tcb, rings[0][2:] + rings[1] + rings[2][2:] + ["bnL", "bnR", "snL", "snR"], **JOINT)
    cam = body.n("stick_cam", W("stick", 1.35, 0.42, 0.0), weight=3.0, collision=False)
    connect(p, cam, nearest(veh, veh.pos(cam), rings[1] + rings[2], 4), **LIGHT)

    # side link body: K pins + P pins + rod-eye node Pe
    Pp = SPAWN["plane"]["P"]
    Kp = SPAWN["plane"]["K"]
    for sd, sx in (("L", 1), ("R", -1)):
        side.n(f"lp{sd}", K.plane_to_world(Pp, A.xa + sx * 0.25), weight=30.0,
               groups=["fd_sidelink", "fd_hlink"])
    pe_plane = Pp + 0.05 * (Kp - Pp) / np.linalg.norm(Kp - Pp)
    side.n("lpe", K.plane_to_world(pe_plane, A.xa), weight=20.0 + M.CYL["tilt"].mass / 2, collision=False,
           groups=["fd_sidelink"])
    connect(p, "lpL", ["lkL", "lkR"], **JOINT)
    connect(p, "lpR", ["lkL", "lkR"], **JOINT)
    p.beam("lpL", "lpR", **JOINT)
    connect(p, "lpe", ["lkL", "lkR", "lpL", "lpR"], **JOINT)
    # The rod eye sits in the plane of the four link pins while the cylinder pushes almost normal to
    # that plane, so the in-plane beams alone give it no first-order stiffness along the cylinder
    # axis (the claw would wobble on geometric stiffness).  A brace node off the plane (the link
    # plates' real depth) makes the rod eye a proper tetrahedral joint.
    pl = [veh.pos(n) for n in ("lkL", "lkR", "lpL", "lpR")]
    nrm = np.cross(pl[1] - pl[0], pl[2] - pl[0])
    nrm /= np.linalg.norm(nrm)
    if (veh.pos("tcb") - veh.pos("lpe")) @ nrm < 0:
        nrm = -nrm
    side.n("lpo", veh.pos("lpe") + 0.16 * nrm, weight=30.0, collision=False, groups=["fd_sidelink"])
    connect(p, "lpo", ["lkL", "lkR", "lpL", "lpR", "lpe"], **JOINT)
    # mass of the H-link sits on the link pins (C pins belong to the attachment)
    side.fixed_weight["lpL"] += hlink_mass / 2
    side.fixed_weight["lpR"] += hlink_mass / 2

    add_cylinder_hydro(p, "stick", "scb", "sct")
    add_cylinder_hydro(p, "tilt", "tcb", "lpe")
    props_meta += MESH.cylinder_props("fd25_stick_hd", "stick", [("scb", "sct", "bm2tl", "st0tl")])
    props_meta += MESH.cylinder_props("fd25_stick_hd", "tilt", [("tcb", "lpe", "st1tl", "lpL")])
    p.section("camerasInternal", [
        ["type", "x", "y", "z", "fov", "id1:", "id2:", "id3:", "id4:", "id5:", "id6:"],
        {"nodeWeight": 1.0, "selfCollision": False, "collision": False, "beamSpring": 8.0e5, "beamDamp": 150},
        ["claw_cam"] + [r4(c) for c in (veh.pos("stick_cam") + np.array([0.0, -0.10, 0.0]))] + [80,
         rings[1][2], rings[1][3], rings[2][2], rings[2][3], rings[2][0], rings[1][1]],
    ])
    return body


# ============================================================ attachment (shear)
def build_attachment(veh, p, bodies, props_meta):
    T = M.ATT
    brk = Body(p, "bracket", "fd_bracket", T.bracket_mass + T.rotator_mass * 0.6)
    head = Body(p, "head", "fd_head", T.head_mass + T.rotator_mass * 0.4)
    jaw = Body(p, "jaw", "fd_jaw", T.jaw_mass)
    bodies += [brk, head, jaw]
    half = T.pin_spacing / 2
    # H-link C pins (shared with H-link)
    for sd, sx in (("L", 1), ("R", -1)):
        brk.n(f"ac{sd}", W("att", 0.0, +half, sx * T.pin_half), weight=40.0, groups=["fd_bracket", "fd_hlink"])
    # H-link beams (body owned by the stick part conceptually, written here because C lives here)
    connect(p, "acL", ["lpL", "lpR"], **JOINT)
    connect(p, "acR", ["lpL", "lpR"], **JOINT)
    p.beam("acL", "acR", **JOINT)
    bk = []
    for sb, b in (("f", 0.28), ("b", -0.28)):
        for sd, sx in (("L", 1), ("R", -1)):
            bk.append(brk.n(f"bk{sb}{sd}", W("att", 0.20, b, sx * 0.28)))
    top = [brk.n("bktL", W("att", -0.10, 0.0, 0.24)), brk.n("bktR", W("att", -0.10, 0.0, -0.24))]
    ring_beams(p, [bk[0], bk[1], bk[3], bk[2]], **STIFF)
    pins = ["snL", "snR", "acL", "acR"]
    for n in pins + top:
        connect(p, n, bk, **JOINT)
    for a in top:
        connect(p, a, pins, **JOINT)
    p.beam("bktL", "bktR", **JOINT)
    # stator ring + axis nodes
    rr = T.rotator_r - 0.02
    stator = []
    for k in range(6):
        a = 2 * math.pi * k / 6 + math.pi / 6
        stator.append(brk.n(f"rst{k}", W("att", T.ring_a[0] + 0.02, rr * math.cos(a), rr * math.sin(a)),
                            weight=48.0, collision=False))
    ring_beams(p, stator, **STIFF)
    for k in range(6):
        p.beam(stator[k], stator[(k + 2) % 6], **STIFF)
        connect(p, stator[k], nearest(veh, veh.pos(stator[k]), bk, 3), **STIFF)
    rt0 = brk.n("rt0", W("att", T.axis_top_a, 0.0, 0.0), weight=60.0, collision=False,
                groups=["fd_bracket", "fd_head"])
    rt1 = brk.n("rt1", W("att", T.axis_bot_a, 0.0, 0.0), weight=60.0, collision=False,
                groups=["fd_bracket", "fd_head"])
    connect(p, rt0, bk + top + stator, **JOINT)
    connect(p, rt1, stator + bk, **JOINT)

    # ---- rotating head
    hr = []
    for k in range(6):
        a = 2 * math.pi * k / 6 + math.pi / 6
        hr.append(head.n(f"hr{k}", W("head", T.ring_a[1] + 0.04, 0.27 * math.cos(a), 0.27 * math.sin(a)), weight=48.0))
    ring_beams(p, hr, **STIFF)
    for k in range(6):
        p.beam(hr[k], hr[(k + 2) % 6], **STIFF)
    hm, hb = [], []
    for sb, b in (("f", T.head_hd), ("b", -T.head_hd)):
        for sd, sx in (("L", 1), ("R", -1)):
            hm.append(head.n(f"hm{sb}{sd}", W("head", 0.98, b, sx * T.head_hw)))
            hb.append(head.n(f"hb{sb}{sd}", W("head", T.head_a[1] - 0.02, b, sx * T.head_hw)))
    mid_ring = [hm[0], hm[1], hm[3], hm[2]]
    bot_ring = [hb[0], hb[1], hb[3], hb[2]]
    ring_beams(p, mid_ring, **STIFF)
    ring_beams(p, bot_ring, **STIFF)
    truss(p, [mid_ring, bot_ring], **STIFF)
    for n in hr:
        connect(p, n, nearest(veh, veh.pos(n), hm, 3), **STIFF)
    connect(p, rt0, hr, **JOINT)
    connect(p, rt1, hr + hm + hb, **JOINT)
    skin_rings(p, veh, [mid_ring, bot_ring], caps=(False, False))
    # jaw pivots (shared with the moving jaw)
    for sd, sx in (("L", 1), ("R", -1)):
        head.n(f"jp{sd}", W("head", *T.jaw_pivot, x=sx * 0.16), weight=35.0, groups=["fd_head", "fd_jaw"])
    connect(p, "jpL", hb + hm[2:], **JOINT)
    connect(p, "jpR", hb + hm[2:], **JOINT)
    p.beam("jpL", "jpR", **JOINT)
    jcb = head.n("jcb", W("head", *T.jawcyl_base), weight=30.0 + M.CYL["jaw"].mass / 2, collision=False)
    connect(p, jcb, hr + hm[:2] + ["jpL", "jpR"], **JOINT)

    # fixed fork jaw: two plates
    fl = T.fork_line
    nst = len(fl)
    plates = {}
    for sd, sx in (("L", 1), ("R", -1)):
        xp = sx * (T.fork_gap + T.fork_plate_t / 2)
        rows = []
        for i, (a, b) in enumerate(fl):
            if i < nst - 1:
                ta = np.subtract(fl[i + 1], fl[i])
            else:
                ta = np.subtract(fl[i], fl[i - 1])
            ta = ta / np.linalg.norm(ta)
            nb = np.array([-ta[1], ta[0]])  # perpendicular in (a,b)
            depth = T.fork_depth * (1.0 - 0.55 * i / (nst - 1))
            grip = dict(frictionCoef=1.6) if True else {}
            if i < nst - 1:
                f_ = head.n(f"fk{i}f{sd}", W("head", a + nb[0] * depth / 2, b + nb[1] * depth / 2, xp), **grip)
                k_ = head.n(f"fk{i}k{sd}", W("head", a - nb[0] * depth / 2, b - nb[1] * depth / 2, xp), **grip)
                rows.append((f_, k_))
            else:
                t_ = head.n(f"fk{i}t{sd}", W("head", a, b, xp), **grip)
                rows.append((t_, t_))
        plates[sd] = rows
    for i in range(nst):
        fL, fR = plates["L"][i], plates["R"][i]
        p.beam(fL[0], fR[0], **STIFF)
        if fL[1] != fL[0]:
            p.beam(fL[1], fR[1], **STIFF)
            p.beam(fL[0], fL[1], **STIFF); p.beam(fR[0], fR[1], **STIFF)
            p.beam(fL[0], fR[1], **STIFF); p.beam(fL[1], fR[0], **STIFF)
        if i + 1 < nst:
            gL, gR = plates["L"][i + 1], plates["R"][i + 1]
            for a_, b_ in ((fL, gL), (fR, gR)):
                p.beam(a_[0], b_[0], **STIFF); p.beam(a_[1], b_[1], **STIFF)
                p.beam(a_[0], b_[1], **STIFF); p.beam(a_[1], b_[0], **STIFF)
            p.beam(fL[0], gR[1], **STIFF); p.beam(fR[0], gL[1], **STIFF)
    root = plates["L"][0] + plates["R"][0]
    for n in root:
        connect(p, n, hb + ["jpL", "jpR"], **STIFF)
    connect(p, "rt1", [plates["L"][1][0], plates["R"][1][0]], **STIFF)
    # fork collision skin: outer & inner faces of each plate, front & gripping edges
    for sd in "LR":
        rows = plates[sd]
        cen = np.mean([veh.pos(n) for r in rows for n in r], axis=0)
        sx = 1 if sd == "L" else -1
        cen_out = cen - np.array([sx * 0.3, 0, 0])  # outward normal points away from plate centre laterally
        for i in range(nst - 1):
            a, b_ = rows[i], rows[i + 1]
            for cpos in (cen + np.array([sx * 0.2, 0, 0]), cen - np.array([sx * 0.2, 0, 0])):
                pass
    _fork_skin(p, veh, plates)

    # ---- moving jaw (single thick plate)
    jl = T.jaw_line
    jrows = []
    for i, (a, b) in enumerate(jl):
        if i == 0:
            continue
        if i < len(jl) - 1:
            ta = np.subtract(jl[i + 1], jl[i - 1])
        else:
            ta = np.subtract(jl[i], jl[i - 1])
        ta = ta / np.linalg.norm(ta)
        nb = np.array([-ta[1], ta[0]])
        depth = T.jaw_depth * (1.0 - 0.6 * (i - 1) / (len(jl) - 2))
        row = []
        for sd, sx in (("L", 1), ("R", -1)):
            xp = sx * T.jaw_plate_half
            if i < len(jl) - 1:
                row.append(jaw.n(f"jw{i}f{sd}", W("jaw", a + nb[0] * depth / 2, b + nb[1] * depth / 2, xp), frictionCoef=1.6))
                row.append(jaw.n(f"jw{i}k{sd}", W("jaw", a - nb[0] * depth / 2, b - nb[1] * depth / 2, xp), frictionCoef=1.6))
            else:
                n_ = jaw.n(f"jw{i}t{sd}", W("jaw", a, b, xp), frictionCoef=1.6)
                row += [n_, n_]
        jrows.append(row)  # [fL, kL, fR, kR]
    for i, r in enumerate(jrows):
        fL, kL, fR, kR = r
        p.beam(fL, fR, **STIFF)
        if fL != kL:
            p.beam(kL, kR, **STIFF); p.beam(fL, kL, **STIFF); p.beam(fR, kR, **STIFF)
            p.beam(fL, kR, **STIFF); p.beam(kL, fR, **STIFF)
        if i + 1 < len(jrows):
            g = jrows[i + 1]
            for x_ in range(4):
                p.beam(r[x_], g[x_], **STIFF)
            p.beam(fL, g[1], **STIFF); p.beam(kL, g[0], **STIFF)
            p.beam(fR, g[3], **STIFF); p.beam(kR, g[2], **STIFF)
            p.beam(fL, g[3], **STIFF); p.beam(kR, g[0], **STIFF)
    jlev = jaw.n("jl", W("jaw", *T.jaw_lever), weight=30.0 + M.CYL["jaw"].mass / 2, collision=False)
    connect(p, "jl", ["jpL", "jpR"] + jrows[0], **JOINT)
    for n in jrows[0] + jrows[1]:
        connect(p, n, ["jpL", "jpR"], **JOINT)
    _jaw_skin(p, veh, jrows)

    # rotator torsion hydros (stator k <-> head ring k about rt0 -> rt1)
    for k in (0, 2, 4):
        p.torsion_hydro(f"rst{k}", "rt0", "rt1", f"hr{k}", spring=2.6e6, damp=1.6e3,
                        strength=FLT_MAX, deform=FLT_MAX,
                        inputSource="fd_rot", inLimit=-M.RANGE.rot[1], outLimit=M.RANGE.rot[1],
                        inRate=3.0, outRate=3.0, autoCenterRate=3.0, inputFactor=1)
    add_cylinder_hydro(p, "jaw", "jcb", "jl")
    props_meta += MESH.cylinder_props("fd25_dismantling_shear", "jaw", [("jcb", "jl", "hr0", "jpL")])
    return head


def _fork_skin(p, veh, plates):
    for sd in "LR":
        rows = plates[sd]
        sx = 1 if sd == "L" else -1
        pts = [veh.pos(n) for r in rows for n in r]
        cen = np.mean(pts, axis=0)
        # faces of the plate polygon (front edge f, gripping edge k)
        outer_c = cen - np.array([sx * 0.4, 0, 0])
        inner_c = cen + np.array([sx * 0.4, 0, 0])
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            quad = [a[0], b[0], b[1], a[1]] if b[0] != b[1] else [a[0], b[0], a[1]]
            # outer face (normal away from gap)
            if len(quad) == 4:
                tri_outward(p, veh, quad[0], quad[1], quad[2], outer_c)
                tri_outward(p, veh, quad[0], quad[2], quad[3], outer_c)
                tri_outward(p, veh, quad[0], quad[2], quad[1], inner_c)
                tri_outward(p, veh, quad[0], quad[3], quad[2], inner_c)
            else:
                tri_outward(p, veh, quad[0], quad[1], quad[2], outer_c)
                tri_outward(p, veh, quad[0], quad[2], quad[1], inner_c)
    # bridging faces across both plates (front edge band & gripping edge band)
    L, R = plates["L"], plates["R"]
    cen = np.mean([veh.pos(n) for r in L + R for n in r], axis=0)
    for i in range(len(L) - 1):
        for e in (0, 1):
            a1, a2 = L[i][e], L[i + 1][e]
            b1, b2 = R[i][e], R[i + 1][e]
            tri_outward(p, veh, a1, a2, b2, cen)
            tri_outward(p, veh, a1, b2, b1, cen)


def _jaw_skin(p, veh, jrows):
    cen = np.mean([veh.pos(n) for r in jrows for n in r], axis=0)
    for i in range(len(jrows) - 1):
        r, g = jrows[i], jrows[i + 1]
        fL, kL, fR, kR = r
        gfL, gkL, gfR, gkR = g
        # left face, right face, front edge, gripping edge
        for q in ((fL, gfL, gkL, kL), (fR, gfR, gkR, kR), (fL, gfL, gfR, fR), (kL, gkL, gkR, kR)):
            a, b, c, d = q
            if b == c:
                tri_outward(p, veh, a, b, d, cen)
            else:
                tri_outward(p, veh, a, b, c, cen)
                tri_outward(p, veh, a, c, d, cen)


# ============================================================ clamp arm
def build_clamp(veh, p, bodies, props_meta):
    Cl = M.CLAMP
    body = Body(p, "clamp", "fd_clamp", Cl.mass)
    bodies.append(body)
    stations = (0.30, 1.00, 1.65, Cl.arm_len)
    arms = {}
    for sd, sx in (("L", 1), ("R", -1)):
        rings = []
        for i, s in enumerate(stations):
            xc = sx * (Cl.pivot_x + (Cl.fork_half - Cl.pivot_x) * s / Cl.arm_len)
            r = [body.n(f"ca{sd}{i}bo", W("clamp", s, -0.16, xc + sx * 0.09)),
                 body.n(f"ca{sd}{i}bi", W("clamp", s, -0.16, xc - sx * 0.09)),
                 body.n(f"ca{sd}{i}ti", W("clamp", s, 0.16, xc - sx * 0.09)),
                 body.n(f"ca{sd}{i}to", W("clamp", s, 0.16, xc + sx * 0.09))]
            rings.append(r)
        truss(p, rings, **STIFF)
        skin_rings(p, veh, rings)
        arms[sd] = rings
        connect(p, f"cp{sd}", rings[0], **JOINT)
        o = "R" if sd == "L" else "L"
        connect(p, f"cp{o}", rings[0][1:3], **JOINT)
    # cross bracing between the two arms (front crossbar + mid tube)
    for i in (1, 3):
        a, b = arms["L"][i], arms["R"][i]
        p.beam(a[1], b[1], **STIFF); p.beam(a[2], b[2], **STIFF)
        p.beam(a[1], b[2], **STIFF); p.beam(a[2], b[1], **STIFF)
    p.beam(arms["L"][0][1], arms["R"][1][2], **STIFF); p.beam(arms["R"][0][1], arms["L"][1][2], **STIFF)
    p.beam(arms["L"][2][1], arms["R"][3][2], **STIFF); p.beam(arms["R"][2][1], arms["L"][3][2], **STIFF)
    # crossbar mid nodes for collision
    xb = []
    for k, x in enumerate((0.30, -0.30)):
        b_ = body.n(f"cx{k}b", W("clamp", Cl.arm_len, -0.16, x))
        t_ = body.n(f"cx{k}t", W("clamp", Cl.arm_len, 0.16, x))
        xb.append((b_, t_))
        connect(p, b_, arms["L"][3][:3] + arms["R"][3][:3] + arms["L"][2][:2] + arms["R"][2][:2], **STIFF)
        connect(p, t_, arms["L"][3][2:] + arms["R"][3][2:] + [b_] + arms["L"][2][2:] + arms["R"][2][2:], **STIFF)
    p.beam(xb[0][0], xb[1][0], **STIFF); p.beam(xb[0][1], xb[1][1], **STIFF)
    p.beam(xb[0][0], xb[1][1], **STIFF)
    cen = np.mean([veh.pos(n) for pr in xb for n in pr], axis=0) + np.array([0, 0.3, 0])
    tri_outward(p, veh, xb[0][0], xb[1][0], xb[1][1], cen); tri_outward(p, veh, xb[0][0], xb[1][1], xb[0][1], cen)
    # fingers (downward, hooked back)
    fpts = ((Cl.arm_len + 0.13, -0.20), (Cl.arm_len + 0.10, -0.16 - Cl.finger_len + 0.05),
            (Cl.arm_len - 0.06, -0.16 - Cl.finger_len + 0.10))
    for sd, sx in (("L", 1), ("R", -1)):
        x0 = sx * Cl.fork_half
        fn = []
        for k, (s, t) in enumerate(fpts):
            fn.append((body.n(f"cf{sd}{k}o", W("clamp", s, t, x0 + sx * 0.05), frictionCoef=1.5),
                       body.n(f"cf{sd}{k}i", W("clamp", s, t, x0 - sx * 0.05), frictionCoef=1.5)))
        end = arms[sd][3]
        for a, b in fn:
            connect(p, a, end + [b], **STIFF)
            connect(p, b, end, **STIFF)
        for k in range(len(fn) - 1):
            p.beam(fn[k][0], fn[k + 1][0], **STIFF); p.beam(fn[k][1], fn[k + 1][1], **STIFF)
            p.beam(fn[k][0], fn[k + 1][1], **STIFF)
        p.beam(fn[0][0], fn[2][1], **STIFF)
        cen = np.mean([veh.pos(n) for pr in fn for n in pr], axis=0)
        for side_ in (0, 1):
            ids = [pr[side_] for pr in fn]
            cc = cen + np.array([(1 if side_ == 1 else -1) * sx * 0.3, 0, 0])
            tri_outward(p, veh, ids[0], ids[1], ids[2], cc)
        tri_outward(p, veh, fn[0][0], fn[1][0], fn[1][1], cen); tri_outward(p, veh, fn[0][0], fn[1][1], fn[0][1], cen)
        tri_outward(p, veh, fn[1][0], fn[2][0], fn[2][1], cen); tri_outward(p, veh, fn[1][0], fn[2][1], fn[1][1], cen)
    # cylinder rod lugs
    for sd, sx in (("L", 1), ("R", -1)):
        lug = body.n(f"ccr{sd}", W("clamp", *Cl.cyl_rod_on_arm, x=sx * Cl.cyl_x),
                     weight=40.0 + M.CYL["clamp"].mass / 2, collision=False)
        connect(p, lug, nearest(veh, veh.pos(lug), arms["L"][0] + arms["L"][1] + arms["R"][0] + arms["R"][1], 6)
                + ["cpL", "cpR"], **JOINT)
        add_cylinder_hydro(p, "clamp", f"ccb{sd}", f"ccr{sd}")
    props_meta += MESH.cylinder_props("fd25_clamparm", "clamp", [("ccbL", "ccrL", "ccbR", "ccrR"),
                                                                 ("ccbR", "ccrR", "ccbL", "ccrL")])
    return body


# ============================================================ hydraulics / controller parts
def controller_params(fast=False):
    H = M.HYD
    lim = K.cylinder_ranges()
    spawn = K.solve()
    chans = OrderedDict()
    for name, key in (("boom", "boom_l"), ("stick", "stick"), ("tilt", "tilt"), ("jaw", "jaw"), ("clamp", "clamp_l")):
        c = M.CYL[name]
        r = lim[key]
        chans[name] = OrderedDict([
            ("type", "linear"), ("electrics", f"fd_{name}"),
            ("lengthMin", r4(r["min"])), ("lengthMax", r4(r["max"])), ("lengthSpawn", r4(r["spawn"])),
            ("maxSpeed", r4(c.max_speed * (1.6 if fast else 1.0))),
            ("pistonArea", r4(math.pi * c.bore ** 2 / 4 * c.count)),
            ("rodArea", r4(math.pi * (c.bore ** 2 - c.rod ** 2) / 4 * c.count)),
            ("accel", r4((2.4 if fast else 1.6) * c.max_speed)),
        ])
    chans["slew"] = OrderedDict([("type", "angular"), ("electrics", "fd_slew"),
                                 ("angleLimit", r4(M.RANGE.slew[1])),
                                 ("maxSpeed", r4(H.slew_speed * (1.5 if fast else 1.0))),
                                 ("accel", r4(H.slew_accel * (1.6 if fast else 1.0))),
                                 ("decel", r4(H.slew_decel * (1.6 if fast else 1.0))),
                                 ("flowCoef", 0.0018)])
    chans["rot"] = OrderedDict([("type", "angular"), ("electrics", "fd_rot"),
                                ("angleLimit", r4(M.RANGE.rot[1])),
                                ("maxSpeed", r4(H.rot_speed * (1.5 if fast else 1.0))),
                                ("accel", r4(H.rot_accel * (1.5 if fast else 1.0))),
                                ("decel", r4(H.rot_accel * (1.8 if fast else 1.2))),
                                ("flowCoef", 0.0009)])
    p = OrderedDict([
        ("fileName", "fd25Hydraulics"), ("name", "fd25Hydraulics"),
        ("pumpFlowMax", r4(H.pump_flow_max * (1.6 if fast else 1.0))),
        ("idleRPM", H.idle_rpm), ("workRPM", H.work_rpm), ("maxRPM", H.max_rpm),
        ("precisionScale", H.precision_scale), ("spoolTau", H.spool_tau), ("cushion", H.cushion),
        ("travelSpeedLow", r4(H.travel_speed_low)), ("travelSpeedHigh", r4(H.travel_speed_high)),
        ("trackDirection", 1), ("slewSign", 1), ("rotSign", 1),
        ("autoCalibrate", True), ("loadHolding", False), ("holdTau", 1.5), ("levelGain", 3.0), ("levelRateTau", 0.08), ("stickAxis", ["bnL", "snL"]),
        ("antiWindup", 0.035),
        ("slewAxis", ["sl0", "sl1"]), ("slewRefUC", "sst0"), ("slewRefSS", "ssr0"),
        ("rotAxis", ["rt0", "rt1"]), ("rotRefStator", "rst0"), ("rotRefHead", "hr0"),
        ("attAxis", ["snL", "snR"]),
        ("attPitchNodes", ["rt0", "rt1"]),
        ("soundNodeEngine", "hd0"), ("soundNodePump", "sd23t"), ("soundNodeTrack", "uc_ref"),
        ("beaconElectrics", "lightbar"),
        ("shoePitch", M.UC.shoe_pitch), ("idlerRadius", M.UC.idler_r),
        ("sprocketRadius", r4(MESH.SPROCKET_PITCH_R)),
        ("tiltLengthPerRad", r4(tilt_gain())),
        ("soundDir", f"vehicles/{M.MODEL_ID}/sounds/"),
        ("pumpEvent", "event:>Vehicle>Hydraulics>Pump_Big"),
        ("travelAlarmEvent", "event:>Vehicle>Electrics>Reverse>Beep_01"),
        ("channels", chans),
        ("poses", OrderedDict([
            ("stow", OrderedDict([("boom", 0.62), ("stick", 0.97), ("tilt", 0.15), ("jaw", 0.95), ("clamp", 0.0),
                                  ("slew", 0.0), ("rot", 0.0)])),
            ("work", OrderedDict([("boom", 0.75), ("stick", 0.88), ("tilt", 0.38), ("jaw", 0.35), ("clamp", 0.75),
                                  ("slew", 0.0), ("rot", 0.0)])),
        ])),
    ])
    return p


def tilt_gain():
    """d(tilt cylinder length)/d(attachment absolute angle) at spawn [m/rad]."""
    s0 = K.solve()
    d = 1e-4
    s1 = K.solve({"att_rel": s0["pose"]["att_rel"] + d})
    return (s1["cyl"]["tilt"][2] - s0["cyl"]["tilt"][2]) / d


def build_hydraulics_parts(veh, p_std, p_fast):
    for part, fast in ((p_std, False), (p_fast, True)):
        params = controller_params(fast)
        part.section("controller", [["fileName"], ["fd25Hydraulics", {"name": "fd25Hydraulics"}]])
        part.section("fd25Hydraulics", params)


# =============================================================================
def main(outdir=None):
    veh, bodies, props_meta = build()
    os.makedirs(VDIR, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    P = veh.parts
    files = OrderedDict([
        ("ferrovore_fd25_undercarriage.jbeam", ["ferrovore_fd25_undercarriage"]),
        ("fd25_superstructure.jbeam", ["fd25_superstructure"]),
        ("fd25_cab.jbeam", ["fd25_cab_standard", "fd25_cab_elevated"]),
        ("fd25_boom.jbeam", ["fd25_boom_mono"]),
        ("fd25_stick.jbeam", ["fd25_stick_hd"]),
        ("fd25_attachment_shear.jbeam", ["fd25_dismantling_shear"]),
        ("fd25_clamparm.jbeam", ["fd25_clamparm", "fd25_clamparm_none"]),
        ("fd25_hydraulics.jbeam", ["fd25_hydraulics_standard", "fd25_hydraulics_arcade"]),
    ])
    # cameras / ref nodes on the main part
    pm = P["ferrovore_fd25_undercarriage"]
    pm.section("refNodes", [["ref:", "back:", "left:", "up:"], ["uc_ref", "uc_back", "cbL1b", "uc_up"]])
    pm.section("cameraExternal", OrderedDict([("distance", 15.0), ("distanceMin", 7.0),
                                              ("offset", {"x": 0.0, "y": -1.2, "z": 2.4}), ("fov", 62)]))
    # emissive material switching (lua/common/jbeam/materials.lua): work lights follow the
    # headlight switch, the amber beacon follows the controller's pulsing fd_beacon electric
    pm.section("glowMap", OrderedDict([
        ("fd25_lamp", OrderedDict([("simpleFunction", {"lowbeam": 1.0, "highbeam": 1.0}),
                                   ("off", "fd25_lamp"), ("on", "fd25_lamp_on")])),
        ("fd25_beacon", OrderedDict([("simpleFunction", "fd_beacon"),
                                     ("off", "fd25_beacon"), ("on", "fd25_beacon_on")])),
    ]))
    for fn, names in files.items():
        write_jbeam(os.path.join(VDIR, fn), [P[n] for n in names])
    with open(os.path.join(BUILD, "nodes.json"), "w") as f:
        json.dump(veh.node_index(), f, indent=0)
    with open(os.path.join(BUILD, "props.json"), "w") as f:
        json.dump(props_meta, f, indent=1)
    bod = {b.name: dict(nodes=b.nodes, mass=b.mass, group=b.group) for b in bodies}
    with open(os.path.join(BUILD, "bodies.json"), "w") as f:
        json.dump(bod, f, indent=0)
    # summary
    nn = len(veh.nodes)
    nb = sum(len(p.beam_rows) for p in P.values())
    nt = sum(len(p.tri_rows) for p in P.values())
    mass = sum(n.opts["nodeWeight"] for n in veh.nodes.values() if not n.part.startswith("fd25_cab_elevated"))
    print(f"nodes {nn}  beams {nb}  triangles {nt}  mass(default cfg, incl. both cabs excluded elevated) {mass:.0f} kg")
    return veh


if __name__ == "__main__":
    main(*sys.argv[1:])
