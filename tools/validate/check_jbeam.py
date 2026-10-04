"""Static validation of the generated JBeam.

Checks references, element sanity, numerical stability at BeamNG's 2 kHz
physics rate, mass / centre of gravity, tipping stability over the slew
range, flexbody / prop references and a full 3D rigidity (degree-of-freedom)
analysis of the assembled mechanism.

Run:  python -m tools.validate.check_jbeam [--quick]
Exit code != 0 when an error is found.
"""
import json
import math
import os
import sys

import numpy as np

from tools.spec import machine as M
from tools import kinematics as K
from tools.validate.jbeam_load import assemble, groups_of, ROOT

DT = M.PHYS_DT


def num(v, default):
    if v == "FLT_MAX" or v is None:
        return default
    return float(v)


def dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.atan2(float(m1 @ n2), float(n1 @ n2))


class Report:
    def __init__(self):
        self.errors, self.warnings, self.info = [], [], []

    def err(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warnings.append(m)

    def note(self, m):
        self.info.append(m)

    def text(self):
        out = []
        for t, lst in (("ERROR", self.errors), ("WARN", self.warnings), ("INFO", self.info)):
            for m in lst:
                out.append(f"[{t}] {m}")
        return "\n".join(out)


def check(config=None, quick=False, rep=None):
    rep = rep or Report()
    V = assemble(config)
    nodes = V["nodes"]
    names = list(nodes)
    idx = {n: i for i, n in enumerate(names)}
    pos = np.array([nodes[n]["pos"] for n in names])
    mass = np.array([num(nodes[n].get("nodeWeight"), 25.0) for n in names])
    rep.note(f"config parts: {', '.join(V['parts'])}")
    rep.note(f"nodes {len(names)}, beams {len(V['beams'])}, hydros {len(V['hydros'])}, "
             f"torsionHydros {len(V['thydros'])}, triangles {len(V['tris'])}, rotators {len(V['rotators'])}")

    # ---------------------------------------------------------- references
    def ref(n, where):
        if n not in idx:
            rep.err(f"{where}: unknown node '{n}'")
            return False
        return True

    edges = []
    for kind, lst in (("beam", V["beams"]), ("hydro", V["hydros"])):
        for b in lst:
            a, c = b["id1"], b["id2"]
            if ref(a, kind) and ref(c, kind):
                L = np.linalg.norm(pos[idx[a]] - pos[idx[c]])
                if L < 0.015:
                    rep.err(f"{kind} {a}-{c} too short ({L:.4f} m)")
                k = num(b.get("beamSpring"), 4.3e6)
                d = num(b.get("beamDamp"), 580)
                edges.append((idx[a], idx[c], k, d, kind, b))
    for t in V["thydros"]:
        for key in ("id1", "id2", "id3", "id4"):
            ref(t[key], "torsionHydro")
    for t in V["tris"]:
        ok = all(ref(t[k], "triangle") for k in ("id1", "id2", "id3"))
        if ok:
            a, b, c = (pos[idx[t[k]]] for k in ("id1", "id2", "id3"))
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if area < 2e-4:
                rep.warn(f"degenerate triangle {t['id1']},{t['id2']},{t['id3']} area {area:.5f}")
    for pr in V["props"]:
        for k in ("idRef", "idX", "idY"):
            ref(pr[k], f"prop {pr['mesh']}")
        a, b, c = (pos[idx[pr[k]]] for k in ("idRef", "idX", "idY"))
        if np.linalg.norm(np.cross(b - a, c - a)) < 1e-4:
            rep.err(f"prop {pr['mesh']} has collinear frame nodes")
    for r in V["rotators"]:
        for k in ("node1", "node2", "nodeArm"):
            ref(r[k], f"rotator {r['name']}")
    # duplicate beams
    seen = {}
    for a, c, k, d, kind, b in edges:
        key = (min(a, c), max(a, c), kind)
        if key in seen:
            rep.warn(f"duplicate {kind} {names[a]}-{names[c]}")
        seen[key] = 1

    # ---------------------------------------------------------- groups / flexbodies
    groups = {}
    for n in names:
        for g in groups_of(nodes[n]):
            groups.setdefault(g, []).append(n)
    for fb in V["flex"]:
        gl = fb["group"] if isinstance(fb.get("group"), list) else [fb.get("group")]
        members = [n for g in gl for n in groups.get(g, [])]
        if len(members) < 3:
            rep.err(f"flexbody {fb['mesh']} groups {gl} have only {len(members)} nodes")
            continue
        P = np.array([pos[idx[n]] for n in members])
        if np.linalg.matrix_rank(P - P.mean(0), tol=1e-3) < 2:
            rep.err(f"flexbody {fb['mesh']} group nodes are collinear")
    for r in V["rotators"]:
        g = r["group"][0] if isinstance(r["group"], list) else r["group"]
        if len(groups.get(g, [])) < 6:
            rep.err(f"rotator {r['name']} group {g} too small")

    # ---------------------------------------------------------- stability at 2 kHz
    worst_w, worst_c = (0, None), (0, None)
    for a, c, k, d, kind, b in edges:
        inv = 1 / mass[a] + 1 / mass[c]
        w = math.sqrt(k * inv) * DT
        cd = d * inv * DT
        if w > worst_w[0]:
            worst_w = (w, f"{kind} {names[a]}-{names[c]} k={k:.3g} m={mass[a]:.1f}/{mass[c]:.1f}")
        if cd > worst_c[0]:
            worst_c = (cd, f"{kind} {names[a]}-{names[c]} d={d:.3g}")
        if w > 1.1:
            rep.err(f"stiffness unstable (w*dt={w:.2f}) {kind} {names[a]}-{names[c]} k={k:.3g} m={mass[a]:.1f}/{mass[c]:.1f}")
        elif w > 0.95:
            rep.warn(f"stiffness close to limit (w*dt={w:.2f}) {kind} {names[a]}-{names[c]}")
        if cd > 1.0:
            rep.err(f"damping unstable (c*dt/m={cd:.2f}) {kind} {names[a]}-{names[c]}")
    rep.note(f"max w*dt {worst_w[0]:.3f} ({worst_w[1]}); max c*dt/m {worst_c[0]:.3f} ({worst_c[1]})")
    for t in V["thydros"]:
        ids = [idx[t[k]] for k in ("id1", "id2", "id3", "id4")]
        ax0, ax1 = pos[ids[1]], pos[ids[2]]
        axis = (ax1 - ax0) / np.linalg.norm(ax1 - ax0)
        for lever in (ids[0], ids[3]):
            v = pos[lever] - ax0
            r = np.linalg.norm(v - (v @ axis) * axis)
            k_lin = num(t.get("spring"), 0) / (r * r)
            w = math.sqrt(k_lin * 2 / mass[lever]) * DT
            c_lin = num(t.get("damp"), 0) / (r * r)
            cd = c_lin * 2 / mass[lever] * DT
            if w > 1.0 or cd > 1.0:
                rep.err(f"torsionHydro {t['id1']}..{t['id4']} unstable (w*dt {w:.2f}, c*dt/m {cd:.2f}, r {r:.2f})")
            elif w > 0.8:
                rep.warn(f"torsionHydro {t['id1']}..{t['id4']} near limit (w*dt {w:.2f}, r {r:.2f})")

    # per-node explicit-integration budget (sum of attached springs / dampers)
    Ksum = np.zeros(len(names)); Csum = np.zeros(len(names))
    for a, c, k, d, kind, b in edges:
        Ksum[a] += k; Ksum[c] += k; Csum[a] += d; Csum[c] += d
    wn = np.sqrt(Ksum / mass) * DT
    cn = Csum * DT / mass
    worst = int(np.argmax(wn)); worstc = int(np.argmax(cn))
    rep.note(f"node budget: max sqrt(sum k/m)*dt {wn[worst]:.2f} ({names[worst]}), max sum c*dt/m {cn[worstc]:.2f} ({names[worstc]})")
    for i in np.nonzero(wn > 1.5)[0]:
        rep.err(f"node {names[i]} too stiff for its mass (sqrt(sum k/m)*dt = {wn[i]:.2f})")
    for i in np.nonzero(cn > 1.0)[0]:
        rep.err(f"node {names[i]} over-damped for its mass (sum c*dt/m = {cn[i]:.2f})")

    # ---------------------------------------------------------- mass / CoG
    total = mass.sum()
    cog = (mass[:, None] * pos).sum(0) / total
    rep.note(f"total mass {total:.0f} kg, CoG ({cog[0]:.3f}, {cog[1]:.3f}, {cog[2]:.3f})")
    if not (24000 < total < 32000):
        rep.warn(f"total mass {total:.0f} kg outside 24-32 t")

    # ---------------------------------------------------------- hydro ranges
    for h in V["hydros"]:
        a, c = idx[h["id1"]], idx[h["id2"]]
        L = np.linalg.norm(pos[a] - pos[c])
        lo, hi = num(h.get("inExtent"), 0), num(h.get("outExtent"), 0)
        if not (lo - 1e-3 <= L <= hi + 1e-3):
            rep.err(f"hydro {h['id1']}-{h['id2']} spawn length {L:.3f} outside [{lo:.3f}, {hi:.3f}]")
        if hi / max(lo, 1e-6) > 1.9:
            rep.warn(f"hydro {h['id1']}-{h['id2']} stroke ratio {hi / lo:.2f}")

    # ---------------------------------------------------------- rigidity / DOF analysis
    if not quick:
        dof_analysis(V, names, idx, pos, edges, rep)
    return rep, V


def dof_analysis(V, names, idx, pos, edges, rep):
    n = len(names)
    rows = []
    for a, c, k, d, kind, b in edges:
        r = np.zeros(3 * n)
        dvec = pos[a] - pos[c]
        dvec /= np.linalg.norm(dvec)
        r[3 * a:3 * a + 3] = dvec
        r[3 * c:3 * c + 3] = -dvec
        rows.append(r)
    for t in V["thydros"]:
        ids = [idx[t[k]] for k in ("id1", "id2", "id3", "id4")]
        P = [pos[i].copy() for i in ids]
        g = np.zeros(3 * n)
        eps = 1e-6
        base = dihedral(*P)
        for j, i in enumerate(ids):
            for ax in range(3):
                Q = [p.copy() for p in P]
                Q[j][ax] += eps
                g[3 * i + ax] = (dihedral(*Q) - base) / eps
        rows.append(g / (np.linalg.norm(g) + 1e-12))
    R = np.array(rows)
    s = np.linalg.svd(R, compute_uv=False)
    tol = s.max() * 1e-7
    rank = int((s > tol).sum())
    dof = 3 * n - rank
    expected = 6 + len(V["rotators"])  # rigid body + free spinning road wheels
    msg = f"rigidity: {3 * n} coords, rank {rank} -> {dof} DOF (expected {expected}: 6 rigid + {len(V['rotators'])} wheel spins)"
    if dof == expected:
        rep.note(msg + "  OK")
    else:
        rep.err(msg)
        # diagnose: nullspace modes, remove rigid & wheel motions, list most-moving nodes
        _, sv, Vt = np.linalg.svd(R, full_matrices=True)
        null = Vt[rank:]
        basis = []
        # rigid body modes
        c = pos.mean(0)
        for ax in range(3):
            v = np.zeros((n, 3)); v[:, ax] = 1; basis.append(v.ravel())
            w = np.zeros(3); w[ax] = 1
            basis.append(np.cross(w, pos - c).ravel())
        for r_ in V["rotators"]:
            g = r_["group"][0] if isinstance(r_["group"], list) else r_["group"]
            a0, a1 = pos[idx[r_["node1"]]], pos[idx[r_["node2"]]]
            ax = (a1 - a0) / np.linalg.norm(a1 - a0)
            v = np.zeros((n, 3))
            for nm in names:
                gg = V["nodes"][nm].get("group")
                gl = gg if isinstance(gg, list) else [gg]
                if g in gl:
                    i = idx[nm]
                    v[i] = np.cross(ax, pos[i] - a0)
            basis.append(v.ravel())
        B = np.array(basis).T
        Q, _ = np.linalg.qr(B)
        resid = null.T - Q @ (Q.T @ null.T)
        U, S2, _ = np.linalg.svd(resid, full_matrices=False)
        for m in range(min(6, (S2 > 1e-6).sum())):
            mode = U[:, m].reshape(n, 3)
            mag = np.linalg.norm(mode, axis=1)
            top = np.argsort(-mag)[:8]
            rep.err("  floppy mode %d involves: %s" % (m, ", ".join(f"{names[i]}({mag[i]:.2f})" for i in top)))


def hydro_anchoring(rep, V):
    """Every cylinder end node must be held by its body along the cylinder axis at first order.
    A node lying in the plane of its anchor nodes while the cylinder pushes out of that plane only
    gets geometric (second-order) stiffness and the driven part wobbles."""
    pos = {k: n["pos"] for k, n in V["nodes"].items()}
    adj = {}
    for b in V["beams"]:
        if b.get("beamType") in ("|BOUNDED", "|SUPPORT"):
            continue
        k = num(b.get("beamSpring"), 4.3e6)
        adj.setdefault(b["id1"], []).append((b["id2"], k))
        adj.setdefault(b["id2"], []).append((b["id1"], k))
    worst = (1.0, "")
    for h in V["hydros"]:
        for end, other in ((h["id1"], h["id2"]), (h["id2"], h["id1"])):
            d = pos[other] - pos[end]
            d /= np.linalg.norm(d)
            nb = adj.get(end, [])
            tot = sum(k for _, k in nb) or 1.0
            along = sum(k * (((pos[n] - pos[end]) / np.linalg.norm(pos[n] - pos[end])) @ d) ** 2 for n, k in nb)
            a = along / tot
            if a < worst[0]:
                worst = (a, f"{end} ({h.get('inputSource')})")
            if a < 0.05:
                rep.err(f"hydro end {end} ({h.get('inputSource')}) is not anchored along the cylinder axis "
                        f"(axial stiffness share {a:.3f})")
    rep.note(f"hydro anchoring: weakest end {worst[1]} with axial stiffness share {worst[0]:.2f} (>= 0.05 required)")


def tipping_check(rep, payload=1500.0):
    """CoG vs. track footprint over slew angles with the arm reaching out + payload in the claw."""
    with open(os.path.join(ROOT, "build", "bodies.json")) as f:
        bodies = json.load(f)
    with open(os.path.join(ROOT, "build", "nodes.json")) as f:
        nodes = json.load(f)
    spawn = K.solve()
    arm_frames = {"boom": "boom", "stick": "stick", "sidelink": "sidelink", "bracket": "att",
                  "head": "head", "jaw": "jaw"}
    worst = 1e9
    for reach in ("spawn", "far"):
        pose = {} if reach == "spawn" else {"boom": math.radians(5), "stick_rel": math.radians(-60)}
        for slew_deg in range(0, 360, 15):
            pp = dict(pose, slew=math.radians(slew_deg))
            s = K.solve(pp)
            tot, mom = 0.0, np.zeros(3)
            for bname, b in bodies.items():
                if bname == "cabe":
                    continue
                for nn in b["nodes"]:
                    nd = nodes[nn]
                    p = np.array(nd["pos"] + [1.0])
                    if bname in arm_frames:
                        fr = arm_frames[bname]
                        T = s["frames"][fr] @ np.linalg.inv(spawn["frames"][fr])
                        p = T @ p
                    elif bname in ("ss", "cw", "cab"):
                        p = K.slew_matrix(pp["slew"]) @ p
                    m = nd["weight"]
                    tot += m
                    mom += m * p[:3]
            tip = s["pts"]["claw_tip_jaw"]
            tot += payload
            mom += payload * tip
            cg = mom / tot
            mx = M.UC.track_x + M.UC.shoe_width / 2 - abs(cg[0])
            my = 2.2 - abs(cg[1])
            worst = min(worst, mx, my)
            if mx < 0.15 or my < 0.15:
                rep.warn(f"tipping margin small ({min(mx, my):.2f} m) at slew {slew_deg} deg, {reach} reach")
    rep.note(f"tipping: worst CoG margin to track edge {worst:.2f} m (payload {payload:.0f} kg in claw)")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    rep, V = check(quick=quick)
    hydro_anchoring(rep, V)
    check({"fd25_cab": "fd25_cab_elevated"}, quick=True, rep=rep)
    check({"fd25_clamparm": "fd25_clamparm_none"}, quick=True, rep=rep)
    check({"fd25_hydraulics": "fd25_hydraulics_arcade"}, quick=True, rep=rep)
    tipping_check(rep)
    print(rep.text())
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    with open(os.path.join(ROOT, "build", "jbeam_report.txt"), "w") as f:
        f.write("JBEAM STATIC CHECK\n" + "=" * 60 + "\n" + rep.text() + "\n")
    sys.exit(1 if rep.errors else 0)
