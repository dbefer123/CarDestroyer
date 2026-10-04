"""Headless 2 kHz soft-body co-simulation of the generated FD-25 JBeam.

This is *not* BeamNG, but it reproduces the parts of its vehicle physics the
mod relies on closely enough to catch real problems before shipping:
  * nodes / beams (spring-damper), hydros incl. rate limits and the
    forceLimit "flex" relief logic from lua/vehicle/hydros.lua,
  * torsion hydros (dihedral spring with commanded rest angle),
  * ground plane contact with regularised Coulomb friction,
  * rotators driven by the two electric travel motors + per-side brakes.
The real controller (fd25Hydraulics.lua) runs in LuaJIT via lua_env.
"""
import math

import numpy as np

from tools.spec import machine as M
from tools.validate.jbeam_load import assemble

DT = M.PHYS_DT
G = 9.81


def fnum(v, d):
    if v is None or v == "FLT_MAX":
        return d
    return float(v)


def dihedral_and_grad(P1, P2, P3, P4):
    """Vectorised dihedral angle (same convention as check_jbeam.dihedral) and gradients."""
    b1, b2, b3 = P2 - P1, P3 - P2, P4 - P3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = np.linalg.norm(b2, axis=1, keepdims=True)
    m1 = np.cross(n1, b2 / b2n)
    x = np.sum(n1 * n2, axis=1)
    y = np.sum(m1 * n2, axis=1)
    ang = np.arctan2(y, x)
    n1sq = np.sum(n1 * n1, axis=1, keepdims=True) + 1e-12
    n2sq = np.sum(n2 * n2, axis=1, keepdims=True) + 1e-12
    g1 = b2n * n1 / n1sq
    g4 = -b2n * n2 / n2sq
    b2sq = b2n ** 2
    f1 = -np.sum(b1 * b2, axis=1, keepdims=True) / b2sq
    f3 = -np.sum(b3 * b2, axis=1, keepdims=True) / b2sq
    g2 = (f1 - 1) * g1 - f3 * g4
    g3 = (f3 - 1) * g4 - f1 * g1
    return ang, (g1, g2, g3, g4)


class Sim:
    def __init__(self, config=None, ground_k=2.5e6, mu_ground=1.0, substeps=2):
        V = assemble(config)
        self.dt = DT / substeps
        self.V = V
        names = list(V["nodes"])
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        N = len(names)
        self.N = N
        nd = V["nodes"]
        self.x = np.array([nd[n]["pos"] for n in names], float)
        self.x0 = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.m = np.array([fnum(nd[n].get("nodeWeight"), 25.0) for n in names])
        self.coll = np.array([bool(nd[n].get("collision", True)) and bool(nd[n].get("staticCollision", True))
                              for n in names])
        self.mu = np.array([fnum(nd[n].get("frictionCoef"), 1.0) for n in names]) * mu_ground
        self.ground_k = ground_k
        self.ground_c = 2 * 0.6 * np.sqrt(ground_k * self.m)

        # beams + hydros share one array; hydros get variable rest length
        I, J, K, C, L0 = [], [], [], [], []
        self.hydro_rows = []
        for b in V["beams"]:
            I.append(self.idx[b["id1"]]); J.append(self.idx[b["id2"]])
            K.append(fnum(b.get("beamSpring"), 4.3e6)); C.append(fnum(b.get("beamDamp"), 580))
        nb = len(I)
        for h in V["hydros"]:
            I.append(self.idx[h["id1"]]); J.append(self.idx[h["id2"]])
            K.append(fnum(h.get("beamSpring"), 4.3e6)); C.append(fnum(h.get("beamDamp"), 580))
        self.I, self.J = np.array(I), np.array(J)
        self.K, self.C = np.array(K), np.array(C)
        self.Lref = np.linalg.norm(self.x[self.I] - self.x[self.J], axis=1)
        self.L0 = self.Lref.copy()
        self.nbeams = nb
        # hydro metadata (beam cid = index in this array)
        self.hyd = []
        for k, h in enumerate(V["hydros"]):
            cid = nb + k
            L = self.Lref[cid]
            inl = fnum(h.get("inExtent"), 0) / L if h.get("inExtent") is not None else fnum(h.get("inLimit"), 0)
            outl = fnum(h.get("outExtent"), 0) / L if h.get("outExtent") is not None else fnum(h.get("outLimit"), 2)
            self.hyd.append(dict(cid=cid, src=h["inputSource"], inl=inl, outl=outl,
                                 rate_in=fnum(h.get("inRate"), 2), rate_out=fnum(h.get("outRate"), 2),
                                 flim=fnum(h.get("forceLimit"), 1e30), state=1.0, offset=0.0,
                                 data=h))
        # torsion hydros
        T = V["thydros"]
        self.tq = np.array([[self.idx[t[k]] for k in ("id1", "id2", "id3", "id4")] for t in T], int).reshape(-1, 4)
        self.tk = np.array([fnum(t.get("spring"), 0) for t in T])
        self.tc = np.array([fnum(t.get("damp"), 0) for t in T])
        self.tsrc = [t.get("inputSource") for t in T]
        self.tin = np.array([fnum(t.get("inLimit"), -1) for t in T])
        self.tout = np.array([fnum(t.get("outLimit"), 1) for t in T])
        self.trate = np.array([fnum(t.get("inRate"), 2) for t in T])
        P = [self.x[self.tq[:, k]] for k in range(4)]
        self.t0 = dihedral_and_grad(*P)[0] if len(T) else np.zeros(0)
        self.tstate = np.zeros(len(T))
        self.tprev = self.t0.copy()

        # rotators (road wheels)
        self.wheels = []
        groups = {}
        for n in names:
            g = nd[n].get("group", "")
            for gg in (g if isinstance(g, list) else [g]):
                groups.setdefault(gg, []).append(self.idx[n])
        for r in V["rotators"]:
            g = r["group"][0] if isinstance(r["group"], list) else r["group"]
            self.wheels.append(dict(name=r["name"], ring=np.array(groups[g]), a1=self.idx[r["node1"]],
                                    a2=self.idx[r["node2"]], arm=self.idx[r["nodeArm"]],
                                    brake=r.get("brakeElectricsName", "brake"),
                                    brakeT=fnum(r.get("brakeTorque"), 0), side=r["name"][3],
                                    radius=fnum(r.get("radius"), 0.3), omega=0.0))
        self.wheel_ids = {w["name"]: i for i, w in enumerate(self.wheels)}
        self.motors = {}
        for side in "LR":
            md = V["sections"].get(f"travelMotor{side}")
            if md:
                curve = np.array(md["torque"][1:], float)
                self.motors[side] = dict(curve=curve, thr=md["electricsThrottleName"], maxRPM=md["maxRPM"])
        self.gear = 62.0
        for row in V["sections"].get("powertrain", [])[1:]:
            if row[0] == "shaft" and isinstance(row[-1], dict):
                self.gear = row[-1].get("gearRatio", self.gear)
        self.electrics = {}
        self.t = 0.0
        self.ref = self.idx["uc_ref"]
        self.back = self.idx["uc_back"]
        self.stress = np.zeros(len(self.I))

    # ---------------------------------------------------------------- interface for lua_env
    @property
    def node_cids(self):
        return self.idx

    @property
    def hydros(self):
        out = []
        for h in self.hyd:
            out.append({"inputSource": h["src"], "beamCID": h["cid"], "forceLimit": h["flim"],
                        "beamSpring": float(self.K[h["cid"]])})
        return out

    @property
    def beams(self):
        return {h["cid"]: {"beamSpring": float(self.K[h["cid"]])} for h in self.hyd}

    def node_pos(self, cid):
        return tuple(self.x[cid])

    def beam_length(self, cid):
        return float(np.linalg.norm(self.x[self.I[cid]] - self.x[self.J[cid]]))

    def beam_stress(self, cid):
        return float(self.stress[cid])

    def velocity(self):
        return tuple(self.v[self.ref])

    def direction(self):
        d = self.x[self.ref] - self.x[self.back]
        return tuple(d / np.linalg.norm(d))

    def wheel_speed(self, wid):
        w = self.wheels[wid]
        return w["omega"] * w["radius"]

    # ---------------------------------------------------------------- physics
    def _hydro_targets(self, dt):
        for h in self.hyd:
            v = float(self.electrics.get(h["src"], 0.0) or 0.0)
            v = max(-1.0, min(1.0, v + h["offset"]))
            target = 1 + v * (h["outl"] - 1) if v >= 0 else 1 + v * (1 - h["inl"])
            s = h["state"]
            rate = (h["rate_out"] if target > s else h["rate_in"]) * dt
            s += max(-rate, min(rate, target - s))
            h["state"] = s
            self.L0[h["cid"]] = self.Lref[h["cid"]] * s

    def _hydro_flex(self, dt_gfx):
        # replicate hydros.lua forceLimit: shift the input offset so force stays near the limit
        for h in self.hyd:
            stress = self.stress[h["cid"]]
            if abs(stress) > h["flim"]:
                mult = min(h["outl"] - 1, 1 - h["inl"])
                h["offset"] += math.copysign(abs(stress) - h["flim"], stress) / (self.K[h["cid"]] * self.Lref[h["cid"]] * max(mult, 1e-3))
            else:
                minrate = min(h["rate_in"], h["rate_out"]) * dt_gfx
                h["offset"] = math.copysign(max(0.0, abs(h["offset"]) - minrate), h["offset"])

    def _torsion_targets(self, dt):
        for k, src in enumerate(self.tsrc):
            v = float(self.electrics.get(src, 0.0) or 0.0)
            v = max(-1.0, min(1.0, v))
            target = v * self.tout[k] if v >= 0 else -v * self.tin[k]
            r = self.trate[k] * dt
            self.tstate[k] += max(-r, min(r, target - self.tstate[k]))

    def step(self):
        dt = self.dt
        x, v = self.x, self.v
        F = np.zeros_like(x)
        F[:, 2] -= self.m * G
        # beams
        d = x[self.J] - x[self.I]
        L = np.linalg.norm(d, axis=1)
        u = d / L[:, None]
        dv = v[self.J] - v[self.I]
        vrel = np.sum(dv * u, axis=1)
        f = self.K * (L - self.L0) + self.C * vrel
        self.stress = f
        fv = f[:, None] * u
        for ax in range(3):
            F[:, ax] += np.bincount(self.I, weights=fv[:, ax], minlength=self.N)
            F[:, ax] -= np.bincount(self.J, weights=fv[:, ax], minlength=self.N)
        # torsion hydros
        if len(self.tq):
            P = [x[self.tq[:, k]] for k in range(4)]
            ang, grads = dihedral_and_grad(*P)
            rel = ang - self.t0
            rel = (rel + np.pi) % (2 * np.pi) - np.pi
            dang = ((ang - self.tprev + np.pi) % (2 * np.pi) - np.pi) / dt
            self.tprev = ang
            torque = -self.tk * (rel - self.tstate) - self.tc * dang
            for k in range(4):
                np.add.at(F, self.tq[:, k], torque[:, None] * grads[k])
        # wheels: drive + brake torques
        if self.wheels:
            self._wheel_forces(F)
        # ground
        z = x[:, 2]
        pen = (z < 0) & self.coll
        if pen.any():
            ii = np.nonzero(pen)[0]
            fn = self.ground_k * (-z[ii]) - self.ground_c[ii] * v[ii, 2]
            fn = np.maximum(fn, 0)
            F[ii, 2] += fn
            vt = v[ii, :2]
            sp = np.linalg.norm(vt, axis=1)
            fmax = self.mu[ii] * fn
            ft = -vt / np.maximum(sp, 0.03)[:, None] * fmax[:, None]
            F[ii, :2] += ft
        # integrate (semi-implicit Euler)
        v += F / self.m[:, None] * dt
        x += v * dt
        self.t += dt

    def _wheel_forces(self, F):
        x, v = self.x, self.v
        side_omega = {"L": [], "R": []}
        for w in self.wheels:
            a1, a2 = x[w["a1"]], x[w["a2"]]
            ax = a2 - a1
            ax /= np.linalg.norm(ax)
            c = 0.5 * (a1 + a2)
            vc = 0.5 * (v[w["a1"]] + v[w["a2"]])
            r = x[w["ring"]] - c
            r -= np.outer(r @ ax, ax)
            vr = v[w["ring"]] - vc
            num = np.sum(np.cross(r, vr) @ ax)
            den = np.sum(r * r)
            w["omega"] = num / den
            w["r"], w["ax"], w["den"] = r, ax, den
            side_omega[w["side"]].append(w["omega"])
        for side, m in self.motors.items():
            thr = float(self.electrics.get(m["thr"], 0.0) or 0.0)
            om = np.mean(side_omega[side]) if side_omega[side] else 0.0
            rpm = abs(om * self.gear) * 60 / (2 * math.pi)
            tq = np.interp(rpm, m["curve"][:, 0], m["curve"][:, 1], right=0.0)
            m["wheel_torque"] = thr * tq * self.gear / 4.0
        for w in self.wheels:
            T = self.motors.get(w["side"], {}).get("wheel_torque", 0.0)
            b = float(self.electrics.get(w["brake"], 0.0) or 0.0)
            Tb_max = w["brakeT"] * b
            Tb = -np.clip(4.0e4 * w["omega"], -Tb_max, Tb_max)
            Ttot = T + Tb
            fr = Ttot * np.cross(w["ax"], w["r"]) / w["den"]
            F[w["ring"]] += fr
            # reaction couple on the frame via nodeArm, balanced on the axle nodes
            ra = x[w["arm"]] - 0.5 * (x[w["a1"]] + x[w["a2"]])
            ra -= (ra @ w["ax"]) * w["ax"]
            fa = -Ttot * np.cross(w["ax"], ra) / (ra @ ra)
            F[w["arm"]] += fa
            F[w["a1"]] -= 0.5 * fa
            F[w["a2"]] -= 0.5 * fa

    # ---------------------------------------------------------------- helpers
    def run(self, seconds, lua=None, gfx_hz=60, on_gfx=None):
        steps_per_gfx = int(round(1 / (gfx_hz * self.dt)))
        n = int(seconds / self.dt)
        dt_gfx = steps_per_gfx * self.dt
        for i in range(n):
            if i % steps_per_gfx == 0:
                if lua is not None:
                    lua.step(dt_gfx)
                    vals = lua.L.globals().electrics["values"]
                    for k in list(vals.keys()):
                        val = vals[k]
                        if isinstance(val, (int, float)):
                            self.electrics[k] = val
                if on_gfx:
                    on_gfx(self)
                self._hydro_flex(dt_gfx)
                self._torsion_targets(dt_gfx)
            self._hydro_targets(self.dt)
            self.step()
            if not np.isfinite(self.x).all():
                raise FloatingPointError(f"simulation exploded at t={self.t:.3f}")

    def max_strain(self):
        L = np.linalg.norm(self.x[self.I] - self.x[self.J], axis=1)
        s = np.abs(L - self.L0) / self.L0
        k = int(np.argmax(s[:self.nbeams]))
        return float(s[:self.nbeams].max()), self.names[self.I[k]], self.names[self.J[k]]

    def kinetic(self):
        return float(0.5 * np.sum(self.m * np.sum(self.v ** 2, axis=1)))

    def pos(self, name):
        return self.x[self.idx[name]].copy()
