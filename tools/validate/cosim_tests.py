"""End-to-end co-simulation tests: generated JBeam + real Lua controller + soft-body sim.

Run:  python -m tools.validate.cosim_tests      (writes build/cosim_report.txt)

Every scenario drives the controller through the same input events BeamNG
uses (input.event(...)) and checks physical outcomes on the simulated machine.
"""
import math
import os
import sys
import time

import numpy as np

from tools.validate.sim import Sim
from tools.validate.lua_env import LuaVehicle
from tools.validate.jbeam_load import ROOT

REPORT = os.path.join(ROOT, "build", "cosim_report.txt")


class Suite:
    def __init__(self):
        self.s = Sim()
        self.lua = LuaVehicle(self.s)
        self.lua.init(self.s.V["sections"]["fd25Hydraulics"])
        self.results = []
        self.t0 = time.time()

    # ------------------------------------------------------------ helpers
    def run(self, sec):
        self.s.run(sec, self.lua)

    def hold(self, name, sec, value=1.0):
        self.lua.press(name, value)
        self.run(sec)
        self.lua.press(name, 0.0)

    def cyl(self, src):
        h = [x for x in self.s.hyd if x["src"] == src][0]
        return self.s.beam_length(h["cid"])

    def ch(self, name):
        st, ch, cfg = self.lua.state()
        return ch[name]

    def check(self, name, cond, detail):
        self.results.append((name, bool(cond), detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)

    def strain(self):
        return self.s.max_strain()[0]

    def yaw(self):
        d = self.s.x[self.s.idx["uc_ref"]] - self.s.x[self.s.idx["uc_back"]]
        return math.degrees(math.atan2(d[0], -d[1]))

    # ------------------------------------------------------------ scenarios
    def scenario_settle(self):
        self.run(2.5)
        ke = self.s.kinetic()
        st = self.strain()
        self.check("settle", ke < 4000 and st < 0.05,
                   f"KE {ke:.0f} J, max strain {st * 100:.2f} %, ref z {self.s.pos('uc_ref')[2]:.3f} m")

    def scenario_boom(self):
        L0, z0 = self.cyl("fd_boom"), self.s.pos("bnL")[2]
        self.hold("fd25BoomPlus", 2.5)
        self.run(0.5)
        L1, z1 = self.cyl("fd_boom"), self.s.pos("bnL")[2]
        self.check("boom raise", L1 - L0 > 0.18 and z1 - z0 > 0.8,
                   f"cylinder {L0:.3f}->{L1:.3f} m, boom nose z {z0:.2f}->{z1:.2f} m")
        self.hold("fd25BoomMinus", 2.0)
        self.run(0.5)
        L2 = self.cyl("fd_boom")
        self.check("boom lower", L2 < L1 - 0.3, f"cylinder {L1:.3f}->{L2:.3f} m")
        # back up to about the spawn height so the hanging claw clears the ground for the next tests
        t = 0.0
        while self.cyl("fd_boom") < L0 - 0.01 and t < 3.0:
            self.hold("fd25BoomPlus", 0.1)
            t += 0.1
        self.run(0.5)

    def scenario_stick(self):
        L0, y0 = self.cyl("fd_stick"), self.s.pos("snL")[1]
        self.hold("fd25StickMinus", 2.5)   # stick out
        self.run(0.4)
        L1, y1 = self.cyl("fd_stick"), self.s.pos("snL")[1]
        self.check("stick out", L1 < L0 - 0.4 and y1 < y0 - 0.5,
                   f"cylinder {L0:.3f}->{L1:.3f} m, stick nose y {y0:.2f}->{y1:.2f}")
        self.hold("fd25StickPlus", 2.0)
        self.run(0.4)
        self.check("stick in", self.cyl("fd_stick") > L1 + 0.3, f"cylinder {L1:.3f}->{self.cyl('fd_stick'):.3f} m")

    def scenario_tilt(self):
        L0 = self.cyl("fd_tilt")
        self.hold("fd25TiltPlus", 1.5)
        self.run(0.3)
        L1 = self.cyl("fd_tilt")
        self.hold("fd25TiltMinus", 1.5)
        self.run(0.3)
        L2 = self.cyl("fd_tilt")
        self.check("tilt in/out", L1 > L0 + 0.18 and L2 < L1 - 0.25, f"cylinder {L0:.3f}->{L1:.3f}->{L2:.3f} m")

    def jaw_gap(self):
        return float(np.linalg.norm(self.s.pos("jw5tL") - self.s.pos("fk5tL")))

    def scenario_jaw(self):
        g0, L0 = self.jaw_gap(), self.cyl("fd_jaw")
        self.hold("fd25JawMinus", 1.6)
        self.run(0.3)
        g1, L1 = self.jaw_gap(), self.cyl("fd_jaw")
        self.hold("fd25JawPlus", 2.2)
        self.run(0.3)
        g2, L2 = self.jaw_gap(), self.cyl("fd_jaw")
        self.check("jaw open/close", g1 > g0 + 0.15 and g2 < g1 - 0.3,
                   f"tip gap {g0:.2f}->{g1:.2f}->{g2:.2f} m, cylinder {L0:.3f}->{L1:.3f}->{L2:.3f}")

    def scenario_clamp(self):
        z0 = self.s.pos("cfL1o")[2]
        self.hold("fd25ClampMinus", 1.5)  # raise
        self.run(0.3)
        z1 = self.s.pos("cfL1o")[2]
        self.check("clamp raise", z1 > z0 + 0.15, f"finger z {z0:.2f}->{z1:.2f} m")
        self.hold("fd25ClampPlus", 1.5)
        self.run(0.3)

    def scenario_relief(self):
        """Push the claw down into the ground with the boom: the relief logic must stall the boom."""
        stall_seen = False
        peak = 0.0
        hs = [x for x in self.s.hyd if x["src"] == "fd_boom"]

        def watch(sim):
            nonlocal stall_seen, peak
            st, ch, cfg = self.lua.state()
            stall_seen = stall_seen or bool(ch.boom.stalled)
            peak = max(peak, max(abs(sim.stress[h["cid"]]) for h in hs))
        self.lua.press("fd25BoomMinus", 1.0)
        self.s.run(6.0, self.lua, on_gfx=watch)
        self.lua.press("fd25BoomMinus", 0.0)
        self.run(0.3)
        claw = self.s.pos("jw5tL")[2]
        lifted = self.s.pos("uc_ref")[2] > 0.55   # boom pushed the machine's front up (real excavator behaviour)
        ok = (stall_seen or lifted) and peak < 1.7 * hs[0]["flim"] and np.isfinite(self.s.x).all()
        self.check("claw into ground -> relief or self-lift", ok,
                   f"claw z {claw:.2f} m, stall detected={stall_seen}, peak cylinder force {peak / 1e3:.0f} kN "
                   f"(relief {hs[0]['flim'] / 1e3:.0f} kN), machine ref z {self.s.pos('uc_ref')[2]:.2f} m")
        self.hold("fd25BoomPlus", 2.0)

    def scenario_slew(self):
        a0 = self.ch("slew").measured or 0.0
        self.lua.set_drive(steering=-1.0)  # work mode: steer left -> swing left
        self.run(3.0)
        self.lua.set_drive()
        self.run(1.0)
        a1 = self.ch("slew").measured
        msgs = [m for m in self.lua.messages() if "Swing" in m]
        self.check("swing left", a1 - a0 > math.radians(60),
                   f"slew {math.degrees(a0):.1f} -> {math.degrees(a1):.1f} deg; calibration msgs {msgs}")
        self.lua.set_drive(steering=1.0)
        self.run(3.0)
        self.lua.set_drive()
        self.run(1.0)
        a2 = self.ch("slew").measured
        self.check("swing right", a2 < a1 - math.radians(60), f"slew -> {math.degrees(a2):.1f} deg")

    def scenario_rot(self):
        r0 = self.ch("rot").measured or 0.0
        self.hold("fd25RotPlus", 2.0)
        self.run(0.5)
        r1 = self.ch("rot").measured
        self.check("rotator", abs(r1 - r0) > math.radians(45), f"rot {math.degrees(r0):.1f} -> {math.degrees(r1):.1f} deg")
        self.hold("fd25RotMinus", 2.0)

    def scenario_pose(self):
        self.lua.call("poseStow")
        self.run(9.0)
        st, ch, cfg = self.lua.state()
        self.check("pose stow macro", st.macro is None,
                   f"macro finished: {st.macro is None}; boom stroke {100 * (ch.boom.target - ch.boom.min) / (ch.boom.max - ch.boom.min):.0f}%")
        self.lua.call("poseWork")
        self.run(9.0)

    def att_pitch(self):
        a0, a1 = self.s.pos("rt0"), self.s.pos("rt1")
        axis = (a1 - a0) / np.linalg.norm(a1 - a0)
        n = self.s.pos("snL") - self.s.pos("snR")
        n /= np.linalg.norm(n)
        f = np.cross(n, [0, 0, 1.0])
        f /= np.linalg.norm(f)
        return math.atan2(axis @ f, -axis[2])

    def scenario_level(self):
        """Claw level assist: the attachment pitch must change far less while the stick moves."""
        res = {}
        for label, on in (("off", False), ("on", True)):
            if on:
                self.lua.call("toggleLevelAssist")
            self.run(0.3)
            p0 = self.att_pitch()
            self.hold("fd25StickMinus", 1.6)
            self.run(0.6)
            res[label] = abs(self.att_pitch() - p0)
            if on:
                self.lua.call("toggleLevelAssist")
            self.hold("fd25StickPlus", 1.6)
            self.run(0.6)
        self.check("claw level assist", res["on"] < 0.5 * res["off"],
                   f"attachment pitch change during 1.6 s stick-out: {math.degrees(res['off']):.1f} deg without, "
                   f"{math.degrees(res['on']):.1f} deg with assist")

    def scenario_travel(self):
        self.lua.call("toggleMode")
        p0 = self.s.pos("uc_ref")
        h0 = math.radians(self.yaw())
        fwd_dir = np.array([math.sin(h0), -math.cos(h0)])      # machine forward (-Y at yaw 0), see yaw()
        lat_dir = np.array([math.cos(h0), math.sin(h0)])       # machine left (+X at yaw 0)
        ph = []
        self.lua.set_drive(throttle=1.0)
        self.s.run(5.0, self.lua, on_gfx=lambda sim: ph.append(sim.electrics.get("fd_trackPhaseL", 0)))
        self.lua.set_drive()
        self.run(1.5)
        p1 = self.s.pos("uc_ref")
        dxy = (p1 - p0)[:2]
        fwd = float(dxy @ fwd_dir)
        lat = float(dxy @ lat_dir)
        moved = len(set(round(x, 3) for x in ph)) > 10
        self.check("travel forward (straight)", fwd > 1.5 and moved and abs(lat) < 0.15 * fwd,
                   f"moved {fwd:.2f} m along the initial heading (sideways {lat:+.2f} m), "
                   f"heading change {self.yaw() - math.degrees(h0):+.1f} deg, track phase animated={moved}")
        y0 = self.yaw()
        self.lua.set_drive(steering=1.0)
        self.run(4.0)
        self.lua.set_drive()
        self.run(1.0)
        y1 = self.yaw()
        self.check("pivot turn", abs(y1 - y0) > 12, f"yaw {y0:.1f} -> {y1:.1f} deg")
        p2 = self.s.pos("uc_ref")
        self.run(2.0)
        drift = np.linalg.norm(self.s.pos("uc_ref") - p2)
        self.check("travel brakes hold", drift < 0.05, f"drift {drift * 1000:.1f} mm in 2 s with no input")
        self.lua.call("toggleMode")

    def scenario_final(self):
        st = self.strain()
        self.check("structure integrity", st < 0.06 and np.isfinite(self.s.x).all(),
                   f"max strain {st * 100:.2f}% at end, sim time {self.s.t:.1f} s")

    def all(self, only=None):
        for f in (self.scenario_settle, self.scenario_boom, self.scenario_stick, self.scenario_tilt,
                  self.scenario_jaw, self.scenario_clamp, self.scenario_relief, self.scenario_slew, self.scenario_rot,
                  self.scenario_pose, self.scenario_level, self.scenario_travel, self.scenario_final):
            if only and f.__name__ != "scenario_settle" and f.__name__.replace("scenario_", "") not in only:
                continue
            try:
                f()
            except Exception as e:  # report and continue
                self.check(f.__name__, False, f"exception {e!r}")
        lines = [f"{'PASS' if ok else 'FAIL'}  {n}: {d}" for n, ok, d in self.results]
        lines.append(f"messages: {self.lua.messages()}")
        lines.append(f"wall time {time.time() - self.t0:.0f} s")
        if only:
            return all(ok for _, ok, _ in self.results)
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        with open(REPORT, "w") as f:
            f.write("\n".join(lines) + "\n")
        return all(ok for _, ok, _ in self.results)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    ok = Suite().all(only)
    sys.exit(0 if ok else 1)
