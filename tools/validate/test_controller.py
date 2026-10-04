"""Fast unit tests for fd25Hydraulics.lua against an ideal hydraulic plant (no soft-body physics).

The plant moves every hydro beam towards the length the controller commands (with the hydro's
own inRate/outRate, exactly like lua/vehicle/hydros.lua) and can block a cylinder to emulate a
jaw crushing something it cannot cut.  Node positions stay at spawn, so the angular channels run
open-loop here; their closed-loop behaviour is covered by tools.validate.cosim_tests.

Run:  python -m tools.validate.test_controller      (writes build/controller_test_report.txt)
"""
import math
import os
import sys

import numpy as np

from tools.validate.jbeam_load import assemble, ROOT
from tools.validate.lua_env import LuaVehicle

REPORT = os.path.join(ROOT, "build", "controller_test_report.txt")
DT = 1 / 60


def fnum(v, d):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class Plant:
    def __init__(self, config=None):
        V = assemble(config)
        self.V = V
        names = list(V["nodes"])
        self.idx = {n: i for i, n in enumerate(names)}
        self.x = np.array([V["nodes"][n]["pos"] for n in names])
        self.hyd = []
        for k, h in enumerate(V["hydros"]):
            p1, p2 = self.x[self.idx[h["id1"]]], self.x[self.idx[h["id2"]]]
            L = float(np.linalg.norm(p1 - p2))
            inl = fnum(h.get("inExtent"), 0) / L
            outl = fnum(h.get("outExtent"), 0) / L
            self.hyd.append(dict(cid=k, src=h["inputSource"], L=L, inl=inl, outl=outl, state=1.0,
                                 rin=fnum(h.get("inRate"), 2), rout=fnum(h.get("outRate"), 2),
                                 flim=fnum(h.get("forceLimit"), 1e30), K=fnum(h.get("beamSpring"), 4e6),
                                 blocked=False))
        self.wheels = [r["name"] for r in V["rotators"]]
        self.wheel_ids = {n: i for i, n in enumerate(self.wheels)}
        self.wspeed = np.zeros(len(self.wheels))
        self.electrics = {}
        self.vel = 0.0

    # interface for LuaVehicle --------------------------------------------------
    node_cids = property(lambda self: self.idx)

    @property
    def hydros(self):
        return [{"inputSource": h["src"], "beamCID": h["cid"], "forceLimit": h["flim"], "beamSpring": h["K"]}
                for h in self.hyd]

    @property
    def beams(self):
        return {h["cid"]: {"beamSpring": h["K"]} for h in self.hyd}

    def node_pos(self, cid):
        return tuple(self.x[cid])

    def beam_length(self, cid):
        h = self.hyd[cid]
        return h["L"] * h["state"]

    def beam_stress(self, cid):
        h = self.hyd[cid]
        return h["flim"] * 1.1 if h["blocked"] else 0.0

    def velocity(self):
        return (0.0, -self.vel, 0.0)

    def direction(self):
        return (0.0, -1.0, 0.0)

    def wheel_speed(self, wid):
        return float(self.wspeed[wid])

    # plant update ------------------------------------------------------------------
    def update(self, ev, dt):
        for h in self.hyd:
            u = ev(h["src"])
            u = max(-1.0, min(1.0, u))
            tgt = 1 + u * (h["outl"] - 1) if u >= 0 else 1 + u * (1 - h["inl"])
            if h["blocked"]:
                continue
            rate = h["rout"] if tgt > h["state"] else h["rin"]
            h["state"] += max(-rate * dt, min(rate * dt, tgt - h["state"]))
        # tracks: first-order response to the signed motor throttle, braked to zero
        for name, wid in self.wheel_ids.items():
            side = name[3]
            thr = ev(f"fd_track{side}")
            brk = ev(f"fd_brake{side}")
            target = 1.6 * thr * (1 - min(1.0, brk))
            self.wspeed[wid] += (target - self.wspeed[wid]) * min(1.0, dt / 0.3)
        self.vel = float(np.mean(self.wspeed))


class Tests:
    def __init__(self, config=None):
        self.p = Plant(config)
        self.lua = LuaVehicle(self.p)
        self.lua.init(self.p.V["sections"]["fd25Hydraulics"])
        self.results = []

    def step(self, sec, hook=None):
        for _ in range(int(round(sec / DT))):
            self.lua.step(DT)
            self.p.update(lambda k: float(self.lua.ev(k, 0.0) or 0.0), DT)
            if hook:
                hook()

    def hold(self, name, sec, value=1.0):
        self.lua.press(name, value)
        self.step(sec)
        self.lua.press(name, 0.0)

    def h(self, src):
        return [x for x in self.p.hyd if x["src"] == src][0]

    def length(self, src):
        h = self.h(src)
        return h["L"] * h["state"]

    def st(self):
        return self.lua.state()[0]

    def ch(self, name):
        return self.lua.state()[1][name]

    def check(self, name, cond, detail):
        self.results.append((name, bool(cond), detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)

    # ------------------------------------------------------------------ tests
    def t_init(self):
        st = self.st()
        chans = [n for n in ("boom", "stick", "tilt", "jaw", "clamp", "slew", "rot") if self.lua.state()[1][n]]
        self.check("init", st.mode == "work" and len(chans) == 7, f"mode={st.mode}, channels={chans}")
        self.step(0.5)
        outs = {k: self.lua.ev(f"fd_{k}") for k in ("boom", "stick", "tilt", "jaw", "clamp")}
        drift = max(abs(self.length(f"fd_{k}") - self.h(f"fd_{k}")["L"]) for k in outs)
        self.check("hold at spawn", drift < 0.002, f"max cylinder drift {drift * 1000:.2f} mm with no input")

    def t_linear_speed(self):
        for name, ev in (("jaw", "fd25JawPlus"), ("stick", "fd25StickPlus"), ("tilt", "fd25TiltPlus")):
            L0 = self.length(f"fd_{name}")
            self.hold(ev, 1.0)
            self.step(0.3)
            dL = self.length(f"fd_{name}") - L0
            c = self.ch(name)
            self.check(f"{name} moves", abs(dL) > 0.04, f"stroke change {dL * 1000:+.0f} mm in 1 s (max speed {c.maxSpeed:.2f} m/s)")
            self.hold(ev.replace("Plus", "Minus"), 1.0)
            self.step(0.3)

    def t_precision(self):
        L0 = self.length("fd_stick")
        self.hold("fd25StickPlus", 0.8)
        self.step(0.4)
        d_norm = abs(self.length("fd_stick") - L0)
        self.hold("fd25StickMinus", 0.8)
        self.step(0.4)
        self.lua.call("togglePrecision")
        L0 = self.length("fd_stick")
        self.hold("fd25StickPlus", 0.8)
        self.step(0.4)
        d_prec = abs(self.length("fd_stick") - L0)
        self.lua.call("togglePrecision")
        self.hold("fd25StickMinus", 0.8)
        self.step(0.4)
        ratio = d_prec / max(d_norm, 1e-9)
        self.check("precision mode", 0.2 < ratio < 0.5, f"precision/normal stroke ratio {ratio:.2f} (target 0.35)")

    def t_flow_sharing(self):
        # one function alone vs. all at once: the pump cannot feed everything at full speed
        self.step(5.0)  # let auto-idle settle so both runs start alike
        L0 = self.length("fd_stick")
        self.hold("fd25StickPlus", 0.6)
        alone = abs(self.length("fd_stick") - L0)
        self.hold("fd25StickMinus", 0.6)
        self.step(5.0)
        L0 = self.length("fd_stick")
        for e in ("fd25StickPlus", "fd25JawPlus", "fd25TiltPlus", "fd25ClampPlus", "fd25BoomPlus"):
            self.lua.press(e, 1.0)
        self.step(0.6)
        use = self.st().flowUse
        for e in ("fd25StickPlus", "fd25JawPlus", "fd25TiltPlus", "fd25ClampPlus", "fd25BoomPlus"):
            self.lua.press(e, 0.0)
        shared = abs(self.length("fd_stick") - L0)
        self.check("pump flow sharing", shared < alone * 0.95 and use > 1.0,
                   f"stick stroke alone {alone * 1000:.0f} mm vs. with 5 functions {shared * 1000:.0f} mm, demand {use * 100:.0f}% of pump")
        for e in ("fd25StickMinus", "fd25JawMinus", "fd25TiltMinus", "fd25ClampMinus", "fd25BoomMinus"):
            self.lua.press(e, 1.0)
        self.step(0.6)
        for e in ("fd25StickMinus", "fd25JawMinus", "fd25TiltMinus", "fd25ClampMinus", "fd25BoomMinus"):
            self.lua.press(e, 0.0)
        self.step(0.5)

    def t_relief(self):
        h = self.h("fd_jaw")
        L_block = h["L"] * h["state"]
        h["blocked"] = True
        self.lua.press("fd25JawPlus", 1.0)
        self.step(2.0)
        c = self.ch("jaw")
        stalled = bool(c.stalled)
        wind = abs(c.target - L_block)
        snd_relief = [s for s in self.lua.sounds.values() if "relief" in s["src"]]
        rel_vol = snd_relief[0]["vol"] if snd_relief else 0
        self.check("relief stall detection", stalled and wind <= c.windup * 1.05 + 1e-6,
                   f"stalled={stalled}, command {wind * 1000:.1f} mm past the blocked cylinder (windup {c.windup * 1000:.1f} mm)")
        self.check("relief sound", rel_vol > 0.3, f"relief loop volume {rel_vol:.2f}")
        self.lua.press("fd25JawPlus", 0.0)
        h["blocked"] = False
        # reversing must respond immediately (no integrator wind-up)
        out0 = self.lua.ev("fd_jaw")
        self.lua.press("fd25JawMinus", 1.0)
        self.step(0.25)
        out1 = self.lua.ev("fd_jaw")
        self.lua.press("fd25JawMinus", 0.0)
        self.check("instant reversal after stall", (out1 - out0) * math.copysign(1, -c.maxSpeed) > 0 or abs(out1 - out0) > 0.01,
                   f"jaw command moved {out1 - out0:+.3f} within 0.25 s of reversing")
        self.step(1.0)

    def t_work_mode_drive_inputs(self):
        st = self.st()
        slew0 = self.ch("slew").target
        boom0 = self.length("fd_boom")
        self.lua.set_drive(steering=-1.0)        # steer left -> swing left (+)
        self.step(1.0)
        self.lua.set_drive()
        self.step(0.5)
        slew1 = self.ch("slew").target
        self.lua.set_drive(throttle=1.0)
        self.step(1.0)
        self.lua.set_drive()
        self.step(0.3)
        boom1 = self.length("fd_boom")
        braked = self.lua.ev("fd_brakeL") > 0.5 and self.lua.ev("fd_brakeR") > 0.5
        self.check("work mode: steering swings", slew1 - slew0 > 0.2, f"swing target {math.degrees(slew1 - slew0):+.1f} deg after 1 s left steer")
        self.check("work mode: throttle raises boom", boom1 - boom0 > 0.05, f"boom cylinder {(boom1 - boom0) * 1000:+.0f} mm after 1 s throttle")
        self.check("work mode: tracks braked", braked, f"brakes L/R {self.lua.ev('fd_brakeL'):.2f}/{self.lua.ev('fd_brakeR'):.2f}")
        self.lua.set_drive(steering=1.0)
        self.step(1.0)
        self.lua.set_drive(brake=1.0)
        self.step(1.0)
        self.lua.set_drive()
        self.step(0.5)

    def t_travel(self):
        self.lua.call("toggleMode")
        self.check("toggle travel mode", self.st().mode == "travel", f"mode={self.st().mode}")
        self.lua.set_drive(throttle=1.0)
        self.step(2.0)
        tl, tr = self.lua.ev("fd_trackL"), self.lua.ev("fd_trackR")
        alarm = [s for s in self.lua.sounds.values() if s["playing"] and "Beep" in s["src"]]
        self.check("travel forward", tl > 0.1 and tr > 0.1 and self.p.vel > 0.3,
                   f"track commands L {tl:+.2f} R {tr:+.2f}, speed {self.p.vel:.2f} m/s")
        self.check("travel alarm", len(alarm) == 1, f"{len(alarm)} alarm source playing")
        self.lua.set_drive(steering=1.0)        # pivot right: left track forward, right track back
        self.step(1.5)
        tl, tr = self.lua.ev("fd_trackL"), self.lua.ev("fd_trackR")
        self.check("pivot turn", tl > 0.05 and tr < -0.05, f"pivot right: L {tl:+.2f} R {tr:+.2f}")
        self.lua.set_drive()
        self.step(2.0)
        hold = self.lua.ev("fd_brakeL") > 0.5 and self.lua.ev("fd_brakeR") > 0.5
        self.check("parking brakes when idle", hold and abs(self.p.vel) < 0.05,
                   f"brakes {self.lua.ev('fd_brakeL'):.2f}/{self.lua.ev('fd_brakeR'):.2f}, speed {self.p.vel:.3f}")
        self.lua.call("toggleTravelSpeed")
        self.check("travel speed toggle", bool(self.st().travelHigh), f"travelHigh={self.st().travelHigh}")
        self.lua.call("toggleTravelSpeed")
        self.lua.call("toggleMode")

    def t_power_and_idle(self):
        self.step(6.0)
        idle = self.st().rpm
        self.lua.press("fd25StickPlus", 1.0)
        self.step(1.0)
        work = self.st().rpm
        self.lua.press("fd25StickPlus", 0.0)
        self.lua.call("cyclePower")
        p = self.st().power
        self.lua.call("cyclePower")
        self.lua.call("cyclePower")
        self.check("auto idle / rev up", idle < work - 300, f"idle {idle:.0f} rpm -> working {work:.0f} rpm")
        self.check("power mode cycle", p == "power" and self.st().power == "standard", f"after one press {p}, after three {self.st().power}")
        self.hold("fd25StickMinus", 1.0)

    def t_macro(self):
        self.lua.call("poseStow")
        self.step(0.3)
        active = self.st().macro == "stow"
        self.lua.press("fd25JawPlus", 1.0)
        self.step(0.2)
        self.lua.press("fd25JawPlus", 0.0)
        cancelled = self.st().macro is None
        self.check("pose macro starts / cancels on input", active and cancelled, f"active={active}, cancelled={cancelled}")
        self.lua.call("poseWork")
        self.step(12.0)
        done = self.st().macro is None
        self.check("pose macro completes", done, f"macro state after 12 s: {self.st().macro}")

    def t_level_assist(self):
        """Closed loop on a kinematic claw: pitch = disturbance + tilt stroke / tiltLengthPerRad.
        A step disturbance (as if the stick had swung the claw) must be cancelled by the tilt cylinder."""
        cfg = self.lua.state()[2]
        i0, i1 = self.p.idx[cfg.attPitchNodes[1]], self.p.idx[cfg.attPitchNodes[2]]
        d0 = self.p.x[i1] - self.p.x[i0]
        L0 = self.length("fd_tilt")
        k = float(cfg.tiltLengthPerRad)
        dist = {"v": 0.0}

        def pitch_offset():
            return dist["v"] + (self.length("fd_tilt") - L0) / k

        def hook():
            a = -pitch_offset()                  # rotation about +X by a changes the measured pitch by -a
            R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            self.p.x[i1] = self.p.x[i0] + R @ d0

        self.lua.call("toggleLevelAssist")
        self.step(0.3, hook)                     # captures the pitch to hold
        base = pitch_offset()
        dist["v"] += 0.15                        # claw knocked 8.6 deg off level
        self.step(2.0, hook)
        err1 = pitch_offset() - base
        dist["v"] -= 0.15
        self.step(2.0, hook)
        err2 = pitch_offset() - base
        self.lua.call("toggleLevelAssist")
        self.p.x[i1] = self.p.x[i0] + d0
        self.check("claw level assist (closed loop)", abs(err1) < 0.02 and abs(err2) < 0.02,
                   f"residual pitch error {math.degrees(err1):+.2f} deg after an 8.6 deg disturbance, "
                   f"{math.degrees(err2):+.2f} deg after it was removed")
        self.step(0.5)

    def t_beacon_hud(self):
        self.lua.call("toggleBeacon")
        vals = []
        for _ in range(60):
            self.step(DT)
            vals.append(self.lua.ev("fd_beacon"))
        self.lua.call("toggleBeacon")
        self.check("beacon pulses", max(vals) > 0.5 and min(vals) == 0, f"fd_beacon range {min(vals):.2f}..{max(vals):.2f}")
        hud = self.lua.L.globals().hud["FD25HudUpdate"]
        keys = ["mode", "rpm", "pressureMPa", "flow", "cyl"] if hud else []
        ok = hud is not None and all(hud[k] is not None for k in keys) and hud["cyl"]["jaw"] is not None
        self.check("HUD data", ok, "FD25HudUpdate carries mode/rpm/pressure/flow/cylinders" if ok else "missing HUD data")

    def t_sounds(self):
        srcs = sorted(s["src"] for s in self.lua.sounds.values())
        need = ["fd25_diesel_loop.ogg", "fd25_relief_loop.ogg", "fd25_track_loop.ogg", "Pump_Big"]
        ok = all(any(n in s for s in srcs) for n in need)
        self.check("sound sources", ok, ", ".join(os.path.basename(s) for s in srcs))

    def t_reset(self):
        self.lua.ctrl.reset(self.lua.params)
        self.step(0.5)
        st = self.st()
        self.check("reset", st.macro is None and abs(self.lua.ev("fd_jaw")) < 0.05,
                   f"after reset: macro={st.macro}, jaw cmd {self.lua.ev('fd_jaw'):+.3f}")

    def run(self):
        for name in ("t_init", "t_sounds", "t_linear_speed", "t_precision", "t_flow_sharing", "t_relief",
                     "t_work_mode_drive_inputs", "t_travel", "t_power_and_idle", "t_macro", "t_level_assist",
                     "t_beacon_hud", "t_reset"):
            try:
                getattr(self, name)()
            except Exception as e:  # noqa: BLE001
                self.check(name, False, f"exception: {e!r}")
        return self.results


def main():
    results = Tests().run()
    for cfg in ({"fd25_hydraulics": "fd25_hydraulics_arcade"}, {"fd25_clamparm": "fd25_clamparm_none"},
                {"fd25_cab": "fd25_cab_elevated"}):
        try:
            t = Tests(cfg)
            t.step(1.0)
            t.hold("fd25JawPlus", 0.5)
            ok = not any(math.isnan(float(t.lua.ev(f"fd_{k}") or 0)) for k in ("boom", "stick", "jaw"))
            results.append((f"config {list(cfg.values())[0]}", ok, "controller initialises and runs"))
            print(f"[{'PASS' if ok else 'FAIL'}] config {list(cfg.values())[0]}")
        except Exception as e:  # noqa: BLE001
            results.append((f"config {cfg}", False, repr(e)))
            print(f"[FAIL] config {cfg}: {e!r}")
    npass = sum(1 for r in results if r[1])
    lines = ["CONTROLLER UNIT TESTS (ideal plant)", "=" * 60]
    lines += [f"[{'PASS' if ok else 'FAIL'}] {n}: {d}" for n, ok, d in results]
    lines.append(f"{npass}/{len(results)} passed")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(lines[-1])
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
