"""Kinematics for the FD-25: forward kinematics, cylinder lengths, frames.

All public functions work in BeamNG vehicle coordinates (+X left, -Y forward,
+Z up).  A *pose* is a dict with keys
    boom, stick_rel, att_rel, jaw, slew, rot, clamp
(angles in radians, see tools/spec/machine.py for conventions).

Body frames are returned as 4x4 numpy matrices mapping body-local points to
world.  Local coordinates are (s, t, x) for planar bodies (x = world lateral
offset, +X left) and (a, b, x) for the attachment bodies.  Because the planar
basis [s, t, x] is left-handed, every frame matrix has determinant -1; the
modeller flips face winding once when baking meshes into world space.  The
*relative* motion between two poses is a proper rotation.
"""
import math
import numpy as np

from tools.spec import machine as M


def _dir(theta):
    return np.array([math.cos(theta), math.sin(theta)])


def _perp(theta):
    return np.array([-math.sin(theta), math.cos(theta)])


class PlaneFrame:
    """In-plane rigid frame: origin (u, w) and direction angle theta."""

    def __init__(self, origin, theta, xa=0.0):
        self.o = np.asarray(origin, float)
        self.theta = theta
        self.xa = xa

    def p(self, s, t=0.0):
        return self.o + s * _dir(self.theta) + t * _perp(self.theta)

    def local(self, uw):
        d = np.asarray(uw, float) - self.o
        return float(d @ _dir(self.theta)), float(d @ _perp(self.theta))

    def matrix(self, slew=0.0):
        """4x4 matrix local(s, t, x) -> world (pre-slew plane mapped then slewed)."""
        su, sw = _dir(self.theta)
        tu, tw = _perp(self.theta)
        # plane (u, w, x) -> world (X, Y, Z) = (x, -u, w)
        S = np.array([0.0, -su, sw])
        T = np.array([0.0, -tu, tw])
        X = np.array([1.0, 0.0, 0.0])
        O = np.array([self.xa, -self.o[0], self.o[1]])
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = S, T, X, O
        return slew_matrix(slew) @ m


def slew_matrix(psi):
    c, s = math.cos(psi), math.sin(psi)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def plane_to_world(uw, x=0.0, slew=0.0):
    p = np.array([x, -uw[0], uw[1], 1.0])
    return (slew_matrix(slew) @ p)[:3]


def circle_intersections(c1, r1, c2, r2):
    d = np.linalg.norm(c2 - c1)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    m = c1 + a * (c2 - c1) / d
    off = h * np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / d
    return m + off, m - off


def default_pose():
    P = M.POSE
    return dict(boom=P.boom, stick_rel=P.stick_rel, att_rel=P.att_rel, jaw=P.jaw,
                slew=P.slew, rot=P.rot, clamp=P.clamp)


def solve(pose=None):
    """Return a dict with frames (4x4), key points (world xyz) and cylinder lengths."""
    pose = dict(default_pose(), **(pose or {}))
    A = M.ARM
    xa = A.xa
    slew = pose["slew"]
    out = {"pose": pose, "frames": {}, "pts": {}, "plane": {}, "cyl": {}, "ok": True}

    boom = PlaneFrame(A.boom_foot, pose["boom"], xa)
    B1 = boom.p(A.boom_len)
    th2 = pose["boom"] + pose["stick_rel"]
    stick = PlaneFrame(B1, th2, xa)
    S1 = stick.p(A.stick_len)
    th3 = th2 + pose["att_rel"]
    half = M.ATT.pin_spacing / 2.0
    # attachment frame: origin at bracket centre; S1 is at (a=0, b=-half)
    att_o = S1 + half * _perp(th3)
    att = PlaneFrame(att_o, th3, xa)
    C = att.p(0.0, +half)
    K = stick.p(*A.linkpivot_on_stick)
    sol = circle_intersections(K, A.sidelink_len, C, A.hlink_len)
    if sol is None:
        out["ok"] = False
        P = (K + C) / 2
    else:
        # choose the solution on the stick's "top" side (larger t)
        P = max(sol, key=lambda q: stick.local(q)[1])
    side = PlaneFrame(K, math.atan2(*(P - K)[::-1]), xa)
    hl = PlaneFrame(P, math.atan2(*(C - P)[::-1]), xa)

    pl = out["plane"]
    pl.update(B0=np.array(A.boom_foot), B1=B1, S1=S1, C=C, K=K, P=P, att_o=att_o)

    # planar frames -> 4x4
    fr = out["frames"]
    fr["boom"] = boom.matrix(slew)
    fr["stick"] = stick.matrix(slew)
    fr["sidelink"] = side.matrix(slew)
    fr["hlink"] = hl.matrix(slew)
    fr["att"] = att.matrix(slew)
    # superstructure frame: world-aligned, slewed
    fr["ss"] = slew_matrix(slew)
    fr["uc"] = np.eye(4)
    # rotator: rotate head about the attachment a-axis (local x axis of att frame = s)
    rot = pose["rot"]
    R = np.eye(4)
    c, s = math.cos(rot), math.sin(rot)
    # local axes of att frame: col0 = a (s), col1 = b (t), col2 = x
    R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    fr["head"] = fr["att"] @ R
    # jaw: rotate about pivot axis (parallel to local x) by -jaw in (a, b) plane
    J = np.eye(4)
    g = -pose["jaw"]
    cg, sg = math.cos(g), math.sin(g)
    pa, pb = M.ATT.jaw_pivot
    J[0, 0], J[0, 1], J[1, 0], J[1, 1] = cg, -sg, sg, cg
    J[0, 3] = pa - (cg * pa - sg * pb)
    J[1, 3] = pb - (sg * pa + cg * pb)
    fr["jaw"] = fr["head"] @ J

    # clamp arm (undercarriage, not slewed, centred on x = 0)
    Cl = M.CLAMP
    clamp = PlaneFrame(Cl.pivot, pose["clamp"], 0.0)
    fr["clamp"] = clamp.matrix(0.0)

    def W(frame, s, t=0.0, x=0.0):
        return (out["frames"][frame] @ np.array([s, t, x, 1.0]))[:3]

    out["W"] = W
    pts = out["pts"]
    # cylinders: (base, rod) world points
    cyl = {}
    bc_base = A.boomcyl_base
    for side_name, sx in (("l", +1), ("r", -1)):
        base = plane_to_world(bc_base, xa + sx * A.boomcyl_x, slew)
        rod = W("boom", *A.boomcyl_rod_on_boom, x=sx * A.boomcyl_x)
        cyl["boom_" + side_name] = (base, rod)
    cyl["stick"] = (W("boom", *A.stickcyl_base_on_boom), W("stick", *A.stick_tail))
    cyl["tilt"] = (W("stick", *A.tiltcyl_base_on_stick), W("sidelink", A.sidelink_len - 0.05, 0.0))
    jb = W("head", *M.ATT.jawcyl_base)
    jl = W("jaw", *M.ATT.jaw_lever)
    cyl["jaw"] = (jb, jl)
    for side_name, sx in (("l", +1), ("r", -1)):
        base = plane_to_world(Cl.cyl_base, sx * Cl.cyl_x, 0.0)
        rod = W("clamp", *Cl.cyl_rod_on_arm, x=sx * Cl.cyl_x)
        cyl["clamp_" + side_name] = (base, rod)
    out["cyl"] = {k: (b, r, float(np.linalg.norm(r - b))) for k, (b, r) in cyl.items()}

    # key world points
    pts["boom_foot"] = W("boom", 0.0)
    pts["boom_nose"] = W("boom", A.boom_len)
    pts["stick_nose"] = W("stick", A.stick_len)
    pts["claw_tip_fixed"] = W("head", *M.ATT.fork_line[-1])
    pts["claw_tip_jaw"] = W("jaw", *M.ATT.jaw_line[-1])
    pts["clamp_tip"] = W("clamp", Cl.arm_len + 0.25, -Cl.finger_len)
    return out


def cylinder_ranges(samples=13):
    """Sweep each joint over its range (others at spawn) and report cylinder lengths."""
    R = M.RANGE
    rep = {}

    def sweep(key, cyls):
        lo, hi = getattr(R, key)
        lens = {c: [] for c in cyls}
        ok = True
        for i in range(samples):
            v = lo + (hi - lo) * i / (samples - 1)
            s = solve({key: v})
            ok = ok and s["ok"]
            for c in cyls:
                lens[c].append(s["cyl"][c][2])
        for c in cyls:
            rep[c] = dict(min=min(lens[c]), max=max(lens[c]), spawn=solve()["cyl"][c][2],
                          monotonic=_monotonic(lens[c]), joint=key, linkage_ok=ok)

    sweep("boom", ["boom_l"])
    sweep("stick_rel", ["stick"])
    sweep("att_rel", ["tilt"])
    sweep("jaw", ["jaw"])
    sweep("clamp", ["clamp_l"])
    return rep


def _monotonic(seq):
    d = np.diff(seq)
    return bool(np.all(d > 0) or np.all(d < 0))


def report():
    s = solve()
    lines = ["== Spawn pose key points (world X, Y, Z) =="]
    for k, v in s["pts"].items():
        lines.append(f"  {k:16s} {v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f}")
    lines.append("== Cylinders: spawn / min / max / ratio / stroke ==")
    for k, r in cylinder_ranges().items():
        ratio = r["max"] / r["min"]
        lines.append(f"  {k:8s} spawn {r['spawn']:.3f}  min {r['min']:.3f}  max {r['max']:.3f}  "
                     f"ratio {ratio:.2f}  stroke {r['max']-r['min']:.3f}  monotonic={r['monotonic']} linkage_ok={r['linkage_ok']}")
    # reach envelope
    reach, depth, height = 0, 0, 0
    for b in np.linspace(*M.RANGE.boom, 9):
        for st in np.linspace(*M.RANGE.stick_rel, 9):
            p = solve({"boom": b, "stick_rel": st})["pts"]["stick_nose"]
            reach = max(reach, -p[1])
            depth = min(depth, p[2])
            height = max(height, p[2])
    lines.append(f"== Envelope (stick nose pin): max reach {reach:.2f} m, max height {height:.2f} m, min z {depth:.2f} m")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
