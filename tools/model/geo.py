"""Procedural hard-surface geometry helpers for bpy (Blender 4.5 as a Python module).

All builders return (verts, faces, mats) "mesh parts" in a simple container so
that parts can be generated with numpy, transformed by 4x4 body frames, merged
and finally turned into one Blender object per exported mesh.

    Part.v : (N,3) float array
    Part.f : list of tuples (vertex indices), counter-clockwise seen from outside
    Part.m : list of material slot names (one per face)
"""
import math

import numpy as np


class Part:
    """Mesh fragment.  uv: optional per-face list of (u, v) per corner (None = auto-unwrap later)."""
    __slots__ = ("v", "f", "m", "uv")

    def __init__(self, v=None, f=None, m=None, uv=None):
        self.v = np.zeros((0, 3)) if v is None else np.asarray(v, float)
        self.f = [] if f is None else list(f)
        self.m = [] if m is None else list(m)
        self.uv = [None] * len(self.f) if uv is None else list(uv)
        if len(self.uv) < len(self.f):
            self.uv += [None] * (len(self.f) - len(self.uv))

    def add(self, other):
        off = len(self.v)
        self.v = np.vstack([self.v, other.v]) if len(self.v) else other.v.copy()
        self.f += [tuple(i + off for i in face) for face in other.f]
        self.m += other.m
        ouv = list(other.uv) + [None] * (len(other.f) - len(other.uv))
        self.uv = list(self.uv) + [None] * (len(self.f) - len(other.f) - len(self.uv)) + ouv[:len(other.f)]
        return self

    def copy(self):
        return Part(self.v.copy(), list(self.f), list(self.m), list(self.uv))

    def _faces_uv(self, flip):
        if not flip:
            return list(self.f), list(self.uv)
        f = [tuple(reversed(face)) for face in self.f]
        uv = [None if u is None else list(reversed(u)) for u in self.uv]
        return f, uv

    def transform(self, M):
        M = np.asarray(M, float)
        v = np.c_[self.v, np.ones(len(self.v))] @ M.T
        f, uv = self._faces_uv(np.linalg.det(M[:3, :3]) < 0)
        return Part(v[:, :3], f, self.m, uv)

    def translate(self, t):
        return Part(self.v + np.asarray(t, float), self.f, self.m, self.uv)

    def scale(self, s):
        s = np.broadcast_to(np.asarray(s, float), (3,))
        f, uv = self._faces_uv(np.prod(s) < 0)
        return Part(self.v * s, f, self.m, uv)

    def mirror_x(self):
        v = self.v.copy()
        v[:, 0] *= -1
        f, uv = self._faces_uv(True)
        return Part(v, f, self.m, uv)

    def set_mat(self, mat):
        self.m = [mat] * len(self.f)
        return self

    def box_uv(self, scale=1.0, offset=(0.0, 0.0)):
        """Planar box projection UVs (per face, by dominant normal axis), in local coordinates."""
        uv = []
        for face in self.f:
            pts = self.v[list(face)]
            n = np.abs(newell(pts))
            ax = int(np.argmax(n))
            a, b = [(1, 2), (0, 2), (0, 1)][ax]
            uv.append([(p[a] * scale + offset[0], p[b] * scale + offset[1]) for p in pts])
        self.uv = uv
        return self

    def rect_uv(self, rect):
        """Map the part's bounding box (largest two extents) into a UV rectangle (u0, v0, u1, v1)."""
        lo, hi = self.v.min(0), self.v.max(0)
        ext = hi - lo
        a, b = np.argsort(-ext)[:2]
        a, b = sorted((a, b))
        u0, v0, u1, v1 = rect
        uv = []
        for face in self.f:
            pts = self.v[list(face)]
            uv.append([(u0 + (p[a] - lo[a]) / max(ext[a], 1e-9) * (u1 - u0),
                        v0 + (p[b] - lo[b]) / max(ext[b], 1e-9) * (v1 - v0)) for p in pts])
        self.uv = uv
        return self


def merge(parts):
    out = Part()
    for p in parts:
        if p is not None and len(p.f):
            out.add(p)
    return out


def mat4(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame_from(origin, xaxis, yhint):
    """Right-handed 4x4 frame with X along xaxis, Y close to yhint."""
    x = np.asarray(xaxis, float)
    x = x / np.linalg.norm(x)
    y = np.asarray(yhint, float)
    y = y - x * (y @ x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin
    return M


def frame_z(origin, zaxis, xhint=(1, 0, 0)):
    """Right-handed frame with Z along zaxis."""
    z = np.asarray(zaxis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(xhint, float)
    if abs(x @ z) > 0.95:
        x = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin
    return M


# ----------------------------------------------------------------------------- primitives
def box(size, center=(0, 0, 0), mat="mat", chamfer=0.0):
    """Axis aligned box; optional chamfer on all 12 edges (as real geometry)."""
    sx, sy, sz = (np.asarray(size, float) / 2)
    c = np.asarray(center, float)
    if chamfer <= 0:
        v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + c
        # index: x*4 + y*2 + z
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        return Part(v, f, [mat] * 6)
    return rounded_box(size, center, chamfer, segments=1, mat=mat)


def rounded_box(size, center=(0, 0, 0), radius=0.02, segments=2, mat="mat"):
    """Box with rounded edges (superellipse-free: quarter circles in each section)."""
    sx, sy, sz = np.asarray(size, float) / 2
    r = min(radius, sx * 0.49, sy * 0.49, sz * 0.49)
    # build as a loft of rounded rectangles along Y
    prof = rounded_rect(2 * sx, 2 * sz, r, segments)
    ys = []
    # rounded ends along y as well
    for k in range(segments + 1):
        a = (math.pi / 2) * k / segments
        ys.append((-sy + r - r * math.cos(a), r * math.sin(a)))
    sections = []
    for (y, inset) in ys:
        sections.append((y, r - inset))
    for (y, inset) in reversed(ys):
        sections.append((-y, r - inset))
    rings = []
    for y, ins in sections:
        p = rounded_rect(2 * sx - 2 * ins, 2 * sz - 2 * ins, max(r - ins, 1e-4), segments)
        rings.append(np.c_[p[:, 0], np.full(len(p), y), p[:, 1]])
    part = loft_rings(rings, mat=mat, cap=True)
    return part.translate(center)


def rounded_rect(w, h, r, segments=2):
    """Rounded rectangle polygon (CCW) centred at origin, in (u, v)."""
    r = max(min(r, w / 2 - 1e-5, h / 2 - 1e-5), 1e-5)
    pts = []
    corners = [(w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180),
               (w / 2 - r, -h / 2 + r, 270)]
    for cx, cy, a0 in corners:
        for k in range(segments + 1):
            a = math.radians(a0 + 90 * k / segments)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def loft_rings(rings, mat="mat", cap=True, cap_mat=None):
    """Bridge consecutive rings of equal vertex count; rings ordered so that faces face outward."""
    rings = [np.asarray(r, float) for r in rings]
    n = len(rings[0])
    v = np.vstack(rings)
    f = []
    for i in range(len(rings) - 1):
        a, b = i * n, (i + 1) * n
        for k in range(n):
            k2 = (k + 1) % n
            f.append((a + k, a + k2, b + k2, b + k))
    mats = [mat] * len(f)
    part = Part(v, f, mats)
    part = orient_outward(part)
    if cap:
        part = cap_ends(part, rings, cap_mat or mat)
    return part


def cap_ends(part, rings, mat):
    n = len(rings[0])
    first = list(range(0, n))
    last = list(range((len(rings) - 1) * n, len(rings) * n))
    c0 = rings[0].mean(0)
    c1 = rings[-1].mean(0)
    dirv = c1 - c0
    out = part.copy()
    for ring_idx, sign in ((first, -1), (last, 1)):
        face = tuple(ring_idx)
        pts = out.v[list(face)]
        nrm = newell(pts)
        if np.dot(nrm, dirv) * sign < 0:
            face = tuple(reversed(face))
        out.f.append(face)
        out.m.append(mat)
    return out


def newell(pts):
    n = np.zeros(3)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    return n


def orient_outward(part):
    """Make all faces point away from the part's centroid (good for convex-ish shells)."""
    c = part.v.mean(0)
    f = []
    for face in part.f:
        pts = part.v[list(face)]
        nrm = newell(pts)
        if np.dot(nrm, pts.mean(0) - c) < 0:
            face = tuple(reversed(face))
        f.append(face)
    return Part(part.v, f, part.m)


def cylinder(radius, length, segments=24, axis="z", center=(0, 0, 0), mat="mat", cap=True,
             radius2=None, bevel=0.0, cap_mat=None):
    """Cylinder along axis, centred. bevel adds a chamfer ring at both ends."""
    r2 = radius if radius2 is None else radius2
    h = length / 2
    zs = [(-h, radius)]
    if bevel > 0:
        zs = [(-h, radius - bevel), (-h + bevel, radius), (h - bevel, r2), (h, r2 - bevel)]
    else:
        zs = [(-h, radius), (h, r2)]
    rings = []
    for z, r in zs:
        ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        rings.append(np.c_[r * np.cos(ang), r * np.sin(ang), np.full(segments, z)])
    p = loft_rings(rings, mat=mat, cap=cap, cap_mat=cap_mat)
    if axis == "x":
        p = p.transform(mat4(rot_y(math.pi / 2)))
    elif axis == "y":
        p = p.transform(mat4(rot_x(-math.pi / 2)))
    return p.translate(center)


def tube(radius_out, radius_in, length, segments=24, axis="z", center=(0, 0, 0), mat="mat"):
    """Hollow cylinder (both shells + end annuli)."""
    h = length / 2
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    co, si = np.cos(ang), np.sin(ang)
    rings = [np.c_[radius_out * co, radius_out * si, np.full(segments, z)] for z in (-h, h)]
    rings_in = [np.c_[radius_in * co, radius_in * si, np.full(segments, z)] for z in (h, -h)]
    v = np.vstack(rings + rings_in)
    n = segments
    f = []
    for a, b in ((0, n), (n, 2 * n), (2 * n, 3 * n), (3 * n, 0)):
        for k in range(n):
            k2 = (k + 1) % n
            f.append((a + k, a + k2, b + k2, b + k))
    p = Part(v, f, [mat] * len(f))
    # orient: outer outward, inner inward -> use radial test
    ff = []
    for face in p.f:
        pts = p.v[list(face)]
        nrm = newell(pts)
        cen = pts.mean(0)
        radial = np.array([cen[0], cen[1], 0.0])
        rad = np.linalg.norm(radial[:2])
        if abs(nrm[2]) > abs(nrm[0]) + abs(nrm[1]):
            want = np.sign(cen[2])
            ok = nrm[2] * want > 0
        else:
            outer = rad > (radius_out + radius_in) / 2
            ok = (nrm @ radial > 0) == outer
        ff.append(face if ok else tuple(reversed(face)))
    p.f = ff
    if axis == "x":
        p = p.transform(mat4(rot_y(math.pi / 2)))
    elif axis == "y":
        p = p.transform(mat4(rot_x(-math.pi / 2)))
    return p.translate(center)


def prism(poly2d, thickness, mat="mat", side_mat=None, center_z=0.0):
    """Extrude a 2D polygon (CCW, in XY) along Z by thickness (centred on center_z)."""
    poly = np.asarray(poly2d, float)
    n = len(poly)
    z0, z1 = center_z - thickness / 2, center_z + thickness / 2
    v = np.vstack([np.c_[poly, np.full(n, z0)], np.c_[poly, np.full(n, z1)]])
    f, m = [], []
    # ensure CCW
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    order = list(range(n)) if area > 0 else list(reversed(range(n)))
    f.append(tuple(n + i for i in order))           # top (+Z)
    m.append(mat)
    f.append(tuple(reversed(order)))                 # bottom
    m.append(mat)
    for k in range(n):
        a, b = order[k], order[(k + 1) % n]
        f.append((a, b, n + b, n + a))
        m.append(side_mat or mat)
    return Part(v, f, m)


def plate_on_frame(poly2d, thickness, M, mat="mat", side_mat=None):
    """Extruded plate whose polygon lives in the XY plane of frame M (Z = plate normal)."""
    return prism(poly2d, thickness, mat=mat, side_mat=side_mat).transform(M)


def offset_polyline(points, half_widths):
    """Thick 2D strip around a polyline -> closed polygon (CCW)."""
    P = np.asarray(points, float)
    hw = np.broadcast_to(np.asarray(half_widths, float), (len(P),))
    left, right = [], []
    for i in range(len(P)):
        if i == 0:
            t = P[1] - P[0]
        elif i == len(P) - 1:
            t = P[-1] - P[-2]
        else:
            t = P[i + 1] - P[i - 1]
        t = t / (np.linalg.norm(t) + 1e-12)
        nrm = np.array([-t[1], t[0]])
        left.append(P[i] + nrm * hw[i])
        right.append(P[i] - nrm * hw[i])
    return np.array(left + right[::-1])


def arc_points(center, r, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return np.c_[center[0] + r * np.cos(a), center[1] + r * np.sin(a)]


def sweep(path, radius, segments=10, mat="mat", cap=True, radius_fn=None):
    """Circular tube swept along a 3D polyline with parallel transport frames."""
    P = np.asarray(path, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    N = np.cross(T[0], ref)
    N /= np.linalg.norm(N)
    rings = []
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    for i in range(n):
        if i > 0:
            # parallel transport
            b = np.cross(T[i - 1], T[i])
            if np.linalg.norm(b) > 1e-9:
                b /= np.linalg.norm(b)
                th = math.acos(max(-1.0, min(1.0, T[i - 1] @ T[i])))
                N = rotate_vec(N, b, th)
        B = np.cross(T[i], N)
        r = radius if radius_fn is None else radius_fn(i / max(n - 1, 1))
        ring = P[i] + r * (np.outer(np.cos(ang), N) + np.outer(np.sin(ang), B))
        rings.append(ring)
    v = np.vstack(rings)
    f = []
    for i in range(n - 1):
        a, b = i * segments, (i + 1) * segments
        for k in range(segments):
            k2 = (k + 1) % segments
            f.append((a + k, a + k2, b + k2, b + k))
    part = Part(v, f, [mat] * len(f))
    if cap:
        part.f.append(tuple(reversed(range(segments))))
        part.m.append(mat)
        part.f.append(tuple(range((n - 1) * segments, n * segments)))
        part.m.append(mat)
    return part


def rotate_vec(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * (axis @ v) * (1 - math.cos(ang))


def catmull(points, samples=8):
    P = np.asarray(points, float)
    if len(P) < 3:
        return np.linspace(P[0], P[-1], samples * 2)
    out = []
    Pe = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]])
    for i in range(1, len(Pe) - 2):
        p0, p1, p2, p3 = Pe[i - 1], Pe[i], Pe[i + 1], Pe[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return np.array(out)


def hexbolt(size=0.03, height=0.018, mat="fd25_steel"):
    """Hex bolt head with washer, base at z=0 pointing +Z."""
    ang = np.linspace(0, 2 * math.pi, 6, endpoint=False) + math.pi / 6
    poly = np.c_[size / 2 * np.cos(ang), size / 2 * np.sin(ang)]
    head = prism(poly, height * 0.7, mat=mat, center_z=height * 0.3 + height * 0.35)
    washer = cylinder(size * 0.62, height * 0.3, segments=10, center=(0, 0, height * 0.15), mat=mat)
    return merge([head, washer])


def place(part, origin, normal, xhint=(1, 0, 0)):
    """Place a part modelled around +Z at origin with Z along normal."""
    return part.transform(frame_z(origin, normal, xhint))


def bolt_circle(center, normal, radius, count, size=0.03, height=0.018, mat="fd25_steel", phase=0.0, xhint=(1, 0, 0)):
    M = frame_z(center, normal, xhint)
    parts = []
    b = hexbolt(size, height, mat)
    for k in range(count):
        a = phase + 2 * math.pi * k / count
        p = np.array([radius * math.cos(a), radius * math.sin(a), 0, 1])
        parts.append(b.transform(M @ mat4(rot_z(a), p[:3])))
    return merge(parts)


def weld_bead(path, radius=0.007, mat="fd25_steel"):
    return sweep(catmull(path, 4), radius, segments=6, mat=mat, cap=False)


def boss(center, axis, r_out, width, r_in=None, segments=40, mat="mat", bevel=0.006):
    """Pin boss (thick ring) around an axis.  If r_in, adds the pin end visible inside."""
    M = frame_z(center, axis)
    p = cylinder(r_out, width, segments, center=(0, 0, 0), mat=mat, bevel=bevel)
    parts = [p.transform(M)]
    if r_in:
        pin = cylinder(r_in, width + 0.03, segments=max(12, segments // 2), mat="fd25_steel", bevel=0.004)
        parts.append(pin.transform(M))
    return merge(parts)
