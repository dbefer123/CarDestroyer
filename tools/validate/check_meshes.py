"""Mesh sanity: every closed connected component must be outward-facing (positive signed volume).

Run: python -m tools.validate.check_meshes
"""
import sys

import numpy as np

from tools.model import parts_uc, parts_ss, parts_arm, parts_att


def components(part):
    parent = list(range(len(part.v)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for f in part.f:
        r = find(f[0])
        for i in f[1:]:
            ri = find(i)
            if ri != r:
                parent[ri] = r
    comp = {}
    for fi, f in enumerate(part.f):
        comp.setdefault(find(f[0]), []).append(fi)
    return list(comp.values())


def check_part(name, part):
    bad, closed = 0, 0
    for faces in components(part):
        edges = {}
        vol = 0.0
        for fi in faces:
            f = part.f[fi]
            a = part.v[f[0]]
            for i in range(1, len(f) - 1):
                vol += np.dot(a, np.cross(part.v[f[i]], part.v[f[i + 1]])) / 6
            for k in range(len(f)):
                e = (f[k], f[(k + 1) % len(f)])
                edges[e] = edges.get(e, 0) + 1
        is_closed = all(edges.get((b, a), 0) == 1 for (a, b) in edges)
        if is_closed:
            closed += 1
            if vol < -1e-9:
                bad += 1
    return bad, closed


def main():
    meshes = {}
    meshes.update(parts_uc.undercarriage_meshes())
    meshes.update(parts_uc.track_meshes())
    meshes.update(parts_uc.prop_meshes_local())
    meshes.update(parts_ss.superstructure_meshes())
    meshes.update(parts_ss.cab_meshes())
    meshes.update(parts_arm.arm_meshes())
    meshes.update(parts_att.attachment_meshes())
    meshes.update(parts_att.clamp_meshes())
    total_bad = 0
    for n, p in meshes.items():
        bad, closed = check_part(n, p)
        total_bad += bad
        flag = "  <-- INVERTED" if bad else ""
        print(f"{n:24s} closed components {closed:5d}  inverted {bad}{flag}")
    return total_bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
