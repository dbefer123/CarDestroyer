"""Minimal JBeam loader for validation & co-simulation.

Loads the generated .jbeam files, resolves a configuration (slot -> part),
expands table sections with modifier dictionaries and returns flat lists of
nodes / beams / hydros / torsion hydros / triangles / flexbodies / props.
"""
import glob
import json
import os
from collections import OrderedDict

import numpy as np

from tools.spec import machine as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)

DEFAULT_CONFIG = {
    "fd25_superstructure": "fd25_superstructure",
    "fd25_cab": "fd25_cab_standard",
    "fd25_boom": "fd25_boom_mono",
    "fd25_stick": "fd25_stick_hd",
    "fd25_attachment": "fd25_dismantling_shear",
    "fd25_clamparm": "fd25_clamparm",
    "fd25_hydraulics": "fd25_hydraulics_standard",
}


def load_parts(vdir=VDIR):
    parts = OrderedDict()
    for fn in sorted(glob.glob(os.path.join(vdir, "*.jbeam"))):
        with open(fn) as f:
            data = json.load(f)
        for k, v in data.items():
            v["_file"] = os.path.basename(fn)
            parts[k] = v
    return parts


def expand_table(rows):
    """Expand a JBeam table into a list of dicts (header keys + modifiers + inline opts)."""
    if not rows:
        return []
    header = rows[0]
    out = []
    mods = {}
    for r in rows[1:]:
        if isinstance(r, dict):
            mods.update(r)
            continue
        d = dict(mods)
        inline = None
        cells = r
        if r and isinstance(r[-1], dict) and len(r) > len(header):
            inline = r[-1]
            cells = r[:-1]
        for h, c in zip(header, cells):
            key = h.rstrip(":")
            if key == "[group]":
                key = "group"
            d[key] = c
        if inline:
            d.update(inline)
        out.append(d)
    return out


def resolve(config=None, parts=None):
    parts = parts or load_parts()
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    main = [k for k, v in parts.items() if v.get("slotType") == "main"]
    assert len(main) == 1, main
    chosen = []

    def visit(pname):
        chosen.append(pname)
        p = parts[pname]
        for row in p.get("slots", [])[1:]:
            stype, default = row[0], row[1]
            sel = cfg.get(stype, default)
            if sel and sel in parts:
                visit(sel)

    visit(main[0])
    return chosen, parts


def assemble(config=None):
    chosen, parts = resolve(config)
    V = dict(nodes=OrderedDict(), beams=[], hydros=[], thydros=[], tris=[], flex=[], props=[],
             rotators=[], controllers=[], sections={}, parts=chosen)
    for pn in chosen:
        p = parts[pn]
        for n in expand_table(p.get("nodes", [])):
            name = n["id"]
            if name in V["nodes"]:
                raise ValueError(f"duplicate node {name} in {pn}")
            n["pos"] = np.array([n["posX"], n["posY"], n["posZ"]], float)
            n["part"] = pn
            V["nodes"][name] = n
        for b in expand_table(p.get("beams", [])):
            b["part"] = pn
            V["beams"].append(b)
        for h in expand_table(p.get("hydros", [])):
            h["part"] = pn
            V["hydros"].append(h)
        for t in expand_table(p.get("torsionHydros", [])):
            t["part"] = pn
            V["thydros"].append(t)
        for t in expand_table(p.get("triangles", [])):
            t["part"] = pn
            V["tris"].append(t)
        for fb in expand_table(p.get("flexbodies", [])):
            fb["part"] = pn
            V["flex"].append(fb)
        for pr in expand_table(p.get("props", [])):
            pr["part"] = pn
            V["props"].append(pr)
        for r in expand_table(p.get("rotators", [])):
            r["part"] = pn
            V["rotators"].append(r)
        for c in expand_table(p.get("controller", [])):
            V["controllers"].append(c)
        for k, v in p.items():
            if k not in ("nodes", "beams", "hydros", "torsionHydros", "triangles", "flexbodies", "props",
                         "rotators", "slots", "information", "slotType", "_file", "controller"):
                V["sections"].setdefault(k, v)
    return V


def groups_of(node):
    g = node.get("group", "")
    if isinstance(g, list):
        return [x for x in g if x]
    return [g] if g else []
