"""Tiny JBeam DSL.

Collects nodes, beams, triangles and arbitrary sections per *part* and writes
strict-JSON .jbeam files (BeamNG also accepts its relaxed syntax, but strict
JSON keeps every downstream tool simple).  Table sections are emitted with
BeamNG "modifier" dictionaries: a dict row changes the defaults for all rows
that follow it, so consecutive rows sharing options are grouped.
"""
import json
from collections import OrderedDict

import numpy as np

FLT_MAX = "FLT_MAX"

NODE_DEFAULTS = OrderedDict([
    ("nodeWeight", 25.0), ("collision", True), ("selfCollision", False),
    ("staticCollision", True), ("frictionCoef", 1.0), ("nodeMaterial", "|NM_METAL"),
    ("group", ""),
])
BEAM_DEFAULTS = OrderedDict([
    ("beamType", "|NORMAL"), ("beamSpring", 4300000), ("beamDamp", 580),
    ("beamDeform", FLT_MAX), ("beamStrength", FLT_MAX),
])
TRI_DEFAULTS = OrderedDict([("groundModel", "metal"), ("dragCoef", 0), ("triangleType", "NORMALTYPE")])


def r4(v):
    return float(round(float(v), 4))


class Node:
    __slots__ = ("name", "pos", "opts", "part", "body")

    def __init__(self, name, pos, opts, part, body):
        self.name, self.pos, self.opts, self.part, self.body = name, np.asarray(pos, float), opts, part, body


class Vehicle:
    """Holds every part; node names are global across parts (as in BeamNG)."""

    def __init__(self):
        self.parts = OrderedDict()
        self.nodes = OrderedDict()
        self.beam_keys = set()

    def clamp_stability(self, dt, w_node=1.4, c_node=0.8, w_beam=0.9, c_beam=0.6):
        """Scale beam/hydro springs and dampers so every node stays inside an explicit-integration
        budget at the physics rate:  sqrt(sum k / m) * dt <= w_node  and  sum c * dt / m <= c_node,
        plus the classic per-beam limit.  Returns (#springs reduced, #dampers reduced)."""
        from collections import defaultdict
        rows = [r for p in self.parts.values() for r in (p.beam_rows + p.hydro_rows)]
        S, D = defaultdict(float), defaultdict(float)
        for a, b, o in rows:
            S[a] += o["beamSpring"]; S[b] += o["beamSpring"]
            D[a] += o["beamDamp"]; D[b] += o["beamDamp"]

        def m(n):
            return self.nodes[n].opts["nodeWeight"]

        n_k = n_d = 0
        for a, b, o in rows:
            fk = min(1.0, (w_node / dt) ** 2 * m(a) / S[a], (w_node / dt) ** 2 * m(b) / S[b])
            inv = 1.0 / m(a) + 1.0 / m(b)
            kmax = min(o["beamSpring"] * fk, (w_beam / dt) ** 2 / inv)
            if kmax < o["beamSpring"] * 0.999:
                o["beamSpring"] = float(f"{kmax:.3g}")
                n_k += 1
            fd = min(1.0, c_node * m(a) / (dt * D[a]), c_node * m(b) / (dt * D[b]))
            dmax = min(o["beamDamp"] * fd, c_beam / (dt * inv))
            if dmax < o["beamDamp"] * 0.999:
                o["beamDamp"] = float(f"{dmax:.3g}")
                n_d += 1
        return n_k, n_d

    def part(self, name, **kw):
        if name not in self.parts:
            self.parts[name] = Part(self, name, **kw)
        return self.parts[name]

    def pos(self, n):
        return self.nodes[n].pos

    def node_index(self):
        return {k: dict(pos=[r4(c) for c in v.pos], part=v.part, body=v.body,
                        group=v.opts.get("group", ""), weight=v.opts.get("nodeWeight", 25.0),
                        opts={k2: v2 for k2, v2 in v.opts.items() if k2 not in ("group",)})
                for k, v in self.nodes.items()}


class Part:
    def __init__(self, veh, name, slotType, info_name, description="", value=0):
        self.veh = veh
        self.name = name
        self.d = OrderedDict()
        self.d["information"] = OrderedDict([("authors", "CarDestroyer project"), ("name", info_name), ("value", value)])
        self.d["slotType"] = slotType
        self.description = description
        self.node_rows = []
        self.beam_rows = []
        self.tri_rows = []
        self.hydro_rows = []
        self.thydro_rows = []
        self.flex_rows = []
        self.prop_rows = []
        self.slots = []
        self.extra = OrderedDict()

    # ------------------------------------------------------------ elements
    def node(self, name, pos, body="", **opts):
        if name in self.veh.nodes:
            raise ValueError(f"duplicate node {name}")
        o = OrderedDict(NODE_DEFAULTS)
        o.update(opts)
        n = Node(name, pos, o, self.name, body)
        self.veh.nodes[name] = n
        self.node_rows.append(n)
        return name

    def beam(self, a, b, **opts):
        if a == b:
            raise ValueError(f"degenerate beam {a}")
        key = (min(a, b), max(a, b))
        if key in self.veh.beam_keys:
            return
        self.veh.beam_keys.add(key)
        o = OrderedDict(BEAM_DEFAULTS)
        o.update(opts)
        self.beam_rows.append((a, b, o))

    def tri(self, a, b, c, **opts):
        o = OrderedDict(TRI_DEFAULTS)
        o.update(opts)
        self.tri_rows.append((a, b, c, o))

    def hydro(self, a, b, **opts):
        o = OrderedDict(BEAM_DEFAULTS)
        o.update(opts)
        self.hydro_rows.append((a, b, o))

    def torsion_hydro(self, a, b, c, d, **opts):
        self.thydro_rows.append((a, b, c, d, opts))

    def flexbody(self, mesh, groups, **opts):
        self.flex_rows.append((mesh, list(groups), opts))

    def prop(self, func, mesh, ref, idx, idy, base_rot=(0, 0, 0), rot=(0, 0, 0), trans=(0, 0, 0),
             vmin=-1e9, vmax=1e9, offset=0, mult=1, **opts):
        self.prop_rows.append((func, mesh, ref, idx, idy, base_rot, rot, trans, vmin, vmax, offset, mult, opts))

    def slot(self, slot_type, default, description, **opts):
        self.slots.append((slot_type, default, description, opts))

    def section(self, key, value):
        self.extra[key] = value

    # ------------------------------------------------------------ output
    @staticmethod
    def _table(header, rows, keys_defaults):
        """rows: list of (cells, opts).  Emits modifier dicts when options change."""
        out = [header]
        state = OrderedDict()
        for cells, opts in rows:
            delta = OrderedDict()
            for k, v in opts.items():
                if state.get(k, object()) != v:
                    delta[k] = v
                    state[k] = v
            if delta:
                out.append(delta)
            out.append(cells)
        return out

    def to_dict(self):
        d = OrderedDict(self.d)
        if self.slots:
            rows = [["type", "default", "description"]]
            for t, dflt, desc, o in self.slots:
                r = [t, dflt, desc]
                if o:
                    r.append(o)
                rows.append(r)
            d["slots"] = rows
        for k, v in self.extra.items():
            d[k] = v
        if self.node_rows:
            rows = []
            for n in self.node_rows:
                opts = OrderedDict(n.opts)
                g = opts.pop("group")
                opts["group"] = g
                rows.append(([n.name, r4(n.pos[0]), r4(n.pos[1]), r4(n.pos[2])], opts))
            d["nodes"] = self._table(["id", "posX", "posY", "posZ"], rows, NODE_DEFAULTS)
            d["nodes"].append({"group": "", "collision": True, "selfCollision": False, "frictionCoef": 1.0,
                               "nodeWeight": 25.0, "staticCollision": True})
        if self.beam_rows:
            rows = [([a, b], o) for a, b, o in self.beam_rows]
            d["beams"] = self._table(["id1:", "id2:"], rows, BEAM_DEFAULTS)
            d["beams"].append(OrderedDict(BEAM_DEFAULTS))
        if self.hydro_rows:
            rows = [([a, b], o) for a, b, o in self.hydro_rows]
            d["hydros"] = self._table(["id1:", "id2:"], rows, BEAM_DEFAULTS)
        if self.thydro_rows:
            out = [["id1:", "id2:", "id3:", "id4:"]]
            for a, b, c, dd, o in self.thydro_rows:
                out.append([a, b, c, dd, o])
            d["torsionHydros"] = out
        if self.tri_rows:
            rows = [([a, b, c], o) for a, b, c, o in self.tri_rows]
            d["triangles"] = self._table(["id1:", "id2:", "id3:"], rows, TRI_DEFAULTS)
        if self.flex_rows:
            out = [["mesh", "[group]:", "nonFlexMaterials"]]
            for mesh, groups, o in self.flex_rows:
                r = [mesh, groups]
                if o:
                    r.append(o)
                out.append(r)
            d["flexbodies"] = out
        if self.prop_rows:
            out = [["func", "mesh", "idRef:", "idX:", "idY:", "baseRotation", "rotation", "translation",
                    "min", "max", "offset", "multiplier"]]
            for (func, mesh, ref, idx, idy, br, rot, tr, vmin, vmax, off, mult, o) in self.prop_rows:
                r = [func, mesh, ref, idx, idy,
                     {"x": br[0], "y": br[1], "z": br[2]},
                     {"x": rot[0], "y": rot[1], "z": rot[2]},
                     {"x": tr[0], "y": tr[1], "z": tr[2]},
                     vmin, vmax, off, mult]
                oo = OrderedDict([("baseTranslation", {"x": 0, "y": 0, "z": 0})])
                oo.update(o)
                r.append(oo)
                out.append(r)
            d["props"] = out
        return d


def _row(v):
    return json.dumps(v, separators=(", ", ":"))


def dumps_jbeam(data):
    """Strict JSON with one table row per line (readable, diff-friendly)."""
    lines = ["{"]
    pitems = list(data.items())
    for pi, (pname, pdict) in enumerate(pitems):
        lines.append(f"{json.dumps(pname)}: {{")
        sitems = list(pdict.items())
        for si, (sk, sv) in enumerate(sitems):
            comma = "," if si < len(sitems) - 1 else ""
            if isinstance(sv, list):
                lines.append(f"    {json.dumps(sk)}: [")
                for ri, row in enumerate(sv):
                    rc = "," if ri < len(sv) - 1 else ""
                    lines.append("        " + _row(row) + rc)
                lines.append("    ]" + comma)
            elif isinstance(sv, dict) and sv and any(isinstance(x, (dict, list)) for x in sv.values()):
                body = json.dumps(sv, indent=4).replace("\n", "\n    ")
                lines.append(f"    {json.dumps(sk)}: {body}{comma}")
            else:
                lines.append(f"    {json.dumps(sk)}: {_row(sv)}{comma}")
        lines.append("}" + ("," if pi < len(pitems) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_jbeam(path, parts):
    data = OrderedDict()
    for p in parts:
        data[p.name] = p.to_dict()
    txt = dumps_jbeam(data)
    json.loads(txt)  # sanity: must be strict JSON
    with open(path, "w") as f:
        f.write(txt)
    return path
