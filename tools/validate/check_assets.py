"""Cross-check every asset reference in the mod (no BeamNG needed).

  * every jbeam flexbody / prop mesh exists as a node in one of the vehicle's .dae files
  * every material used by the DAEs (and by glowMap switches) is defined in main.materials.json
  * every texture referenced by main.materials.json exists on disk
  * the DAEs carry two UV sets (atlas + detail) on every mesh
  * .pc configurations reference existing parts that fit the slots they are put in
  * sounds referenced by the controller exist; every JSON file parses; Lua compiles (LuaJIT)
  * the UI app is complete

Run:  python -m tools.validate.check_assets
"""
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

from tools.spec import machine as M
from tools.validate.jbeam_load import load_parts, expand_table

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOD = os.path.join(ROOT, "mod")
VDIR = os.path.join(MOD, "vehicles", M.MODEL_ID)
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
LIGHTS = {"POINTLIGHT", "SPOTLIGHT"}


class Report:
    def __init__(self):
        self.errors, self.notes = [], []

    def err(self, m):
        self.errors.append(m)

    def note(self, m):
        self.notes.append(m)


def dae_inventory(rep):
    meshes, mats, uvsets = {}, set(), {}
    for fn in sorted(glob.glob(os.path.join(VDIR, "*.dae"))):
        root = ET.parse(fn).getroot()
        geo_uv = {}
        for g in root.iterfind(".//c:library_geometries/c:geometry", NS):
            sets = set()
            for inp in g.iterfind(".//c:input", NS):
                if inp.get("semantic") == "TEXCOORD":
                    sets.add(inp.get("set", "0"))
            geo_uv[g.get("id")] = sets
        for m in root.iterfind(".//c:library_materials/c:material", NS):
            mats.add(m.get("name") or m.get("id"))
        for node in root.iterfind(".//c:visual_scene//c:node", NS):
            name = node.get("name") or node.get("id")
            ig = node.find("c:instance_geometry", NS)
            if ig is None:
                continue
            meshes[name] = os.path.basename(fn)
            uvsets[name] = geo_uv.get(ig.get("url", "").lstrip("#"), set())
    for name, sets in uvsets.items():
        if len(sets) < 2:
            rep.err(f"mesh {name}: expected 2 UV sets (atlas + detail), found {sorted(sets)}")
    return meshes, mats


def main():
    rep = Report()
    # ---------------------------------------------------------------- JSON syntax
    for fn in glob.glob(os.path.join(MOD, "**", "*.json"), recursive=True) + \
            glob.glob(os.path.join(MOD, "**", "*.pc"), recursive=True) + \
            glob.glob(os.path.join(MOD, "**", "*.jbeam"), recursive=True):
        try:
            with open(fn) as f:
                json.load(f)
        except Exception as e:  # noqa: BLE001
            rep.err(f"{os.path.relpath(fn, MOD)}: JSON error {e}")
    # ---------------------------------------------------------------- meshes
    meshes, dae_mats = dae_inventory(rep)
    parts = load_parts()
    used_meshes = set()
    for pname, p in parts.items():
        for fb in expand_table(p.get("flexbodies", [])):
            used_meshes.add(fb["mesh"])
            if fb["mesh"] not in meshes:
                rep.err(f"{pname}: flexbody mesh {fb['mesh']} not found in any .dae")
        for pr in expand_table(p.get("props", [])):
            if pr["mesh"] in LIGHTS:
                continue
            used_meshes.add(pr["mesh"])
            if pr["mesh"] not in meshes:
                rep.err(f"{pname}: prop mesh {pr['mesh']} not found in any .dae")
    unused = sorted(set(meshes) - used_meshes)
    if unused:
        rep.note(f"meshes in DAEs not referenced by any part: {unused}")
    # ---------------------------------------------------------------- materials + textures
    mpath = os.path.join(VDIR, "main.materials.json")
    with open(mpath) as f:
        mats = json.load(f)
    defined = {m.get("mapTo", k) for k, m in mats.items()}
    for m in sorted(dae_mats):
        if m not in defined:
            rep.err(f"DAE material {m} has no entry in main.materials.json")
    glow = {}
    for p in parts.values():
        glow.update(p.get("glowMap", {}))
    for orig, g in glow.items():
        for key in ("off", "on", "on_intense"):
            if key in g and g[key] not in defined:
                rep.err(f"glowMap {orig}.{key} -> {g[key]} is not a defined material")
        if orig not in dae_mats:
            rep.err(f"glowMap source material {orig} is not used by any mesh")
    tex_refs = 0
    for name, m in mats.items():
        if m.get("version") != 1.5:
            rep.err(f"material {name}: version must be 1.5")
        for st in m.get("Stages", []):
            for k, v in st.items():
                if isinstance(v, str) and k.endswith("Map"):
                    tex_refs += 1
                    disk = os.path.join(MOD, v.lstrip("/"))
                    if not os.path.exists(disk):
                        rep.err(f"material {name}.{k}: missing texture {v}")
                    elif k == "normalMap" or k == "detailNormalMap":
                        if not v.endswith(".normal.png"):
                            rep.err(f"material {name}.{k}: normal maps should use the .normal.png suffix ({v})")
                    elif k in ("metallicMap", "roughnessMap", "ambientOcclusionMap", "opacityMap"):
                        if not v.endswith(".data.png"):
                            rep.err(f"material {name}.{k}: linear data maps should use .data.png ({v})")
                    elif k in ("baseColorMap", "emissiveMap") and not v.endswith(".color.png"):
                        rep.err(f"material {name}.{k}: colour maps should use .color.png ({v})")
    tex_dir = os.path.join(VDIR, "textures")
    on_disk = set(os.listdir(tex_dir)) if os.path.isdir(tex_dir) else set()
    referenced = set()
    for m in mats.values():
        for st in m.get("Stages", []):
            for k, v in st.items():
                if isinstance(v, str) and k.endswith("Map"):
                    referenced.add(os.path.basename(v))
    orphans = sorted(on_disk - referenced)
    if orphans:
        rep.err(f"textures on disk not used by any material (dead weight in the zip): {orphans}")
    rep.note(f"{len(mats)} materials, {tex_refs} texture references, {len(on_disk)} texture files")
    # ---------------------------------------------------------------- configurations
    slot_of = {}
    for pname, p in parts.items():
        slot_of[pname] = p.get("slotType")
    for pc in sorted(glob.glob(os.path.join(VDIR, "*.pc"))):
        with open(pc) as f:
            d = json.load(f)
        if d.get("model") != M.MODEL_ID:
            rep.err(f"{os.path.basename(pc)}: model {d.get('model')}")
        for slot, part in d.get("parts", {}).items():
            if part and part not in parts:
                rep.err(f"{os.path.basename(pc)}: unknown part {part}")
            elif part and slot_of.get(part) != slot:
                rep.err(f"{os.path.basename(pc)}: part {part} (slotType {slot_of.get(part)}) put in slot {slot}")
        info = os.path.join(VDIR, "info_" + os.path.basename(pc)[:-3] + ".json")
        if not os.path.exists(info):
            rep.err(f"missing {os.path.basename(info)}")
        thumb = pc[:-3] + ".jpg"
        if not os.path.exists(thumb):
            rep.err(f"missing thumbnail {os.path.basename(thumb)}")
    with open(os.path.join(VDIR, "info.json")) as f:
        info = json.load(f)
    if not os.path.exists(os.path.join(VDIR, info["default_pc"] + ".pc")):
        rep.err(f"info.json default_pc {info['default_pc']} has no .pc")
    if info.get("defaultPaintName1") not in info.get("paints", {}):
        rep.err("info.json defaultPaintName1 is not one of its paints")
    if not glob.glob(os.path.join(VDIR, "default.*")):
        rep.err("missing default.jpg / default.png (vehicle selector preview)")
    # ---------------------------------------------------------------- lua + sounds
    lua_files = glob.glob(os.path.join(VDIR, "lua", "**", "*.lua"), recursive=True)
    try:
        from lupa import luajit21 as lupa       # BeamNG runs LuaJIT (Lua 5.1 syntax)
        L = lupa.LuaRuntime(unpack_returned_tuples=True)
        for fn in lua_files:
            with open(fn) as f:
                src = f.read()
            ok, msg = L.eval("function(s, n) local f, e = loadstring(s, n); return f ~= nil, e end")(src, fn)
            if not ok:
                rep.err(f"Lua syntax: {msg}")
    except ImportError:
        rep.note("lupa not installed: Lua syntax not checked")
    ctrl = os.path.join(VDIR, "lua", "controller", "fd25Hydraulics.lua")
    with open(ctrl) as f:
        csrc = f.read()
    for snd in sorted(set(re.findall(r'"(fd25_[a-z_]+\.ogg)"', csrc))):
        if not os.path.exists(os.path.join(VDIR, "sounds", snd)):
            rep.err(f"controller references missing sound {snd}")
    # ---------------------------------------------------------------- input actions vs inputmaps
    with open(os.path.join(VDIR, "input_actions_fd25.json")) as f:
        actions = json.load(f)
    for fn in glob.glob(os.path.join(VDIR, "inputmaps", "*.json")):
        with open(fn) as f:
            imap = json.load(f)
        for b in imap.get("bindings", []):
            if b["action"] not in actions:
                rep.err(f"{os.path.basename(fn)}: binding to unknown action {b['action']}")
    for name, a in actions.items():
        for key in ("onDown", "onUp", "onChange"):
            s = a.get(key)
            if s and "controller.getControllerSafe" in s:
                fn_name = re.search(r"\)\.(\w+)\(", s)
                if fn_name and f"M.{fn_name.group(1)} =" not in csrc:
                    rep.err(f"action {name} calls missing controller function {fn_name.group(1)}")
    # ---------------------------------------------------------------- UI app
    app = os.path.join(MOD, "ui", "modules", "apps", "FD25Hud")
    for fn in ("app.json", "app.js", "app.png"):
        if not os.path.exists(os.path.join(app, fn)):
            rep.err(f"UI app missing {fn}")
    # ---------------------------------------------------------------- report
    out = ["ASSET CHECK", "=" * 60]
    out += [f"ERROR: {e}" for e in rep.errors]
    out += [f"note: {n}" for n in rep.notes]
    out.append(f"{len(rep.errors)} errors, {len(meshes)} meshes in {len(glob.glob(os.path.join(VDIR, '*.dae')))} DAEs, "
               f"{len(dae_mats)} DAE materials")
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    with open(os.path.join(ROOT, "build", "asset_report.txt"), "w") as f:
        f.write(text + "\n")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
