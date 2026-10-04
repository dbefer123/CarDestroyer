"""Package mod/ into dist/ferrovore_fd25.zip (drop the zip into Documents/BeamNG.drive/<version>/mods).

Run:  python -m tools.package
"""
import os
import sys
import zipfile

from tools.spec import machine as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "mod")
DIST = os.path.join(ROOT, "dist")
STORE = (".png", ".jpg", ".ogg")            # already compressed
LIMIT_MB = 95


def main():
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, f"{M.MODEL_ID}.zip")
    tmp = out + ".tmp"
    n = 0
    with zipfile.ZipFile(tmp, "w") as z:
        for base in ("vehicles", "ui"):
            for dirpath, _dirs, files in os.walk(os.path.join(MOD, base)):
                for fn in sorted(files):
                    if fn.startswith(".") or fn.endswith((".pyc", ".blend", ".blend1")):
                        continue
                    full = os.path.join(dirpath, fn)
                    arc = os.path.relpath(full, MOD).replace(os.sep, "/")
                    comp = zipfile.ZIP_STORED if fn.lower().endswith(STORE) else zipfile.ZIP_DEFLATED
                    z.write(full, arc, compress_type=comp, compresslevel=9 if comp == zipfile.ZIP_DEFLATED else None)
                    n += 1
    os.replace(tmp, out)
    mb = os.path.getsize(out) / 1e6
    print(f"{out}: {n} files, {mb:.1f} MB")
    if mb > LIMIT_MB:
        print(f"WARNING: zip exceeds {LIMIT_MB} MB")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
