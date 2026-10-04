"""Run the whole FD-25 pipeline: JBeam -> meshes -> bake -> textures -> sounds -> info -> renders ->
validation -> zip.  Each stage runs as its own process (several use Blender's bpy module).

Run:  python -m tools.build_all [--skip-bake] [--skip-renders] [--skip-cosim]
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = [
    ("jbeam", ["tools.jbeam.generate"]),
    ("inputs", ["tools.jbeam.inputs"]),
    ("meshes", ["tools.model.build_blend"]),
    ("mesh check", ["tools.validate.check_meshes"]),
    ("bake", ["tools.textures.bake"]),
    ("textures", ["tools.textures.compose"]),
    ("sounds", ["tools.sound.synth"]),
    ("vehicle info", ["tools.meta.vehicle_info"]),
    ("renders", ["tools.render.preview", "final"]),
    ("thumbnails", ["tools.render.preview", "thumbs"]),
    ("animation", ["tools.render.animate", "gif"]),
    ("action still", ["tools.render.animate", "still"]),
    ("jbeam check", ["tools.validate.check_jbeam"]),
    ("asset check", ["tools.validate.check_assets"]),
    ("controller tests", ["tools.validate.test_controller"]),
    ("co-simulation", ["tools.validate.cosim_tests"]),
    ("validation report", ["tools.validate.report"]),
    ("package", ["tools.package"]),
]


def main(argv):
    skip = set()
    if "--skip-bake" in argv:
        skip.add("bake")
    if "--skip-renders" in argv:
        skip.update({"renders", "thumbnails", "animation", "action still"})
    if "--skip-cosim" in argv:
        skip.add("co-simulation")
    env = dict(os.environ, PYTHONPATH=ROOT)
    failed = []
    for name, cmd in STAGES:
        if name in skip:
            print(f"== {name}: skipped")
            continue
        t0 = time.time()
        print(f"== {name} ...", flush=True)
        r = subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT, env=env)
        print(f"== {name}: {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
        if r.returncode != 0:
            failed.append(name)
            if name in ("jbeam", "meshes", "bake", "textures"):
                break
    print("FAILED: " + ", ".join(failed) if failed else "all stages ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
