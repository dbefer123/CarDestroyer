"""Collect the validation reports into docs/VALIDATION.md.

Run after the validators:  python -m tools.validate.report
"""
import os
import sys

from tools import kinematics as K
from tools.validate.jbeam_load import ROOT

BUILD = os.path.join(ROOT, "build")
SECTIONS = [
    ("Static JBeam check", "jbeam_report.txt",
     "Covers references, triangles, prop frames, rotators, flexbody groups, mass and CoG, hydro ranges, and "
     "the per-node spring/damper budget for BeamNG's 2 kHz explicit integrator. A rank analysis of the "
     "rigidity matrix checks the degrees of freedom (6 rigid-body + 8 wheel spins = 14 expected). There is "
     "also a tipping check with 1.5 t in the claw at every slew angle. Runs for every configuration."),
    ("Asset cross-check", "asset_report.txt",
     "Checks that every flexbody and prop mesh exists in a DAE, and that every DAE material is defined. Every "
     "texture file must exist and carry the right suffix. Also checks the .pc parts and slots, the sounds, the "
     "input actions against the controller functions, the UI app, and Lua syntax under LuaJIT."),
    ("Controller unit tests", "controller_test_report.txt",
     "Runs the real `fd25Hydraulics.lua` in LuaJIT, with mocks of BeamNG's `input`, `electrics`, `obj` and "
     "`guihooks` that follow the engine source, against an ideal hydraulic plant."),
    ("Physics co-simulation", "cosim_report.txt",
     "Runs the generated JBeam (all nodes, beams, hydros, torsion hydros and wheels) in a numpy soft-body "
     "solver at 2 kHz, with the real Lua controller in the loop at 60 Hz. Each scenario sends the same "
     "`input.event` calls as the game and checks the physical outcome."),
]


def main():
    out = ["# Validation report", "",
           "BeamNG.drive is not available in the build environment, so the mod is checked offline. Each "
           "validator below is part of `python -m tools.build_all`, and the reports are copied from `build/`.", ""]
    for title, fn, desc in SECTIONS:
        path = os.path.join(BUILD, fn)
        out += [f"## {title}", "", desc, ""]
        if os.path.exists(path):
            with open(path) as f:
                out += ["```", f.read().rstrip(), "```", ""]
        else:
            out += ["_(report not generated)_", ""]
    out += ["## Kinematics", "", "Spawn pose, cylinder strokes and reach envelope, from `tools/kinematics.py`:", "",
            "```", K.report().rstrip(), "```", ""]
    path = os.path.join(ROOT, "docs", "VALIDATION.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
