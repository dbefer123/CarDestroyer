"""Write info.json, the .pc configurations and their info_<config>.json files.

Run:  python -m tools.meta.vehicle_info      (after tools.jbeam.generate)
"""
import json
import os
from collections import OrderedDict

import numpy as np

from tools.spec import machine as M
from tools.validate.jbeam_load import assemble, DEFAULT_CONFIG

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)

# paint finish per palette entry: (clearcoat, clearcoatRoughness, metallic, roughness)
FINISH = {
    "Construction Yellow": (0.55, 0.22, 0.0, 0.55),
    "Safety Orange": (0.55, 0.22, 0.0, 0.55),
    "Industrial Grey": (0.50, 0.25, 0.15, 0.55),
    "Recycling Green": (0.50, 0.25, 0.0, 0.55),
    "Weathered Yellow": (0.08, 0.60, 0.0, 0.80),
}

CONFIGS = OrderedDict([
    ("fd25_dismantler", dict(
        name="Car Dismantler",
        desc="25-tonne crawler excavator rigged for end-of-life vehicle dismantling: 360-degree rotating "
             "dismantling shear with a forked fixed jaw and hooked moving jaw, plus a hydraulic clamp arm "
             "that pins the car body down while the shear tears it apart.",
        parts={}, paint="Construction Yellow", value=385000, population=4)),
    ("fd25_dismantler_elevated", dict(
        name="Car Dismantler (Elevated Cab)",
        desc="Dismantler with the operator cab raised 0.95 m on a riser for a better view into the car body.",
        parts={"fd25_cab": "fd25_cab_elevated"}, paint="Safety Orange", value=402000, population=2)),
    ("fd25_dismantler_shear_only", dict(
        name="Shear Only (No Clamp Arm)",
        desc="Dismantler without the front clamp arm, for yards that hold the car with a second machine.",
        parts={"fd25_clamparm": "fd25_clamparm_none"}, paint="Weathered Yellow", value=352000, population=2)),
    ("fd25_arcade", dict(
        name="Car Dismantler (Arcade Hydraulics)",
        desc="Same machine with roughly doubled hydraulic speeds and gentler limits - for fun, not realism.",
        parts={"fd25_hydraulics": "fd25_hydraulics_arcade"}, paint="Recycling Green", value=385000, population=1)),
])


def paint_entry(name):
    c, cr, m, r = FINISH[name]
    rgb = M.PAINTS[name]
    return OrderedDict([("baseColor", [round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3), 1.2]),
                        ("clearcoat", c), ("clearcoatRoughness", cr), ("metallic", m), ("roughness", r)])


def config_parts(cfg):
    return OrderedDict((k, cfg["parts"].get(k, v)) for k, v in DEFAULT_CONFIG.items())


def stats(parts):
    V = assemble(parts)
    pos = np.array([n["pos"] for n in V["nodes"].values()])
    mass = sum(float(n.get("nodeWeight", 0)) for n in V["nodes"].values())
    return mass, pos.min(0), pos.max(0)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    info = OrderedDict([
        ("Name", M.MODEL_NAME),
        ("Brand", M.BRAND),
        ("Author", "FD-25 project (procedurally generated)"),
        ("Country", "Fictional"),
        ("Type", "Truck"),
        ("Body Style", "Excavator"),
        ("Derby Class", "Heavy"),
        ("default_pc", next(iter(CONFIGS))),
        ("defaultPaintName1", M.DEFAULT_PAINT),
        ("defaultPaintName2", M.DEFAULT_PAINT),
        ("defaultPaintName3", "Industrial Grey"),
        ("paints", OrderedDict((n, paint_entry(n)) for n in M.PAINTS)),
    ])
    write_json(os.path.join(VDIR, "info.json"), info)
    for key, cfg in CONFIGS.items():
        parts = config_parts(cfg)
        mass, lo, hi = stats(dict(parts))
        pc = OrderedDict([
            ("format", 2),
            ("model", M.MODEL_ID),
            ("parts", parts),
            ("paints", [paint_entry(cfg["paint"]), paint_entry(cfg["paint"]), paint_entry("Industrial Grey")]),
            ("vars", {}),
        ])
        write_json(os.path.join(VDIR, f"{key}.pc"), pc)
        top = M.HYD.travel_speed_high          # m/s
        ci = OrderedDict([
            ("Configuration", cfg["name"]),
            ("Description", cfg["desc"]),
            ("Config Type", "Factory"),
            ("Value", cfg["value"]),
            ("Population", cfg["population"]),
            ("Drivetrain", "Tracked (hydrostatic)"),
            ("Transmission", "Hydrostatic"),
            ("Fuel Type", "Diesel"),
            ("Propulsion", "ICE"),
            ("Induction Type", "Turbo"),
            ("Power", 174),
            ("PowerPeakRPM", "1950 - 2050"),
            ("Torque", 790),
            ("TorquePeakRPM", "1400 - 1600"),
            ("Weight", round(mass)),
            ("Top Speed", round(top, 3)),
            ("Off-Road Score", 95),
            ("Years", {"min": 2024, "max": 2026}),
            ("BoundingBox", [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]]),
            ("defaultPaintName1", cfg["paint"]),
            ("defaultPaintName2", cfg["paint"]),
            ("defaultPaintName3", "Industrial Grey"),
        ])
        write_json(os.path.join(VDIR, f"info_{key}.json"), ci)
        print(f"  {key}: {mass / 1000:.1f} t, bbox {np.round(lo, 2)} .. {np.round(hi, 2)}")


if __name__ == "__main__":
    main()
