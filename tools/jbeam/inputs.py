"""Generate vehicle-specific input actions and default bindings for the FD-25.

Writes
  mod/vehicles/ferrovore_fd25/input_actions_fd25.json
  mod/vehicles/ferrovore_fd25/inputmaps/keyboard_fd25.json
  mod/vehicles/ferrovore_fd25/inputmaps/xinput_fd25.json
and returns the control table used by docs/CONTROLS.md.
"""
import json
import os
from collections import OrderedDict

from tools.spec import machine as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)
CTRL = "controller.getControllerSafe('fd25Hydraulics')"

# channel, plus title, minus title, axis title, keyboard plus, keyboard minus
CHANNELS = [
    ("boom", "Boom raise", "Boom lower", "Boom (analog axis, + raises)", None, None),
    ("slew", "Swing left", "Swing right", "Swing (analog axis, + = left)", None, None),
    ("stick", "Stick in (pull towards cab)", "Stick out (push away)", "Stick (analog axis, + = in)", "k", "i"),
    ("tilt", "Claw curl in", "Claw curl out", "Claw tilt (analog axis, + = curl in)", "l", "o"),
    ("rot", "Rotate claw counter-clockwise", "Rotate claw clockwise", "Claw rotation (analog axis)", "y", "u"),
    ("jaw", "Jaw close / grip", "Jaw open", "Jaw (analog axis, + = close)", "g", "t"),
    ("clamp", "Clamp arm press down", "Clamp arm raise", "Clamp arm (analog axis, + = down)", "period", "comma"),
]

TOGGLES = [
    ("fd25_toggle_mode", "Work / Travel mode", "Switch between WORK (arrows swing + boom) and TRAVEL (arrows drive the tracks)",
     f"{CTRL}.toggleMode()", "b"),
    ("fd25_toggle_precision", "Precision control", "Slow, fine hydraulic control for delicate dismantling",
     f"{CTRL}.togglePrecision()", "shift b"),
    ("fd25_cycle_power", "Power mode", "Cycle ECO / STANDARD / POWER engine & pump modes",
     f"{CTRL}.cyclePower()", "shift t"),
    ("fd25_toggle_level", "Claw level assist", "Keep the claw angle constant while the boom and stick move",
     f"{CTRL}.toggleLevelAssist()", "shift g"),
    ("fd25_pose_stow", "Pose: stow / transport", "Automatically fold the arm into transport position",
     f"{CTRL}.poseStow()", "shift y"),
    ("fd25_pose_work", "Pose: work ready", "Automatically move the arm into the dismantling position",
     f"{CTRL}.poseWork()", "shift u"),
    ("fd25_toggle_travel_speed", "Travel speed (hare / tortoise)", "Toggle LOW / HIGH travel speed",
     f"{CTRL}.toggleTravelSpeed()", "shift k"),
    ("fd25_toggle_beacon", "Amber beacon", "Switch the rotating amber warning beacon on the cab roof",
     f"{CTRL}.toggleBeacon()", "shift o"),
]

XINPUT = [
    # control, action, extra
    ("thumbly", "fd25_stick_axis", {}),
    ("btn_l", "fd25_rot_plus", {}),
    ("btn_r", "fd25_rot_minus", {}),
    ("btn_a", "fd25_jaw_plus", {}),
    ("btn_b", "fd25_jaw_minus", {}),
    ("dpadu", "fd25_tilt_minus", {}),
    ("dpadd", "fd25_tilt_plus", {}),
    ("dpadl", "fd25_clamp_minus", {}),
    ("dpadr", "fd25_clamp_plus", {}),
    ("btn_y", "fd25_toggle_mode", {}),
    ("btn_x", "fd25_toggle_precision", {}),
]


def cap(c):
    return c[0].upper() + c[1:]


def build_actions():
    acts = OrderedDict()
    order = 1
    for ch, tplus, tminus, taxis, kplus, kminus in CHANNELS:
        for kind, title in (("plus", tplus), ("minus", tminus)):
            name = f"fd25_{ch}_{kind}"
            acts[name] = OrderedDict([
                ("cat", "vehicle_specific"), ("order", order), ("ctx", "vlua"),
                ("onChange", f"input.event('fd25{cap(ch)}{cap(kind)}', VALUE, FILTERTYPE)"),
                ("title", f"FD-25: {title}"), ("desc", f"Ferrovore FD-25 hydraulics - {title.lower()}"),
            ])
            order += 1
        acts[f"fd25_{ch}_axis"] = OrderedDict([
            ("cat", "vehicle_specific"), ("order", order), ("ctx", "vlua"), ("isCentered", True),
            ("onChange", f"input.event('fd25{cap(ch)}Axis', VALUE, FILTERTYPE)"),
            ("title", f"FD-25: {taxis}"), ("desc", "Bind a joystick / gamepad axis for proportional control"),
        ])
        order += 1
    for side, title in (("L", "Left track pedal"), ("R", "Right track pedal")):
        acts[f"fd25_track{side}_axis"] = OrderedDict([
            ("cat", "vehicle_specific"), ("order", order), ("ctx", "vlua"), ("isCentered", True),
            ("onChange", f"input.event('fd25Track{side}Axis', VALUE, FILTERTYPE)"),
            ("title", f"FD-25: {title} (analog, + = forward)"),
            ("desc", "Excavator-style per-track travel pedal / lever"),
        ])
        order += 1
    for name, title, desc, code, _key in TOGGLES:
        acts[name] = OrderedDict([("cat", "vehicle_specific"), ("order", order), ("ctx", "vlua"),
                                  ("onDown", code), ("title", f"FD-25: {title}"), ("desc", desc)])
        order += 1
    return acts


def build_keyboard():
    b = []
    for ch, *_rest, kplus, kminus in CHANNELS:
        if kplus:
            b.append({"control": kplus, "action": f"fd25_{ch}_plus"})
        if kminus:
            b.append({"control": kminus, "action": f"fd25_{ch}_minus"})
    for name, _t, _d, _c, key in TOGGLES:
        if key:
            b.append({"control": key, "action": name})
    return OrderedDict([("bindings", b), ("name", "Keyboard"), ("devicetype", "keyboard"), ("version", 1)])


def build_xinput():
    b = []
    for control, action, extra in XINPUT:
        d = OrderedDict([("control", control), ("action", action)])
        d.update(extra)
        b.append(d)
    return OrderedDict([("bindings", b), ("name", "XInput Controller"), ("devicetype", "xinput"), ("version", 1)])


def main():
    os.makedirs(os.path.join(VDIR, "inputmaps"), exist_ok=True)
    acts = build_actions()
    with open(os.path.join(VDIR, "input_actions_fd25.json"), "w") as f:
        json.dump(acts, f, indent=1)
    with open(os.path.join(VDIR, "inputmaps", "keyboard_fd25.json"), "w") as f:
        json.dump(build_keyboard(), f, indent=1)
    with open(os.path.join(VDIR, "inputmaps", "xinput_fd25.json"), "w") as f:
        json.dump(build_xinput(), f, indent=1)
    print(f"{len(acts)} input actions written")
    return acts


if __name__ == "__main__":
    main()
