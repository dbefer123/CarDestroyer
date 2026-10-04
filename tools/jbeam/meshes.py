"""Mesh registry shared by the JBeam generator and the Blender modeller.

FLEXBODIES: mesh name -> owning part, node groups, optional flexbody options.
Prop helpers return dicts describing animated props (cylinders, track,
sprockets, lights).  Prop meshes are modelled in their *prop-local* frame:
    origin = idRef node, X = idRef->idX, Y = idRef->idY (orthogonalised), Z = X x Y
The modeller computes these frames from build/nodes.json.
"""
import math
from collections import OrderedDict

from tools.spec import machine as M

MAIN = "ferrovore_fd25_undercarriage"

FLEXBODIES = OrderedDict([
    # undercarriage
    ("fd25_uc_frame", dict(part=MAIN, groups=["fd_uc"])),
    ("fd25_uc_rollers", dict(part=MAIN, groups=["fd_uc"])),
    # superstructure
    ("fd25_ss_frame", dict(part="fd25_superstructure", groups=["fd_ss"])),
    ("fd25_ss_counterweight", dict(part="fd25_superstructure", groups=["fd_ss"])),
    ("fd25_ss_hood", dict(part="fd25_superstructure", groups=["fd_ss"])),
    ("fd25_ss_tanks", dict(part="fd25_superstructure", groups=["fd_ss"])),
    ("fd25_ss_details", dict(part="fd25_superstructure", groups=["fd_ss"])),
    ("fd25_ss_boomfoot", dict(part="fd25_superstructure", groups=["fd_ss"])),
    ("fd25_ss_decals", dict(part="fd25_superstructure", groups=["fd_ss"])),
    # cab (standard); the elevated part reuses the meshes with a z offset
    ("fd25_cab_shell", dict(part="fd25_cab_standard", groups=["fd_cab"])),
    ("fd25_cab_glass", dict(part="fd25_cab_standard", groups=["fd_cab"])),
    ("fd25_cab_interior", dict(part="fd25_cab_standard", groups=["fd_cab"])),
    ("fd25_cab_guard", dict(part="fd25_cab_standard", groups=["fd_cab"])),
    # boom
    ("fd25_boom", dict(part="fd25_boom_mono", groups=["fd_boom"])),
    ("fd25_boom_hoses", dict(part="fd25_boom_mono", groups=["fd_boom"])),
    ("fd25_hose_boomfoot", dict(part="fd25_boom_mono", groups=["fd_ss", "fd_boom"])),
    # stick and linkage
    ("fd25_stick", dict(part="fd25_stick_hd", groups=["fd_stick"])),
    ("fd25_hose_boomstick", dict(part="fd25_stick_hd", groups=["fd_boom", "fd_stick"])),
    ("fd25_sidelink", dict(part="fd25_stick_hd", groups=["fd_sidelink"])),
    ("fd25_hlink", dict(part="fd25_stick_hd", groups=["fd_hlink"])),
    # attachment
    ("fd25_att_bracket", dict(part="fd25_dismantling_shear", groups=["fd_bracket"])),
    ("fd25_att_rotator", dict(part="fd25_dismantling_shear", groups=["fd_bracket"])),
    ("fd25_att_ring", dict(part="fd25_dismantling_shear", groups=["fd_head"])),
    ("fd25_att_head", dict(part="fd25_dismantling_shear", groups=["fd_head"])),
    ("fd25_att_fork", dict(part="fd25_dismantling_shear", groups=["fd_head"])),
    ("fd25_att_jaw", dict(part="fd25_dismantling_shear", groups=["fd_jaw"])),
    ("fd25_hose_stickatt", dict(part="fd25_dismantling_shear", groups=["fd_stick", "fd_bracket"])),
    # clamp arm
    ("fd25_clamp", dict(part="fd25_clamparm", groups=["fd_clamp"])),
])

ELEVATED_CAB = OrderedDict([
    ("fd25_cab_shell", dict(groups=["fd_cab"], opts={"pos": {"x": 0, "y": 0, "z": M.CAB.riser_elevated}})),
    ("fd25_cab_glass", dict(groups=["fd_cab"], opts={"pos": {"x": 0, "y": 0, "z": M.CAB.riser_elevated}})),
    ("fd25_cab_interior", dict(groups=["fd_cab"], opts={"pos": {"x": 0, "y": 0, "z": M.CAB.riser_elevated}})),
    ("fd25_cab_guard", dict(groups=["fd_cab"], opts={"pos": {"x": 0, "y": 0, "z": M.CAB.riser_elevated}})),
    ("fd25_cab_riser", dict(groups=["fd_cab"], opts={})),
])

STATIC = "fd_static"  # non-existent electrics -> 0 -> prop stays put relative to its frame

# sprocket: 13 teeth matching the track pitch -> pitch radius r = N p / 2 pi
SPROCKET_TEETH = 13
SPROCKET_PITCH_R = SPROCKET_TEETH * M.UC.shoe_pitch / (2 * math.pi)


def track_props(S):
    part = MAIN
    out = []
    # straight runs translate along local Z (= -Y world = forward), meters
    out.append(dict(part=part, func=f"fd_trackPhase{S}", mesh="fd25_track_bottom", ref=f"idl{S}c",
                    idx=f"idl{S}p", idy=f"idl{S}u", trans=(0, 0, -1), opts={"translationUseMeters": True}))
    out.append(dict(part=part, func=f"fd_trackPhase{S}", mesh="fd25_track_top", ref=f"idl{S}c",
                    idx=f"idl{S}p", idy=f"idl{S}u", trans=(0, 0, 1), opts={"translationUseMeters": True}))
    # arcs rotate about local X (= +X world); degrees per metre of phase
    k_i = 180.0 / (math.pi * M.UC.idler_r)
    k_s = 180.0 / (math.pi * SPROCKET_PITCH_R)
    out.append(dict(part=part, func=f"fd_trackPhase{S}", mesh="fd25_track_front", ref=f"idl{S}c",
                    idx=f"idl{S}p", idy=f"idl{S}u", rot=(k_i, 0, 0)))
    out.append(dict(part=part, func=f"fd_trackPhase{S}", mesh="fd25_track_rear", ref=f"spr{S}c",
                    idx=f"spr{S}p", idy=f"spr{S}u", rot=(k_s, 0, 0)))
    out.append(dict(part=part, func=f"fd_idlerAngle{S}", mesh="fd25_idler", ref=f"idl{S}c",
                    idx=f"idl{S}p", idy=f"idl{S}u", rot=(1, 0, 0)))
    out.append(dict(part=part, func=f"fd_sprocketAngle{S}", mesh="fd25_sprocket", ref=f"spr{S}c",
                    idx=f"spr{S}p", idy=f"spr{S}u", rot=(1, 0, 0)))
    return out


def cylinder_props(part, cname, instances):
    """instances: (base, rod, y_base, y_rod).  Barrel at base pointing to rod; rod at rod pointing to base."""
    out = []
    for base, rod, yb, yr in instances:
        out.append(dict(part=part, func=STATIC, mesh=f"fd25_cyl_{cname}_barrel", ref=base, idx=rod, idy=yb))
        out.append(dict(part=part, func=STATIC, mesh=f"fd25_cyl_{cname}_rod", ref=rod, idx=base, idy=yr))
    return out


def light_props(part, ref, idx, idy, range_=30, bright=1.0, func="lowbeam", color=(255, 238, 215)):
    return [dict(part=part, func=func, mesh="POINTLIGHT", ref=ref, idx=idx, idy=idy,
                 opts={"lightRange": range_, "lightBrightness": bright,
                       "lightColor": {"r": color[0], "g": color[1], "b": color[2], "a": 255},
                       "lightAttenuation": {"x": 0, "y": 1, "z": 1},
                       "flareName": "vehicleHeadLightFlare", "flareScale": 0.05,
                       "lightCastShadows": False})]


def cab_props(part, sfx, nodes, lamps, beacon):
    out = []
    any_ = list(nodes.values())
    for lamp in lamps:
        out += light_props(part, lamp, any_[0], any_[1], range_=36, bright=1.2)
    out += [dict(part=part, func="fd_beacon", mesh="POINTLIGHT", ref=beacon, idx=any_[0], idy=any_[1],
                 opts={"lightRange": 14, "lightBrightness": 0.9,
                       "lightColor": {"r": 255, "g": 150, "b": 20, "a": 255},
                       "lightAttenuation": {"x": 0, "y": 1, "z": 1},
                       "flareName": "vehicleHeadLightFlare", "flareScale": 0.04,
                       "lightCastShadows": False, "lightScaling": {}})]
    return out
