"""Ferrovore FD-25 car dismantler -- single source of truth.

Everything that describes the machine lives here: dimensions, pin locations,
spawn pose, masses, hydraulic cylinders, speeds, joint ranges and colours.
The JBeam generator, the Blender modeller, the texture generator, the
validators and the renderer all read from this module so that meshes and
physics line up by construction.

Coordinate system (BeamNG vehicle frame):
    +X = left, -Y = forward, +Z = up.  Units: metres, kilograms, seconds, radians.

Arm-plane helpers use
    u = forward distance (= -Y), w = height (= Z)
measured in the superstructure frame (before slewing).  Body-local frames use
(s, t, x): s along the body, t = s rotated +90 deg inside the arm plane
(towards +w when the body points forward) and x = world-lateral (+X = left)
offset from the arm centre line.
"""
from types import SimpleNamespace as NS
import math

deg = math.radians

MODEL_ID = "ferrovore_fd25"
BRAND = "Ferrovore"
MODEL_NAME = "FD-25 Car Dismantler"
AUTHOR = "CarDestroyer project"

PHYS_DT = 1.0 / 2000.0  # BeamNG physics step

# ---------------------------------------------------------------- undercarriage
UC = NS(
    track_gauge=2.39,            # track centre to track centre
    shoe_width=0.60,
    track_x=2.39 / 2.0,          # |x| of track centre lines
    # track belt path (pitch line)
    idler_y=-1.825, idler_z=0.45, idler_r=0.36,
    sprocket_y=1.825, sprocket_z=0.47, sprocket_r=0.38,
    bottom_run_z=0.09,
    top_run_sag=0.025,
    shoe_pitch=0.19,
    shoe_plate_t=0.022,
    grouser_h=0.035,
    link_h=0.10,
    n_bottom_rollers=7, bottom_roller_r=0.105,
    n_carrier_rollers=2, carrier_roller_r=0.085,
    # track frame box
    frame_y0=-1.58, frame_y1=1.48, frame_z0=0.24, frame_z1=0.64, frame_w=0.34,
    # centre (car) body
    cb_x=0.55, cb_y=0.95, cb_z0=0.44, cb_z1=0.92,
    slew_ring_r=0.62, slew_ring_z=0.95,
    # hidden steel road wheels (physics only, inside track frames)
    road_wheel_r=0.30, road_wheel_y=(-1.35, -0.45, 0.45, 1.35),
    road_wheel_rays=20, road_wheel_halfwidth=0.11, road_wheel_axle_half=0.24,
    road_wheel_mass=260.0,
    road_wheel_friction=1.15,
    mass=8600.0,
)

# ---------------------------------------------------------------- superstructure
SS = NS(
    frame_z0=1.00, deck_z=1.16,
    front_y=-1.22, rear_y=2.10, half_w=1.375,
    tail_radius=2.75,
    cw_y0=1.92, cw_z0=1.08, cw_z1=2.30,         # counterweight
    hood_y0=0.72, hood_y1=1.95, hood_z=2.36,    # engine hood
    tank_x0=-1.36, tank_x1=-0.52, tank_y0=-1.20, tank_y1=0.68, tank_z=2.08,
    exhaust=(-0.62, 1.48, 2.95),
    slew_axis_low=0.42, slew_axis_high=2.30,
    mass=6300.0,
    counterweight_mass=5400.0,
)

# ---------------------------------------------------------------- cab
CAB = NS(
    x0=0.44, x1=1.37, y0=-1.20, y1=0.56, z0=1.16, z1=3.02,
    riser=0.0,                    # elevated cab variant overrides
    riser_elevated=0.95,
    eye=(0.92, -0.30, 2.60),      # operator eye (standard cab)
    mass=650.0,
    riser_mass=550.0,
)

# ---------------------------------------------------------------- arm
ARM = NS(
    xa=-0.18,                     # arm centre line (world X)
    boom_foot=(0.85, 1.85),       # (u, w) in superstructure frame
    boom_len=5.70,
    boom_knee_s=0.52, boom_knee_t=0.62,
    boom_width=0.56, boom_depth_foot=0.52, boom_depth_knee=0.80, boom_depth_nose=0.50,
    boom_pin_half=0.30,
    boom_mass=2000.0,
    stick_len=2.40,               # boom-stick pin -> stick nose pin
    stick_tail=(-0.62, 0.52),     # stick-cylinder rod lug, stick local (s, t)
    stick_width=0.46, stick_depth=0.46,
    stick_pin_half=0.24,
    stick_mass=1100.0,
    # four-bar linkage
    linkpivot_on_stick=(2.00, 0.10),          # K, stick local
    sidelink_len=0.60,
    hlink_len=0.60,
    tiltcyl_base_on_stick=(-0.10, 0.55),      # T0, stick local
    linkage_mass=220.0,
    # boom-cylinder lugs
    boomcyl_base=(1.12, 1.25),                # superstructure (u, w)
    boomcyl_rod_on_boom=(0.40 * 5.70, -0.33), # boom local (s, t)
    boomcyl_x=0.43,                           # +/- lateral offset from xa
    stickcyl_base_on_boom=(0.53 * 5.70, 1.02),# boom local (s, t)
)

# spawn pose and joint ranges (absolute in-plane angles; 0 = pointing forward)
POSE = NS(
    boom=deg(38.0),               # boom chord angle
    stick_rel=deg(-128.0),        # stick angle relative to boom chord
    att_rel=deg(0.0),             # attachment axis relative to stick
    jaw=deg(18.0),                # moving jaw opening
    slew=0.0,
    rot=0.0,
    clamp=deg(8.0),               # clamp arm angle
)
RANGE = NS(
    boom=(deg(-28.0), deg(62.0)),
    stick_rel=(deg(-160.0), deg(-42.0)),
    att_rel=(deg(-95.0), deg(55.0)),
    jaw=(deg(-2.0), deg(46.0)),
    slew=(deg(-170.0), deg(170.0)),
    rot=(deg(-170.0), deg(170.0)),
    clamp=(deg(-5.0), deg(38.0)),
)

# ---------------------------------------------------------------- attachment
# attachment-local frame (a, b, x): a along the attachment axis, from the
# bracket centre towards the claw tip; b = a rotated +90 deg in the arm plane
# (points forward when the attachment hangs down); x = lateral offset.
ATT = NS(
    pin_spacing=0.44,             # stick-nose pin (b=-0.22) / link pin (b=+0.22)
    pin_half=0.22,                # half pin length (lateral)
    bracket_a=(-0.10, 0.22),
    rotator_a=(0.22, 0.54), rotator_r=0.31, ring_a=(0.46, 0.54),
    head_a=(0.54, 1.42), head_hw=0.25, head_hd=0.24,
    axis_top_a=0.10, axis_bot_a=0.98,
    fork_gap=0.075,               # half gap between fork plates (moving jaw passes)
    fork_plate_t=0.07,
    jaw_pivot=(1.30, -0.06),      # (a, b) in head frame
    jaw_plate_half=0.055,
    jawcyl_base=(0.56, 0.16),     # head frame (a, b)
    jaw_lever=(1.20, -0.34),      # jaw frame at jaw angle 0 (a, b)
    # fixed (fork) jaw centre line, head frame (a, b) -- curves forward to a hook
    fork_line=((1.40, -0.20), (1.70, -0.13), (1.98, -0.03), (2.22, 0.07), (2.40, 0.14), (2.47, 0.24)),
    fork_depth=0.20,
    # moving jaw centre line at jaw angle 0 (closed), head frame (a, b)
    jaw_line=((1.30, -0.06), (1.55, -0.20), (1.85, -0.26), (2.12, -0.22), (2.33, -0.10), (2.44, 0.06)),
    jaw_depth=0.22,
    bracket_mass=420.0, rotator_mass=380.0, head_mass=1050.0, jaw_mass=520.0,
    ring_color=(0.05, 0.22, 0.62),
)

# ---------------------------------------------------------------- clamp arm
CLAMP = NS(
    pivot=(1.96, 0.60),           # undercarriage (u, w), x = +/- pivot_x
    pivot_x=0.62,
    arm_len=2.05,
    fork_half=0.80,               # finger centre |x|
    finger_len=0.45,              # downward finger length
    finger_hook=0.16,
    cyl_base=(1.15, 0.88),        # undercarriage (u, w)
    cyl_x=0.50,
    cyl_rod_on_arm=(0.75, 0.42),  # arm local (s, t)
    mass=1200.0,
)

# ---------------------------------------------------------------- hydraulics
# cylinders: bore/rod in metres; speeds derived from pump flow sharing.
CYL = {
    "boom":  NS(bore=0.135, rod=0.095, count=2, relief=34.3e6, max_speed=0.34, channel="boom", mass=290.0),
    "stick": NS(bore=0.140, rod=0.100, count=1, relief=34.3e6, max_speed=0.42, channel="stick", mass=320.0),
    "tilt":  NS(bore=0.125, rod=0.090, count=1, relief=34.3e6, max_speed=0.40, channel="tilt", mass=190.0),
    "jaw":   NS(bore=0.200, rod=0.110, count=1, relief=32.0e6, max_speed=0.22, channel="jaw", mass=160.0),
    "clamp": NS(bore=0.140, rod=0.090, count=2, relief=30.0e6, max_speed=0.16, channel="clamp", mass=120.0),
}
HYD = NS(
    pump_flow_max=0.0072,          # m^3/s (~430 L/min, two pumps combined)
    idle_rpm=900, work_rpm=1950, max_rpm=2050,
    power_modes={"eco": 0.72, "standard": 0.86, "power": 1.0},
    slew_speed=0.80, slew_accel=0.75, slew_decel=1.1,   # rad/s, rad/s^2
    rot_speed=0.75, rot_accel=2.2,
    precision_scale=0.35,
    spool_tau=0.12,
    cushion=0.05,                 # fraction of stroke for end cushioning
    travel_speed_low=3.0 / 3.6, travel_speed_high=5.5 / 3.6,
)

# ---------------------------------------------------------------- appearance
PAINTS = {
    "Construction Yellow": (0.96, 0.66, 0.05),
    "Safety Orange": (0.95, 0.38, 0.05),
    "Industrial Grey": (0.42, 0.44, 0.45),
    "Recycling Green": (0.12, 0.42, 0.20),
    "Weathered Yellow": (0.78, 0.60, 0.20),
}
DEFAULT_PAINT = "Construction Yellow"
