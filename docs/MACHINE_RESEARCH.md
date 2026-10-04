# What machine is in the reel?

The four screenshots come from a Facebook reel ("Crc MechLife"). It shows a yellow tracked machine tearing
cars apart inside a recycling hall, with the camera in the operator's cab. The machine is a **hydraulic
crawler excavator in the 20–30 t class, set up as an end-of-life-vehicle (ELV) car dismantler**. In
Chinese yards this is called a 拆车机 ("car-dismantling machine"). Several manufacturers sell these
attachments and conversion kits for standard 20–36 t excavators.

## Features seen in the photos

| Feature in the photos | What it is |
|---|---|
| Yellow excavator: tracked undercarriage, cab front-left, engine hood and curved counterweight at the rear | A standard 20–30 t crawler excavator carrier |
| Long straight boom with a stick hanging almost vertically in front of the cab | Mono boom plus a short heavy-duty stick |
| **Blue ring** between the stick and the claw (photos 1 and 4) | The **360° hydraulic rotator** (slewing ring with a hydraulic motor) of the dismantling shear |
| Claw with a **forked fixed lower jaw** and a **hooked moving "eagle-beak" upper jaw** that closes between the forks | The **car-dismantling shear** ("demolition claw"). An internal cylinder drives the moving jaw. The fork tines hold sheet metal while the hook pulls, tears and cuts. |
| U-shaped frame in the foreground of photos 1, 3 and 4, pressing the car down | The **hydraulic clamp arm** ("car-body holder"). It is pinned to the front of the undercarriage, where a dozer blade would normally go, and has two cylinders. It pins the car so the shear can rip doors, roof and wiring loom off it. |
| Interior view from the operator seat (photo 4) | Recreated in the mod as the *driver* camera at the operator's eye point |

## Reference data (22–36 t class)

| Item | Value | Source |
|---|---|---|
| Carrier weight | 20–36 t | CTHB, Hongfang |
| Shear (attachment) weight | 1.7–2.4 t | Hongfang HF200JC / HF300JC, CTHB CD250S |
| Attachment length | ≈ 2.7 m | Hongfang |
| Max. jaw opening | 705–740 mm | Hongfang |
| Rotation | 360° continuous | all |
| Shear force | ≈ 70 t at the tip, ≈ 260 t at the root | Hongfang |
| System pressure | 31–35 MPa | typical 20–30 t excavators |
| Travel speed | 3 / 5.5 km/h | typical 20–30 t excavators |

## How the mod represents it ("Ferrovore FD-25", fictional brand)

All numbers come from `tools/spec/machine.py` and `tools/kinematics.py`.

| | FD-25 |
|---|---|
| Operating weight | **27.9 t** (default config) |
| Undercarriage | ≈ 4.6 m overall track length, 2.39 m gauge, 600 mm shoes, 190 mm pitch, 7 bottom rollers |
| Superstructure | 2.75 m tail-swing radius, 5.4 t counterweight |
| Boom / stick | 5.70 m mono boom, 2.40 m heavy-duty stick, 4-bar attachment linkage |
| Cylinders | Boom 2 × Ø135/95 mm, stick Ø140, tilt Ø125, jaw Ø200, clamp 2 × Ø140; relief 34.3 MPa |
| Shear | Forked fixed jaw plus hooked moving jaw (0–46° opening, about 0.74 m at the tips), ±170° rotator with the blue ring, 2.3 t |
| Clamp arm | 2.05 m U-fork arm with toothed pads, −5° to +38° |
| Reach (stick nose pin) | 8.5 m max reach, 7.7 m max height, 3.2 m below ground |
| Swing | ±170°, 0.8 rad/s |
| Travel | 3.0 / 5.5 km/h, skid / pivot steering |

### Simplifications

* **Swing and rotator travel ±170° (340°) instead of turning continuously.** BeamNG's torsion hydros are
  position servos with limits. This keeps the physics robust, and every orientation is still reachable.
* **The tracks are modelled as eight hidden steel road wheels** (four per side) driven by two electric
  motors in a skid-steer arrangement. The visible track shoes, sprockets and idlers are animated props
  that follow the measured wheel speed. BeamNG has no native chain-track simulation.
* The brand, model name, decals and badges are fictional.

## Sources

* CTHB scrap car dismantler: <https://www.cthbbreaker.com/cthb-product/scrap-car-dismantler/>
* Hongfang car disassemble shear: <https://www.hongfanggroup.com/products_details_1/16.html>
* Kobelco SK210D dismantler: <https://www.kobelco.com.au/product/sk210d/>
* BeamNG documentation:
  * hydros: <https://documentation.beamng.com/modding/vehicle/sections/hydros/>
  * torsion hydros: <https://documentation.beamng.com/modding/vehicle/sections/torsionhydros/>
  * vehicle-specific bindings: <https://documentation.beamng.com/modding/input/vehicle-specific-bindings/>
  * props: <https://documentation.beamng.com/modding/vehicle/sections/props/>
  * materials: <https://documentation.beamng.com/modding/materials/vehicle/typicalmaterials/>
  * paints: <https://documentation.beamng.com/modding/vehicle/vehicle_paints/paint_definition/>
  * coordinate systems: <https://documentation.beamng.com/modding/vehicle/coordinate_systems/>
  * UI apps: <https://documentation.beamng.com/modding/ui/app_creation/>
* BeamNG vehicle Lua, used to verify the engine APIs the controller relies on: <https://github.com/wlkmanist/BeamNG_lua>
