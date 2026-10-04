# FD-25 controls

The FD-25's hydraulics are run by `lua/controller/fd25Hydraulics.lua`. All of its actions are under
**Options → Controls → Bindings → Vehicle Specific**, where you can rebind anything. The defaults are listed below.

## Two modes

| Mode | Arrow keys / left stick + triggers | Arm keys |
|---|---|---|
| **WORK** (default) | ← / → **swing** the upper structure, ↑ / ↓ **raise / lower the boom**. The tracks are braked. | active |
| **TRAVEL** | ↑ / ↓ drive, ← / → steer. This is skid steering, with a pivot turn when standing still. The machine holds its heading automatically. | still active |

Press **B** to switch modes. A message appears on screen, and the *FD-25 Hydraulics* UI app shows the current mode.

## Keyboard (defaults)

| Function | Key(s) |
|---|---|
| Swing left / right (WORK) | ← / → (steering) |
| Boom raise / lower (WORK) | ↑ / ↓ (throttle / brake) |
| Stick in (pull towards the cab) / out | **K** / **I** |
| Claw curl in / out (tilt) | **L** / **O** |
| Rotate claw counter-clockwise / clockwise | **Y** / **U** |
| Jaw close (grip, crush) / open | **G** / **T** |
| Clamp arm press down / raise | **.** / **,** |
| Work ↔ Travel mode | **B** |
| Precision mode (35 % speed for delicate work) | **Shift + B** |
| Power mode ECO → STANDARD → POWER | **Shift + T** |
| Claw level assist (holds the claw angle while the boom and stick move) | **Shift + G** |
| Auto pose: stow / transport | **Shift + Y** |
| Auto pose: work-ready | **Shift + U** |
| Travel speed LOW ↔ HIGH (3.0 / 5.5 km/h) | **Shift + K** |
| Amber warning beacon on / off | **Shift + O** |
| Work lights | your normal headlight key |

Moving any arm control cancels an auto pose that is running.

> **Lights.** The work lights on the cab roof, boom and rear deck follow the normal headlight switch.
> The amber beacon flashes when you switch it on with **Shift + O**. It also flashes whenever the vehicle's
> standard *light bar* electric is on, the same one BeamNG uses for police and utility light bars.

## Gamepad (XInput defaults)

| Function | Control |
|---|---|
| Swing (WORK) / steer (TRAVEL) | left stick X (normal steering) |
| Boom raise / lower (WORK) / drive (TRAVEL) | RT / LT (normal throttle / brake) |
| Stick in / out | left stick Y |
| Rotate claw | LB / RB |
| Jaw close / open | A / B |
| Claw curl in / out | D-pad down / up |
| Clamp arm down / up | D-pad right / left |
| Work ↔ Travel | Y |
| Precision | X |

All seven hydraulic functions also have **analog axis actions** (`FD-25: … (analog axis)`). You can bind
a joystick, a HOTAS or a second gamepad stick to them for fully proportional control, like real excavator
joysticks. There are also two **track pedal** axes (`FD-25: Left/Right track pedal`) for excavator-style
travel with one lever per track.

## What the hydraulics do

The controller is built to feel like a real load-sensing excavator system:

* **Velocity control with valve-spool lag.** A key or stick sets a cylinder *speed*, not a position. The
  speed builds up and dies away over about 0.1 s, the way a real spool valve does. Swing and rotator ramp more gently.
* **Pump flow sharing.** All functions share one pump. Using many at once slows each of them down, the same
  way a real machine does. Engine speed rises automatically with demand and drops to auto-idle after 4 s of inactivity.
* **End-of-stroke cushioning.** Cylinders slow down just before full stroke.
* **Relief valves.** The jaw, the clamp and every other cylinder have a force limit, so they cannot exceed
  their real system pressure (34.3 MPa). When the jaw crushes something it cannot cut, the cylinder stalls at
  relief and you hear the relief squeal. The bar in the UI app turns red ("RELIEF"). Reversing responds
  instantly, with no wind-up.
* **Swing and rotator range.** Each turns ±170° from centre (340° total), so every orientation is
  reachable.
* **Auto-calibration.** The swing, rotator and track directions calibrate themselves the first time you use
  them. If a direction ever seems inverted, the controller corrects it and prints a message.

## Cameras

* **Driver** (cab view, operator's eye point, like photo 4 of the reference reel)
* **Claw cam**, mounted on the stick and looking down at the shear
* the usual orbit / external cameras

## Tuning without rebuilding

Every speed, range, pressure and behaviour flag lives in `fd25_hydraulics.jbeam`, in the `fd25Hydraulics`
section of each hydraulics part. Edit the numbers in a text editor and press **Ctrl + R** in game to
reload the vehicle. Examples: `maxSpeed` per channel, `pumpFlowMax`, `precisionScale`, `holdTau`,
`autoCalibrate`, `loadHolding`.
