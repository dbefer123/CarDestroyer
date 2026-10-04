# Validation report

BeamNG.drive is not available in the build environment, so the mod is checked offline. Each validator below is part of `python -m tools.build_all`, and the reports are copied from `build/`.

## Static JBeam check

Covers references, triangles, prop frames, rotators, flexbody groups, mass and CoG, hydro ranges, and the per-node spring/damper budget for BeamNG's 2 kHz explicit integrator. A rank analysis of the rigidity matrix checks the degrees of freedom (6 rigid-body + 8 wheel spins = 14 expected). There is also a tipping check with 1.5 t in the claw at every slew angle. Runs for every configuration.

```
JBEAM STATIC CHECK
============================================================
[WARN] tipping margin small (0.11 m) at slew 90 deg, far reach
[WARN] tipping margin small (0.15 m) at slew 105 deg, far reach
[WARN] tipping margin small (0.11 m) at slew 270 deg, far reach
[WARN] tipping margin small (0.15 m) at slew 285 deg, far reach
[INFO] config parts: ferrovore_fd25_undercarriage, fd25_superstructure, fd25_cab_standard, fd25_boom_mono, fd25_stick_hd, fd25_dismantling_shear, fd25_clamparm, fd25_hydraulics_standard
[INFO] nodes 726, beams 4232, hydros 7, torsionHydros 6, triangles 429, rotators 8
[INFO] max w*dt 0.900 (beam jw5tL-jw5tR k=3.69e+07 m=22.8/22.8); max c*dt/m 0.192 (beam cfL0o-cfL0i d=4e+03)
[INFO] node budget: max sqrt(sum k/m)*dt 1.40 (jpL), max sum c*dt/m 0.80 (caL3ti)
[INFO] total mass 27924 kg, CoG (-0.037, -0.720, 1.552)
[INFO] rigidity: 2178 coords, rank 2164 -> 14 DOF (expected 14: 6 rigid + 8 wheel spins)  OK
[INFO] hydro anchoring: weakest end bcbR (fd_boom) with axial stiffness share 0.11 (>= 0.05 required)
[INFO] config parts: ferrovore_fd25_undercarriage, fd25_superstructure, fd25_cab_elevated, fd25_boom_mono, fd25_stick_hd, fd25_dismantling_shear, fd25_clamparm, fd25_hydraulics_standard
[INFO] nodes 730, beams 4254, hydros 7, torsionHydros 6, triangles 429, rotators 8
[INFO] max w*dt 0.900 (beam jw5tL-jw5tR k=3.69e+07 m=22.8/22.8); max c*dt/m 0.192 (beam cfL0o-cfL0i d=4e+03)
[INFO] node budget: max sqrt(sum k/m)*dt 1.40 (jpL), max sum c*dt/m 0.80 (caL3ti)
[INFO] total mass 28474 kg, CoG (-0.019, -0.712, 1.584)
[INFO] config parts: ferrovore_fd25_undercarriage, fd25_superstructure, fd25_cab_standard, fd25_boom_mono, fd25_stick_hd, fd25_dismantling_shear, fd25_clamparm_none, fd25_hydraulics_standard
[INFO] nodes 676, beams 3915, hydros 5, torsionHydros 6, triangles 359, rotators 8
[INFO] max w*dt 0.900 (beam jw5tL-jw5tR k=3.69e+07 m=22.8/22.8); max c*dt/m 0.176 (beam jw1kL-jw1fR d=4e+03)
[INFO] node budget: max sqrt(sum k/m)*dt 1.40 (jpL), max sum c*dt/m 0.80 (cbL1b)
[INFO] total mass 26724 kg, CoG (-0.039, -0.602, 1.587)
[INFO] config parts: ferrovore_fd25_undercarriage, fd25_superstructure, fd25_cab_standard, fd25_boom_mono, fd25_stick_hd, fd25_dismantling_shear, fd25_clamparm, fd25_hydraulics_arcade
[INFO] nodes 726, beams 4232, hydros 7, torsionHydros 6, triangles 429, rotators 8
[INFO] max w*dt 0.900 (beam jw5tL-jw5tR k=3.69e+07 m=22.8/22.8); max c*dt/m 0.192 (beam cfL0o-cfL0i d=4e+03)
[INFO] node budget: max sqrt(sum k/m)*dt 1.40 (jpL), max sum c*dt/m 0.80 (caL3ti)
[INFO] total mass 27924 kg, CoG (-0.037, -0.720, 1.552)
[INFO] tipping: worst CoG margin to track edge 0.11 m (payload 1500 kg in claw)
```

## Asset cross-check

Checks that every flexbody and prop mesh exists in a DAE, and that every DAE material is defined. Every texture file must exist and carry the right suffix. Also checks the .pc parts and slots, the sounds, the input actions against the controller functions, the UI app, and Lua syntax under LuaJIT.

```
ASSET CHECK
============================================================
note: 20 materials, 82 texture references, 69 texture files
0 errors, 45 meshes in 6 DAEs, 18 DAE materials
```

## Controller unit tests

Runs the real `fd25Hydraulics.lua` in LuaJIT, with mocks of BeamNG's `input`, `electrics`, `obj` and `guihooks` that follow the engine source, against an ideal hydraulic plant.

```
CONTROLLER UNIT TESTS (ideal plant)
============================================================
[PASS] init: mode=work, channels=['boom', 'stick', 'tilt', 'jaw', 'clamp', 'slew', 'rot']
[PASS] hold at spawn: max cylinder drift 0.00 mm with no input
[PASS] sound sources: event:>Vehicle>Hydraulics>Pump_Big, fd25_diesel_loop.ogg, fd25_relief_loop.ogg, fd25_track_loop.ogg
[PASS] jaw moves: stroke change +98 mm in 1 s (max speed 0.22 m/s)
[PASS] stick moves: stroke change +79 mm in 1 s (max speed 0.42 m/s)
[PASS] tilt moves: stroke change +337 mm in 1 s (max speed 0.40 m/s)
[PASS] precision mode: precision/normal stroke ratio 0.44 (target 0.35)
[PASS] pump flow sharing: stick stroke alone 115 mm vs. with 5 functions 35 mm, demand 595% of pump
[PASS] relief stall detection: stalled=True, command 35.0 mm past the blocked cylinder (windup 35.0 mm)
[PASS] relief sound: relief loop volume 0.55
[PASS] instant reversal after stall: jaw command moved +0.176 within 0.25 s of reversing
[PASS] work mode: steering swings: swing target +35.2 deg after 1 s left steer
[PASS] work mode: throttle raises boom: boom cylinder +191 mm after 1 s throttle
[PASS] work mode: tracks braked: brakes L/R 1.00/1.00
[PASS] toggle travel mode: mode=travel
[PASS] travel forward: track commands L +0.56 R +0.56, speed 0.89 m/s
[PASS] travel alarm: 1 alarm source playing
[PASS] pivot turn: pivot right: L +0.84 R -0.87
[PASS] parking brakes when idle: brakes 1.00/1.00, speed -0.000
[PASS] travel speed toggle: travelHigh=True
[PASS] auto idle / rev up: idle 900 rpm -> working 1814 rpm
[PASS] power mode cycle: after one press power, after three standard
[PASS] pose macro starts / cancels on input: active=True, cancelled=True
[PASS] pose macro completes: macro state after 12 s: None
[PASS] claw level assist (closed loop): residual pitch error -0.00 deg after an 8.6 deg disturbance, +0.00 deg after it was removed
[PASS] beacon pulses: fd_beacon range 0.00..1.00
[PASS] HUD data: FD25HudUpdate carries mode/rpm/pressure/flow/cylinders
[PASS] reset: after reset: macro=None, jaw cmd +0.000
[PASS] config fd25_hydraulics_arcade: controller initialises and runs
[PASS] config fd25_clamparm_none: controller initialises and runs
[PASS] config fd25_cab_elevated: controller initialises and runs
31/31 passed
```

## Physics co-simulation

Runs the generated JBeam (all nodes, beams, hydros, torsion hydros and wheels) in a numpy soft-body solver at 2 kHz, with the real Lua controller in the loop at 60 Hz. Each scenario sends the same `input.event` calls as the game and checks the physical outcome.

```
PASS  settle: KE 652 J, max strain 1.21 %, ref z 0.426 m
PASS  boom raise: cylinder 2.452->2.671 m, boom nose z 5.22->6.78 m
PASS  boom lower: cylinder 2.671->2.045 m
PASS  stick out: cylinder 3.572->2.624 m, stick nose y -5.33->-7.77
PASS  stick in: cylinder 2.624->3.307 m
PASS  tilt in/out: cylinder 1.968->2.505->1.981 m
PASS  jaw open/close: tip gap 0.61->1.14->0.15 m, cylinder 0.722->0.582->0.819
PASS  clamp raise: finger z 0.29->1.27 m
PASS  claw into ground -> relief or self-lift: claw z 0.02 m, stall detected=True, peak cylinder force 519 kN (relief 516 kN), machine ref z 1.36 m
PASS  swing left: slew -0.0 -> 94.0 deg; calibration msgs ['FD-25: Swing direction calibrated']
PASS  swing right: slew -> -36.9 deg
PASS  rotator: rot -0.0 -> 46.7 deg
PASS  pose stow macro: macro finished: True; boom stroke 62%
PASS  claw level assist: attachment pitch change during 1.6 s stick-out: 55.2 deg without, 1.3 deg with assist
PASS  travel forward (straight): moved 5.54 m along the initial heading (sideways +0.01 m), heading change -0.3 deg, track phase animated=True
PASS  pivot turn: yaw 13.1 -> -17.2 deg
PASS  travel brakes hold: drift 0.3 mm in 2 s with no input
PASS  structure integrity: max strain 1.52% at end, sim time 89.4 s
messages: ['FD-25: Swing direction calibrated', 'FD-25: Rotator direction calibrated', 'FD-25: moving to stow pose (touch any control to cancel)', 'FD-25: pose reached', 'FD-25: moving to work pose (touch any control to cancel)', 'FD-25: pose reached', 'FD-25: claw level assist ON', 'FD-25: claw level assist OFF', 'FD-25: TRAVEL mode (arrows drive the tracks)', 'FD-25: WORK mode (arrows swing / boom)']
wall time 773 s
```

## Kinematics

Spawn pose, cylinder strokes and reach envelope, from `tools/kinematics.py`:

```
== Spawn pose key points (world X, Y, Z) ==
  boom_foot         -0.180  -0.850   1.850
  boom_nose         -0.180  -5.342   5.359
  stick_nose        -0.180  -5.342   2.959
  claw_tip_fixed    -0.180  -5.802   0.489
  claw_tip_jaw      -0.180  -5.264   0.538
  clamp_tip          0.000  -4.300   0.474
== Cylinders: spawn / min / max / ratio / stroke ==
  boom_l   spawn 2.456  min 1.761  max 2.690  ratio 1.53  stroke 0.928  monotonic=True linkage_ok=True
  stick    spawn 3.573  min 2.576  max 3.676  ratio 1.43  stroke 1.100  monotonic=True linkage_ok=True
  tilt     spawn 1.965  min 1.609  max 2.551  ratio 1.59  stroke 0.942  monotonic=True linkage_ok=True
  jaw      spawn 0.721  min 0.581  max 0.822  ratio 1.42  stroke 0.241  monotonic=True linkage_ok=True
  clamp_l  spawn 1.513  min 1.252  max 1.595  ratio 1.27  stroke 0.343  monotonic=True linkage_ok=True
== Envelope (stick nose pin): max reach 8.48 m, max height 7.70 m, min z -3.22 m
```
