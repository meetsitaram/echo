# echo

An open humanoid robot design: 1.727 m tall, 29 joints, built to Figure 03's published specs and look from parts you
can buy. One Python spec generates the CAD, the robot models, the checks and the pictures.

<img src="docs/media/echo_walk.webp" alt="echo walking a retargeted motion capture" width="100%">

A human motion capture retargeted to echo and played back ([MP4](docs/media/echo_walk.mp4)).

![echo from the front, at 60 degrees and from the side](docs/media/echo_views.png)

## Overview

- **Size:** 1.727 m, 57.3 kg (every part's mass from its material; Figure 03 is 61 kg).
- **Joints:** 29: each leg 6, waist 1, neck 2, each arm 7.
- **Actuators:** 29 off-the-shelf RobStride modules (RS00 to RS04 and RS06).
- **Battery:** 2.3 kWh, the same as Figure 03's.
- **Payload:** 10 kg for a short carry, 3 kg continuously.
- **Deliverables:** the Fusion 360 design, a STEP file, and URDF and MuJoCo models.
- **Verified:** 246 physical checks pass, and the robot stands and squats in MuJoCo.

## Design

- **Layout.** One actuator per joint, after the open-source Asimov 1:
  - the hips turn pitch, then roll, then yaw;
  - the waist only turns;
  - the knee motor sits in the thigh and drives a pushrod;
  - two calf motors drive the ankle through rods.
- **Look.** Figure 03's silhouette, measured section by section from its reveal video: a grey knit body, a black
  visor head, black hips and joint discs, and white sneakers. echo carries its own mark, not Figure's logo.
- **Covers that clear every joint.** Rigid covers must move without touching, so each joint gets a shape that turns
  in place: cups round the elbows, a hinge at each knee, sockets at the shoulders and a domed seam at each hip.
- **Real parts, honest numbers.** The actuators are sold today, and masses add up from materials. The 3.7 kg gap to
  Figure's weight stays visible rather than padded.

The joint table, ranges, masses and how each joint was designed are in [docs/DESIGN.md](docs/DESIGN.md).

## Approach

1. **One spec.** `spec/robot.py` places every part relative to a few anchor heights. Everything else is generated
   from it, so a change is one edit.
2. **Check before building.** Offline checks cover:
   - proportions and mass;
   - clearances, and every joint swept through its range;
   - torques and balance in task poses.

   Each check was first proven to catch a deliberately broken design.
3. **Compare with the target.** echo was rendered beside Figure 03's frames at the same scale, and refined section by
   section until it matched.
4. **Build the CAD by script.** Fusion 360 builds the model from the same spec, through MCP.
5. **Export and simulate.** URDF and MuJoCo models carry the bottom-up masses, and a MuJoCo test checks that the
   robot can stand and squat.
6. **Move it.** A human motion capture is retargeted to the joints, replayed to check for collisions, and rendered.

To build your own humanoid this way, follow the [skill doc](skills/humanoid-from-spec/SKILL.md) and keep the
[lessons learned](LESSONS.md) beside it.

## Repository

- `spec/`: the robot, the one source of truth
- `tools/`: checks, masses, renders, robot models, simulation, motion
- `cad/`: the Fusion design (`echo.f3d`), the STEP file and the build scripts
- `urdf/`: URDF and MuJoCo models with meshes
- `reports/`: check results, masses, comparisons and renders
- `docs/`: design details and media
- `skills/`: the method, as a guide and a Claude Code skill

## Quick start

```bash
pip install numpy scipy trimesh manifold3d mujoco pillow imageio matplotlib
```

```bash
python -I spec/robot.py
```

```bash
python -I tools/checks.py
```

```bash
python tools/sim_stand.py
```

The rest of the commands are in [docs/DESIGN.md](docs/DESIGN.md#run-it).

## Licence

Apache-2.0 ([LICENSE](LICENSE)): use it, change it, build on it. The retargeting adapts GMR (MIT); its notice and
the other credits are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Figure and Figure 03 are trademarks of Figure AI, Inc. echo is an independent project, not affiliated with or
endorsed by Figure AI.
