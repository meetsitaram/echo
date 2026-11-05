# echo

An open humanoid robot design: 1.727 m tall, 29 joints, built to Figure 03's published specs and look from parts you
can buy. One Python spec generates the CAD, the robot models, the checks and the pictures.

<img src="docs/media/echo_walk.webp" alt="echo walking a retargeted motion capture" width="100%">

A human motion capture retargeted to echo and played back ([MP4](docs/media/echo_walk.mp4)).

<img src="docs/media/echo_livingroom.webp" alt="echo tidying a living room" width="100%">

echo tidying a living room: the motion is captured from a public video of Figure 03 at work, retargeted to echo and
replayed in the same room, rebuilt from that video ([MP4](docs/media/echo_livingroom.mp4)).

![echo from the front, at 60 degrees and from the side](docs/media/echo_views.png)

## Overview

- **Size:** 1.727 m, 62.2 kg (every part's mass from its material; Figure 03 is 61 kg).
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
- **Real parts, honest numbers.** The actuators are sold today, and masses add up from materials. The 1.2 kg over
  Figure's weight stays visible rather than hidden.

The joint table, ranges, masses and how each joint was designed are in [docs/DESIGN.md](docs/DESIGN.md).

## Approach

1. **One spec.** A single robot description places every part relative to a few anchor heights. Everything else is generated
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
6. **Move it.** A motion capture is retargeted to the joints, replayed to check for collisions, and rendered. The
   capture can be a person in a mocap suit, or a video: a body-pose model reads the robot in each frame, the camera is
   calibrated on the room, and the room itself is rebuilt from the same frames.

## Start here

echo is designed and iterated by an AI agent that follows one skill,
[`.claude/skills/humanoid-from-spec/SKILL.md`](.claude/skills/humanoid-from-spec/SKILL.md), with you reviewing
each round. The skill covers finding reference images on the web, measuring them, editing the spec, checking,
comparing, building the CAD, simulating, motion and publishing. The scripts only do the steps; the agent judges the
results with you.

1. Clone the repo and open it in Claude Code; the skill loads automatically. Other agents: start from `AGENTS.md`.
2. Ask for what you want, for example:
   - "Make echo's hips match Figure 03's front view more closely."
   - "Start a new humanoid modelled on the robot in this video: <link>."
3. Review the before/after pictures the agent shows you each round, and say what to change.

The [lessons learned](LESSONS.md) list every trap we hit and how it was fixed. The agent reads them first.

## Repository

- `spec/`: the robot, the one source of truth
- `tools/`: the steps the skill runs: checks, masses, renders, robot models, simulation, motion
- `cad/`: the Fusion design (`echo.f3d`), the STEP file and the build scripts
- `urdf/`: URDF and MuJoCo models with meshes
- `reports/`: check results, masses, comparisons and renders
- `scenes/`: rooms for echo to work in (the living room: layout measured from a video, CC0 textures)
- `docs/`: design details and media
- `.claude/skills/`: the method, as a guide and a Claude Code skill

## Licence

Apache-2.0 ([LICENSE](LICENSE)): use it, change it, build on it. The retargeting adapts GMR (MIT); its notice and
the other credits are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Figure and Figure 03 are trademarks of Figure AI, Inc. echo is an independent project, not affiliated with or
endorsed by Figure AI.
