---
name: humanoid-from-spec
description: Design and build a humanoid robot, or a look-alike of a reference robot, from one Python spec, the way the echo Figure 03 replica was built. Proportions come from reference photos and actuators are ones you can buy. The covers clear every joint, the offline physical checks are proven on broken designs, and the Fusion 360 model is built by script through MCP. Ends with a URDF/MJCF with bottom-up masses, a MuJoCo stand test, a retargeted motion-capture walk and a public repo with a README video. Use it whenever someone wants to design, model, replicate, re-proportion or iterate a humanoid or legged robot's mechanical design. It also covers CAD or a URDF/MJCF for a robot they are designing, matching a robot's look from photos or video frames, and how echo was built, even when they don't mention Fusion, a spec or this repo.
---

# Build a humanoid from one spec

This is the method behind echo (github.com/meetsitaram/echo). echo is a 1.727 m, 29-joint humanoid built to Figure 03's published
specs and look, with an Asimov 1-style internal layout and off-the-shelf RobStride actuators. Start from that repo:
its tools read `spec/robot.json` and work for any robot described the same way. "Make it yours" below lists what is
echo-specific.

## Five rules, and why

1. **One source of truth.** Everything is generated from `spec/robot.py`: the Fusion model, the URDF/MJCF, masses,
   renders and the web viewer. Never hand-edit the CAD or `robot.json`. A change is then one edit and a rebuild, and
   the CAD, the simulation and the pictures cannot drift apart.
2. **Check before you build.** Fusion is slow: a full build is ~1,800 timeline features, and a link with many cuts
   takes minutes. The offline checks cover the whole robot in about 2.5 minutes. Make the spec pass, then build.
3. **Prove every check on a broken design.** A check that has never failed may not work. In echo, a cosmetic hip
   cover "connected" two parts after the bracket between them was deleted, and the check passed. Each check has a
   `--bad` case that must fail.
4. **Judge the look in pictures.** Every landmark was in tolerance while the hips read as a "boxy butt". Compare
   side by side at the same scale, from level views, in close-up crops, and from whatever angle the user looked from.
   Numbers hide ugly geometry.
5. **Real parts, honest numbers.** Use actuators someone sells, compute masses bottom-up from materials, and report
   the gap to the target rather than padding the model to hit it.

## Before you start

- Fork echo, then read `LESSONS.md`. It lists every trap we hit with its fix; keep it open.
- Python 3.12 with numpy, scipy, trimesh, manifold3d, mujoco, Pillow, imageio and matplotlib; add mink for
  retargeting. Run scripts that read downloaded files with `python -I`. MuJoCo lives in user site-packages, so run the
  sim and render tools with plain `python`.
- Fusion 360 with an MCP server that runs Python scripts in Fusion. It is only needed for the CAD; everything else
  runs offline.
- On Windows, keep working files on short paths: Python cannot open paths over 260 characters.

## Workflow

### 1. Pin the target

Collect the published specs (height, mass, payload, battery, DOF) and the best reference pictures you can get: pause
videos on exact times and use the largest frames. Write the targets to `spec/targets.json`: landmark heights (chin,
shoulder top, elbow, wrist, fingertip, hip line, knee) and widths, each with its source and tolerance.

Measure heights in level views only; a raised camera lifts every top surface. For a 3/4 frame, fit the camera
angle first: render the model at several angles and keep the one whose silhouette widths match (echo's 3/4 frame
was at 60°, not 40°). Zoom into every region of the reference before modelling it. echo's "black head" turned out
to be a grey knit hood with a black visor, visible only in a close-up.

### 2. Choose the architecture

- **Joints.** List the DOF and the order of each chain's joints. Borrow a proven layout from an open-source robot
  (Asimov 1, Unitree G1/H2, AgiBot X2): download their URDFs and compare joint order, offsets and ranges.
- **Before reordering joints for looks, check where the new first actuator goes.** A yaw-first hip needs a motor
  above each hip; echo had no room for one, so the hip covers were moved to another link instead.
- **Actuators.** Put the catalog in `spec/actuators.json`: size, mass, rated and peak torque, gear, rotor inertia.
  Mark retailer numbers as "confirm before ordering".
- **Torque.** Size each joint with the task poses in step 4, not by feel. Trade capability for looks only knowingly
  and with the user: echo took smaller elbow motors to match Figure's arms, at 3 kg continuous instead of 20.
- **Credit references.** Note the licence of every robot you borrow from. Asimov 1 is CERN-OHL-S-2.0 hardware and
  GPL-2.0 software. echo follows its layout but copies none of its files.

### 3. Write the spec

`spec/robot.py` holds:
- anchor heights at the top (hip, knee, ankle, shoulder, elbow, wrist, hem, neck, chin);
- links made of primitives, each with a role (cover, rigid part, actuator, decal, linkage, cut);
- joints with actuator, range and output side;
- materials.

Write the left side once; the right is mirrored. Actuator stators go in the parent link automatically, and an output
flange on the child.

Place every part relative to the anchors, so re-proportioning is a few numbers. Leave a comment with the reason for
each iteration's numbers next to them; it is the design's memory. See `references/spec-patterns.md`.

### 4. Run the checks until 0 failing

```bash
python -I spec/robot.py
```

```bash
python -I tools/checks.py
```

The checks cover:
- the targets and a bottom-up mass;
- actuators on their axes, and no rigid part inside another;
- covers clear of motors, structure that holds together, internals hidden;
- every joint swept from minimum to maximum, covers included;
- task poses: torque against rated and peak with a 1.5 margin, balance over the feet, limits after balancing, and
  collisions.

Fix the design, not the check. When you add a check, add a `--bad` case for it and watch it fail. See
`references/checks.md`.

### 5. Look at it

```bash
python tools/render_pose.py --views
```

```bash
python -I tools/compare.py
```

These render the spec offline and put it beside the reference frames, at the same height, with the target lines
drawn across both, plus close-up crops of each region. For every change, render before and after side by side from
front, side, back, 3/4 and the user's own angle.

To find gaps that let light through, render on a green background: green inside the outline is light through a gap.

### 6. Give every joint room

Rigid covers are where a humanoid design gets hard. For each joint, decide which link each cover rides on and what
moves against it. Make every interface round about the axis that moves it, or a plane square to that axis. Use a
sphere where several axes cross; spheres also hide seams. The patterns, with numbers, are in
`references/joint-covers.md`: openings, hinges, cups, sockets, the multi-axis hip, pads and the sock-shoe.

### 7. Build it in Fusion

Build one link per MCP call (`build.link(name)`), then rebuild its joints (`build.joints([...])`). After any
timeout, ask `build.status()` before retrying: Fusion keeps working after the call times out. Save the document
into a hub folder, then export `cad/<name>.step`, `cad/<name>.f3d` and one STL per link. See
`references/fusion-mcp.md`. If the viewport cannot save images, the offline renders are the same shapes.

### 8. Robot files and physics

```bash
python -I tools/make_robot_files.py
```

```bash
python tools/sim_stand.py
```

These write the URDF and the MuJoCo model, with masses and inertias from materials and joint armature, damping and
friction from the actuators. The robot must then stand and squat to 90° knees and back in MuJoCo, every joint limited
to its peak torque, with little tilt and no saturation.

### 9. Motion

Retarget a human capture (echo used a PICO full-body suit) to the joints with IK, locking planted feet. Play it back
kinematically for videos and the web viewer. Replay it through the collision test, which catches the combined joint
motions that single-joint sweeps cannot. See `references/motion-and-publishing.md`.

### 10. Review, publish, record

Put the new model in a review page (the web viewer, with movies and renders) and wait for the user's go-ahead.
Then commit and push, with the README video re-rendered and every quoted number copied from the reports. Add a
`LESSONS.md` row for each issue the round hit and how it was fixed. Never publish the reference robot's frames,
panels made from them, raw motion captures or anyone's trademark.

## Each round

1. Change `spec/robot.py`, with the reason in a comment.
2. `python -I spec/robot.py`, then `python -I tools/checks.py`: 0 failing.
3. Render before/after from front, side, back, 3/4 and the user's angle; `compare.py` against the reference;
   a green-background render if any seam changed.
4. `python -I tools/motion_collisions.py` if a joint or cover changed.
5. Show the user the pictures and agree the change.
6. Rebuild the changed links in Fusion, rebuild their joints, check `status()`, save, then export STEP, `.f3d` and
   STLs.
7. `make_robot_files.py`, `sim_stand.py`, `mass_model.py`, the README video and views, and the web viewer export.
8. Update the README numbers from the reports; add the `LESSONS.md` rows; review page; commit; push after the
   go-ahead.

## Make it yours

The tools are generic, but some parts are echo's. Change these for a different robot:
- `spec/targets.json` and `measure()` in `tools/checks.py`: your landmarks and how to measure them on the model;
- `POSES` in `tools/checks.py`: your task poses and payloads;
- `PAIRS`, `FRAME_BOX` and `BANDS` in `tools/compare.py`: your reference frames and the regions to crop;
- the leg and arm name keys in `tools/checks.py` and `tools/motion_collisions.py` (`hip`, `knee`, ...);
- `tools/ik/pico_to_echo.json`: body names for retargeting;
- `NAME` in `tools/make_robot_files.py` and `DOC` in `cad/fusion/build.py`;
- the anchors and every link in `spec/robot.py`.

## Reference files

- `references/spec-patterns.md`: how to write links, joints, covers, cuts and materials. Read before writing or
  changing the spec.
- `references/joint-covers.md`: the joint and cover patterns. Read when a joint collides, a cover shows a gap or
  light, or a joint looks wrong.
- `references/checks.md`: what each check guards, its thresholds, and how to add one. Read when a check fails or
  you add one.
- `references/fusion-mcp.md`: driving Fusion through MCP: build, status, save, export, memory. Read before the
  session's first Fusion call.
- `references/motion-and-publishing.md`: retargeting, videos, the web viewer and the walk collision replay, plus repo
  hygiene, licences and the README. Read when making motion or publishing.
- `references/sources.md`: every reference echo was built from: target specs and frames, robot URDFs, parts data,
  code and tools, what each was used for, and what its licence asks. Read before you borrow anything, and
  start your own list on day one.
- `LESSONS.md` at the repo root: the traps, with what fixed each.
