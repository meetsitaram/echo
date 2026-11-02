# References used to build echo

Everything echo was built from, what it was used for, and what its licence asks. When you build your own robot,
keep a list like this from day one. It is how you credit people correctly and avoid copying something you may not.

## The target robot

| reference | used for | terms |
|---|---|---|
| Figure 03 specifications, figure.ai/figure (read 2026-10-31) | height 1.727 m (5'8"), mass 61 kg, payload 20 kg, 5 h runtime, 1.2 m/s, electric (`spec/targets.json`) | facts; "Figure" and "Figure 03" are Figure AI's trademarks: say "replica of", disclaim affiliation, don't use the logo |
| Figure 03 battery and charging, figure.ai | 2.3 kWh pack, 2 kW wireless charging through the feet (battery mass and the foot coil) | facts |
| Figure 03 reveal video, figure.ai (frames: side 0:21.5, 3/4 0:24.5, front 1:50) | proportions, landmark heights and widths, and close-ups of each region (`tools/compare.py`) | copyrighted: kept locally for comparison and never committed (`.gitignore`: `reports/compare/figure_*` and the panels made from them) |

## Internal layout and comparison robots

| reference | used for | licence | obligations for echo |
|---|---|---|---|
| Asimov 1, github.com/menloresearch/asimov-1 (`sim-model/asimov_1_25dof`, `_27dof` URDFs, README) | the layout: one actuator per joint, stator in the parent, 7075 brackets, covers; hip pitch-roll-yaw; yaw-only waist; parallel ankle; joint orders and ranges compared | hardware CERN-OHL-S-2.0 (strongly reciprocal), software GPL-2.0 | none: no files copied. Using their CAD or code would bring their licences with it |
| Unitree G1 (`g1_29dof.urdf`), H1-2 (`h1_2.urdf`), H2 (`H2.urdf`), github.com/unitreerobotics/unitree_ros | joint orders, offsets and ranges; the hip order decision (H1-2 is yaw first, G1 and H2 pitch first) | BSD-3-Clause | none: read only |
| AgiBot A3 (`a3_t3d0/urdf/model.urdf`), github.com/AgibotTech/A3-A3U-robot-model | the same comparisons; the shoulder roll range (A3 stops at 150 deg) | Mulan PSL v2 | none: read only |
| AgiBot X2 (`X2-Ultra.urdf`, v1.3 and v1.4), github.com/AgibotTech/agibot_x2_urdf | the same comparisons | Mulan PSL v2 | none: read only |

## Parts and material data

| reference | used for | notes |
|---|---|---|
| RobStride RS00, RS02, RS03, RS04, RS06 actuators, from retailer listings (roboticscenter.ai, pakronics, openelab) | size, mass, rated and peak torque, speed (`spec/actuators.json`) | not vendor data: confirm with RobStride's datasheets and STEP files before ordering |
| Unitree's 7520 motor (G1 armature 0.0102 at 14.3:1) | estimated rotor inertia for each RobStride frame size, for joint armature | estimate, scaled by frame size |
| Standard densities: 7075-T6 2810 kg/m3, PA12 (MJF) 1010, polycarbonate 1200, TPU 1200, EVA foam 250 | bottom-up masses (`spec/robot.py` MATERIALS, `tools/mass_model.py`) | engineering handbook values |
| Battery packs at 175 Wh/kg with BMS and case; a Jetson-class compute module at 1.6 kg | explicit masses for the pack (13.1 kg) and compute | estimates |

## Process

| reference | used for |
|---|---|
| CDX-3R, the author's earlier robot project: `cad/fusion/LESSONS.md` and `docs/PIPELINE-PLAYBOOK.md` | the pipeline: one spec, offline checks before Fusion, every check proven on a broken design, Fusion through MCP one short job per call |

## Motion

| reference | used for | licence | obligations for echo |
|---|---|---|---|
| PICO full-body tracking with XRoboToolkit's 24-joint body | the operator's walk (session approved by the operator) | capture device and SDK terms | the raw capture is not published (`reports/motion/` ignored); only the robot's retargeted motion is shown |
| GMR, General Motion Retargeting, github.com/YanjieZe/GMR | the IK table (from `xrobot_to_g1.json`) and the two-stage solve loop (`motion_retarget.py`) | MIT | keep GMR's copyright and licence notice with echo (`THIRD_PARTY_NOTICES.md`) |
| agibot-x2-record-and-replay, github.com/meetsitaram/agibot-x2-record-and-replay (the author's) | `x2_ik.py`, the X2 retargeter that `tools/retarget_pico.py` was ported from | the author's own | none |
| The author's fork of NVIDIA's GR00T-WholeBodyControl, github.com/meetsitaram/GR00T-WholeBodyControl (gwbc-supermimic) | the author's own scripts there (`pico_tape_to_x2_gmr.py`, `build_gmr_x2_asset.py`): frame transform, segment morph, arm-plane frames, elbow posture task, rest calibration | the repo is Apache-2.0; these scripts are the author's | none for echo (no NVIDIA-authored code was ported); credited |
| mink, github.com/kevinzakka/mink | the IK solver the retargeter calls | Apache-2.0 | none: a dependency, not copied |
| Figure's home-work video, x.com/Figure_robot (2026-09-17), the living room at 3:28:40 | the living-room motion and the room's layout, measured from it | Figure's copyright; X's terms | the clip, its frames and the poses read from it stay local (`reports/video/` ignored); only echo's motion and our renders are published |
| NLF, Neural Localizer Fields, github.com/isarandi/nlf | the body pose in each video frame | code MIT, models non-commercial research | none in the repo (a tool); note the research-use models before any commercial use |
| HumanoidPose (arXiv 2610.25288), GVHMR, PromptHMR, VideoMimic, GMR's video path | surveyed for video-to-robot motion; GVHMR and PromptHMR need the SMPL files | various | none: consulted |
| Poly Haven, polyhaven.com | nine texture photos for the living room | CC0 | none (listed in `scenes/living_room/cc0/SOURCES.md`) |

## Software

| tool | used for | licence |
|---|---|---|
| Autodesk Fusion 360 and its MCP server | the CAD, built by script (`cad/fusion/`), saved as `cad/echo.f3d` and exported as STEP and STL | commercial; check that your Fusion licence covers your use (the personal licence is non-commercial) |
| MuJoCo | physics (stand and squat test), offline renders, videos | Apache-2.0 |
| manifold3d, trimesh | solids, booleans and collision volumes for the checks | Apache-2.0, MIT |
| numpy, SciPy, Pillow, imageio, matplotlib | maths, interpolation, pictures, video | BSD, MIT-CMU, BSD-2, PSF-based |
| three.js | the private web review page (loaded from a CDN, not in the repo) | MIT |

## What the licences required of echo

1. GMR's MIT notice ships with the repo (`THIRD_PARTY_NOTICES.md`), and the adapted files name GMR.
2. No Asimov, Unitree or AgiBot file is in the repo, so their licences place no conditions on it. Copying any of them
   would: CERN-OHL-S and GPL require the derived work to carry the same licence.
3. Figure: the specs are facts, but the frames are someone else's copyright and the names are trademarks. So: no
   frames in the repo, our own logo, and a no-affiliation statement.
4. Libraries are dependencies, not redistributed: no notices are needed. Vendoring one would bring its licence
   file along.

echo itself is licensed under the Apache License 2.0 (`LICENSE`), chosen so anyone can use it. That covers the code,
the spec, the CAD, the robot models, the documents and the media, and Apache-2.0 also grants a patent licence from
contributors. If you build on echo, keep `LICENSE` and `THIRD_PARTY_NOTICES.md` and say what you changed. For your
own robot, pick a licence early: without one, others can read your repo but have no right to reuse it.
