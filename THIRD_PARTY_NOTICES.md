# Third-party notices

echo is licensed under the Apache License 2.0 (`LICENSE`). Its files were written for this project. The one exception
is the motion retargeting, part of which is adapted from GMR. GMR's licence asks that its notice travel with every
copy, so it is below; those parts stay under GMR's MIT terms as well.

## Adapted into echo: GMR, General Motion Retargeting (MIT licence)

- Source: https://github.com/YanjieZe/GMR
- Used in:
  - `tools/ik/pico_to_echo.json`: the IK table, derived from GMR's `xrobot_to_g1.json` (by way of the X2 table in
    github.com/meetsitaram/agibot-x2-record-and-replay).
  - `tools/retarget_pico.py`: the two-stage IK solve loop, after GMR's `motion_retarget.py`.
- Changes: echo's body names and reference bodies, toe bodies, echo's elbow range, offsets recalibrated per take,
  planted-foot locking and grounding.

> Copyright 2025 Yanjieze
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of
> the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
> THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The rest of the retargeter (capture-to-robot transform, segment morph, arm-plane frames, elbow posture task,
rest-frame calibration) is ported from the author's own scripts, in agibot-x2-record-and-replay and in the author's
fork of NVIDIA's GR00T-WholeBodyControl (github.com/meetsitaram/GR00T-WholeBodyControl). No NVIDIA-authored code is
included.

## Included: Poly Haven textures (CC0)

`scenes/living_room/cc0/` holds nine texture photos from Poly Haven (polyhaven.com), released under CC0 1.0 (public
domain). No conditions apply; they are listed with their sources in `scenes/living_room/cc0/SOURCES.md`.

## Training and retargeting: not included

| what | used for | licence | what it asks |
|---|---|---|---|
| BONES-SEED motion capture, Bones Studio (bones.studio) | the clips echo's walking policy trains on, and the motions in its videos | BONES-SEED Dataset License Agreement (bones.studio/info/seed-license) | Credit wherever results are shown: **Motion Data by Bones Studio**, linking to https://bones.studio/. If a trained model is shared, its README says: "Training data includes Motion Data by Bones Studio. Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement." Raw data and retargeted clips stay out of this repository |
| soma-retargeter, the author's fork (github.com/meetsitaram/soma-retargeter) | retargeting the clips to echo | Apache-2.0, with its dependencies' licences in its `licenses/` folder | Keep its licence with any copy |
| GR00T-WholeBodyControl, the author's fork (github.com/meetsitaram/GR00T-WholeBodyControl) | the SuperMimic training code | Source code Apache-2.0; NVIDIA's model weights under the NVIDIA Open Model License | Keep its licence with any copy. No NVIDIA weights are used in echo's policy, which is trained here |

## Used as tools, not included

| tool | used for | licence |
|---|---|---|
| NLF, Neural Localizer Fields, github.com/isarandi/nlf | reading the robot's body pose in each video frame (`tools/video_nlf.py`) | code MIT; models for non-commercial research use; neither is in this repository |
| PyTorch, torchvision, OpenCV | running NLF | their own licences |

## Consulted, not copied

These were read for comparison or as design references. None of their files, code, meshes or models are in echo, so
their licences place no conditions on it.

| reference | what we took from it | licence |
|---|---|---|
| Asimov 1, github.com/menloresearch/asimov-1 | the internal layout: one actuator per joint, hip pitch-roll-yaw, yaw-only waist, parallel ankle; joint orders and ranges read from its URDF | hardware CERN-OHL-S-2.0, software GPL-2.0 |
| Unitree G1, H1-2, H2 URDFs, github.com/unitreerobotics/unitree_ros | joint orders, offsets and ranges, for comparison | BSD-3-Clause |
| AgiBot A3, github.com/AgibotTech/A3-A3U-robot-model; AgiBot X2, github.com/AgibotTech/agibot_x2_urdf | joint orders, offsets and ranges, for comparison | Mulan PSL v2 |
| RobStride manuals and STEP models, github.com/RobStride/Product_Information | hole patterns and sizes in `spec/actuator_interfaces.json`; motor outlines measured into `spec/actuator_outlines.json`. The maker's files are not in this repository | no licence stated in the files we read |
| Figure 01 footage and photos, Figure's patent documents | how the mechanisms are arranged, in `reports/figure01_study.md`: our own measurements and descriptions. No frames or drawings are in this repository | their owners' copyright |

## Trademarks and the reference robot

Figure and Figure 03 are trademarks of Figure AI, Inc. echo is an independent project, neither affiliated with nor
endorsed by Figure AI. Figure 03's published specifications (figure.ai) and frames of its reveal video were used to
compare proportions; the frames are not in this repository, and echo uses its own mark, not Figure's logo. The
living-room motion was measured from Figure's public home-work video (x.com/Figure_robot, 2026-09-17); the video, its
frames and the poses read from it are not in this repository, only echo's retargeted motion and our own renders.

RobStride and its actuator model names, PICO and XRoboToolkit, Unitree, AgiBot, Asimov, Autodesk Fusion and MuJoCo
belong to their owners and are named only to say what was used.

## Libraries

echo's tools import numpy, SciPy, trimesh, manifold3d, MuJoCo, mink, Pillow, imageio and matplotlib, each under its own
licence. They are installed separately and not redistributed here.
