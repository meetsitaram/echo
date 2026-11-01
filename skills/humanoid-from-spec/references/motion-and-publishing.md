# Motion, review and publishing

## Retarget a human capture

echo walks a PICO full-body capture: XRoboToolkit's 24-joint body, recorded at 50 Hz.

```bash
python tools/retarget_pico.py <raw tape .npz> [--start S] [--dur D] [--out reports/motion/<name>.npz]
```

1. The human skeleton is morphed to the robot's segment lengths along the human's directions.
2. A two-stage mink IK follows it, warm-started from the previous frame: a frame task per body, and the elbow flex
   as a posture task. The IK table, `tools/ik/pico_to_echo.json`, is adapted from GMR, which is MIT-licensed: keep
   its notice.
3. Lock planted feet in a second pass, and ground the lowest sole every frame. Without this, PICO's feet drift with
   no contact sensing: echo's stance feet slid 9 cm/s and sank 3 cm, and slide 2 cm/s with no sinking after.
4. Calibrate the arm offsets per take, from a frame where the operator stands relaxed with the arms hanging.

The take holds time, joint angles in MuJoCo order, root pose, and the skeleton at robot scale. Keep raw captures
private: they record a person. Publish only the robot's motion, and only with the operator's approval.

## Play it back

```bash
python tools/render_motion.py reports/motion/<take>.npz --view 34 --start 0 --dur 25 --no-skeleton --out docs/media/<name>_walk.mp4
```

Playback is kinematic: the joint angles are set frame by frame and nothing is simulated. Say so in the captions.
Views are `34`, `side`, `front` and `follow`. `--no-skeleton` hides the operator, for anything public.
`--viewer` opens MuJoCo's interactive viewer.

For the README, use an animated WebP (768 px wide, every second frame, quality 62, about 1.8 MB for 25 s), with
the MP4 linked under it.

```bash
python tools/export_web.py reports/motion/<take>.npz
```

This writes `reports/web/<name>.json` (a glTF model, one mesh per part) and `walk.json` (per-frame link poses and
the skeleton) for a web viewer. echo's review page is a three.js page reading those two files.

## Check the motion

```bash
python -I tools/motion_collisions.py [take] [--step 5]
```

This catches combined joint motions that single-joint sweeps cannot see. Run it after any change at a joint, and
compare it with the previous design: a pair that was already there is the capture's, a new one is yours.

## Review before publishing

Put the new model, movies and renders on the review page. Send before/after pictures from the user's own angle.
Commit locally, and push only after the user says it looks right. A change the user saw first is a change the user
owns.

## Repository hygiene

- Ignore by pattern from the first commit:
  - the reference robot's frames and every panel made from them (`reports/compare/figure_*`, `compare_*.png`,
    `detail_*.png`);
  - raw captures and retargeted takes (`reports/motion/`);
  - viewer exports (`reports/web/`);
  - Fusion's raw STL folder (`urdf/meshes/fusion/`).
- No reference logos: use your own mark, and state that the project is not affiliated with the reference's maker.
- Author commits with the GitHub no-reply address if the account keeps its e-mail private; GitHub rejects pushes
  that expose it (GH007).
- Settle names before committing large binaries. A 20 MB `.f3d` committed under one name and renamed in the next
  commit stays in every clone twice. Removing it means rebuilding commits (`git commit-tree` on the same final tree)
  and a force-push with a lease, which needs the owner's go-ahead.
- Re-render every committed picture from the current spec before committing, and copy quoted numbers (mass, checks)
  from the reports, not from memory.

## Credits and licences

- List every source as you use it: target, layouts, URDFs, parts data, code, tools. See `sources.md` for echo's.
- Reading a design is not copying it. echo follows Asimov 1's layout and read its URDF, but contains none of its
  files, so its CERN-OHL-S and GPL licences place no conditions on echo. Copying their CAD or code would.
- Adapted code keeps its notice: `THIRD_PARTY_NOTICES.md` carries GMR's MIT notice.
- Choose the project's own licence early. Without one, others may read but not reuse. echo is Apache-2.0, so
  anyone can use it, commercially too, provided they keep the licence and notices.

## The README

Lead with the video and a views strip. Then:
- how it matches the target: a table of specs and how each was checked;
- what is inside: joints, actuators, ranges and masses;
- how joints get room;
- the checks table;
- the commands to run;
- the files table;
- what is next.

Keep `LESSONS.md` beside it, with one row per issue and the fix that now prevents it.
