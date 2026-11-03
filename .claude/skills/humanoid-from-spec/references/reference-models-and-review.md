# Matching a reference robot, and working through review

This method was used to build a second robot, a small look-alike on hobby servos, after echo. Its user needed about
twenty corrections across eleven review rounds in one day, nearly all because the agent designed from reasoning and
eyeballed pictures instead of measuring and copying the reference. The user should not have to do that work. Follow
this file for any look-alike, and whenever the reference robot has a model you can open.

## 1. Use the reference's own model when one exists

Many robots publish a URDF or MJCF with meshes, often in their simulation or RL repo. It holds what pictures hide:
joint positions, axes and ranges, which body each actuator rides in, where the boards and battery sit, every outline,
and named poses (keyframes). Get it before drawing anything.

- **Check the licence.** Code is often permissive while meshes are not (CC BY-NC-SA is common). Measure, never copy
  geometry or meshes, keep the model and its renders outside the repo, and list it in your sources.
- **Read the joints first.** List every joint's position, axis and range, and every actuator case with its body and
  orientation. Lay out your joints the same way, then fit your actuators into that layout. Eight rounds of layout by
  reasoning were replaced by one round of copying.
- **Take its zero pose as yours.** If its model stands bent at zero (a bird-like leg, say), build yours in that
  stance: then every actuator case is square to the axes there, and the targets are measured in the same pose.
- **Look up where it keeps its parts** (computer, battery, camera) before packing yours. Guessing put a board in the
  body that the reference keeps in its head, and the body came out 20 % too long.

## 2. Measure, don't draw

- Slice the reference's meshes on a grid (5 mm steps): the side profile, the front outline at several heights, the
  plan width along its length, layer thicknesses (a sole and its upper, a band and its lip). Scale by your height over
  the reference's, and build every visible shape from the numbers.
- Compare close-ups at one scale on a 10 mm grid, region by region (head, thigh, foot). The user asked "did you even
  add a grid?" after shapes drawn by eye kept missing.
- When an outline is still "off angled", take it from the mesh: transform the mesh's vertices into your link's frame
  (undo the reference's stance), divide by the link length, and fit the hull. A thigh cover came out as the hull of
  three equal circles at the two pivots and one corner.
- When your robot cannot be the reference's size (bigger actuators), compare proportions: each landmark scaled by the
  height ratio, with a tolerance, measured in the reference's stance; report the height ratio honestly.
- Look for motion in the maker's launch video, not only the photo gallery: a moving mouth, a fold, a pick-up were only
  in the video. Keep the clip and frames local.

## 3. Validate by moving both robots the same way

Apply the reference's own named poses and single-joint moves to both robots, with the sign of each joint taken from
the two joints' world axes at the zero pose. Plant one foot, and compare where the joints land, the reference scaled
to your size. Render both side by side for each pose.

- Compare joint positions (points), not link frames: if every landmark moves by the same amount, the measurement is
  reading transforms, not positions.
- This found ranges no still picture showed: a fold that needs 90 deg at the hip and knee, a neck that leans back
  60 deg, a roll joint on the wrong side of the yaw axis.

## 4. Structure that holds

- **Hold every joint on both sides.** A joint held only by its output plate flexes and wears its bearing. Use a
  U-bracket: the horn or flange on one side, an idler bearing on the other, bridged across.
- **Map the actuator case's sweep before placing a bracket near it.** An off-centre box sweeps a long reach on one
  side and a short one on the other; over a joint's range its long end visits only some angles. Put bridges in the
  sector it never visits, or outside its reach.
- **Height comes from stacking.** Look for axes the reference puts at one height and actuators it lays flat or stands
  beside each other; each stacked actuator costs its full length.
- **Covers over moving parts need slots or cuts**, and mirrored cuts must be mirrored with their targets.
- **Re-run the checks after any scale change.** Shrinking a part left a bracket with crossed ends (negative size),
  which hung a sampling check for ten minutes.
- **Static one-leg balance needs an ankle roll.** Without one, the gait is dynamic: check single-support poses for
  torque and collisions, and report balance as a warning.

## 5. Working through review

- The user reviews by circling screenshots. Take every circle literally, find the cause in the geometry, fix it, and
  send the same views back. Answer each mid-turn message in a line before continuing.
- Look at every render yourself before sending it. Several rounds went to the user with faults visible in the picture.
- When the user works remotely, send pictures as files: they cannot open local paths.
- Ask only real decisions (size trade-offs, actuator choices, CAD timing). Decide the rest, do it, and report the gap.
- Explain early that the spec is the CAD source and Fusion builds once the look is approved; otherwise the user will
  ask why there is no CAD. Build after the look and the pose validation pass.
- Write each correction into `LESSONS.md` the round it happens: the user's words, the cause, the fix.

## 6. Spec tools that helped (in the second robot's spec)

- Actuator cases as off-centre boxes, the horn on the axis; a flag to put the case in the child link (an actuator
  that rides with the moving link), and one to put it off the axis, driving the joint through a rod.
- Flat side plates from the convex hull of circles (covers, tabs, ears).
- Lofts stacked along x, for shapes wider at the front than the back; a leaning cut plane for a raked face.
- A comparison that renders the reference's own model beside yours at one scale, with landmark lines and grid
  close-ups; a pose validator over the reference's keyframes.

## 7. Practical notes

- Shell heredocs with nested quotes break in Git Bash on Windows: write patch scripts to a short temp path.
- Python cannot open very long paths on Windows: keep scratch files on short paths.
- X videos: the syndication endpoint lists the mp4; OpenCV reads frames when ffmpeg is missing.
