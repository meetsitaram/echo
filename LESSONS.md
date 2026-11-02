# Lessons learned: echo (a Figure 03 replica)

Builds on CDX-3R's `cad/fusion/LESSONS.md` and `docs/PIPELINE-PLAYBOOK.md` (C:\Users\PC\projects\cdx-3r): read those
first. Below: what went wrong in this project, and the rule or test that now prevents it. If you are building your
own humanoid this way, start with `.claude/skills/humanoid-from-spec/SKILL.md` (the method) and keep this file open beside it
(the traps).

## Process

| Issue | Do instead |
|---|---|
| Packaging a 61 kg humanoid by hand in Fusion would have meant many slow rebuilds | Lay out and check offline first (`tools/checks.py`, ~90 s for 296 checks), build in Fusion once the spec passes (CDX-3R lesson 75, applied to a whole robot) |
| A check passed a design it should have failed: deleting the hip bracket was not caught, because the cosmetic hip lobe "connected" the parts | Run every new check on a deliberately broken design first (`checks.py --bad <case>`). Covers carry no load: connectivity is checked on rigid parts alone, and covers separately must be mounted |
| The skin check let a 5 mm actuator corner through the calf (0.3 % of its volume) because the limit was 5 %; it showed as dark marks in Fusion | Limit 0.5 %; look at the Fusion render after every build (numbers hide ugly geometry) |
| The first proportions came from a low-resolution frame: head 4 cm too tall, arm 4-8 cm too low | Measure on the largest frames available (enlarge the video in the page, pause on exact times), same method on both pictures, and draw the targets as lines across both (`tools/compare.py`) |
| An automatic black-part detector on the photos and renders merged elbow, hand and pelvis into one blob | Measure the replica from its own geometry; use detection only for the reference |
| Every proportion change meant editing dozens of absolute coordinates | Anchor heights at the top of `spec/robot.py` (hip, knee, ankle, shoulder, elbow, wrist, hem, neck, chin); every part placed relative to them. Re-proportioning then passed all checks first time |
| All landmarks were in tolerance, yet the user saw a boxy "butt", a busy arm and a head that read as a bean | Shape is not landmarks: compare silhouettes in close-up crops at the same scale (`compare.py` detail_*), and render shaded without edge lines (every body edge drawn black made the model look like a kit of parts) |
| The head was modelled solid black for three iterations; Figure's is a grey knit hood with a black visor, visible only in a close-up of the side view | Zoom into each region of the reference before modelling it; the lowest-resolution look decides nothing |
| Pieces met with notches (pod above a straight thigh, a neck post under a round head) | Shape neighbours as one silhouette: the thigh's back continues the hip's back line, the neck's top section sits inside the hood so the nape is one curve |
| Our 3/4 render was at 40 degrees, Figure's 3/4 frame at about 60: every width read 10 cm too wide and the legs heavy | Fit the reference camera before comparing: project the model at several angles and pick the one whose silhouette widths match the frame (60 deg, rms 3 cm) |
| "The upper body looks skinnier than the lower": the chest measured right; the side frame showed arms 0.096 deep vs Figure's 0.11 and calves 0.135 vs 0.105, and the torso did not taper to the waist | Measure each section in the view it is seen in (arm depth from the side, width from the front) before changing anything; fix proportions, not only the part pointed at |
| The front frame (shot from above) put Figure's shoulders ~4 cm higher than the side and 3/4 frames do | Take heights from level views only; a raised camera lifts every top surface |
| Feet: a flat plate under a slipper, then a tall block with a flat top ("so off") | Zoom on the reference's feet first: Figure's is a low sneaker (outsole with toe spring, white upper rising from the toe, a stripe to the heel over a white window, angled collar). Model the parts the ankle moves on the shank (sock-shoe), the rest on the foot |
| A toe spring lifts the sole's front off the floor | The contact box (balance check, URDF, MuJoCo) covers only the flat part that stands on the floor |
| Figure's neck looked slimmer than ours by eye (0.07 m deep); the side frame measured 0.117 at its narrowest | Measure the frame (`compare.robot_box` and a row scan of the silhouette) before resizing a part |
| The link masses were hand budgets that summed to 61 kg exactly with the battery as the balance, and each budget was spread over the covers as if they were solid | Masses bottom-up from materials (`tools/mass_model.py`): cover walls, frame fill factors, catalogue actuators, explicit masses for the battery, compute and hands. Report the gap to the target (56.3 vs 61 kg) instead of padding the model to it |
| A per-link rule (frame at least 15 % of the actuators' mass) failed a 2 cm neck post that carries a 0.3 kg actuator easily | A check needs a physical reason: a mass ratio says nothing about strength |
| A `--bad` run of the mass check rewrote `reports/mass.md` with the broken design | Checks write only `reports/checks.md`; each tool writes its own report |
| A measured target read the wrong shape: the black-hip height took the hip lobe before its cuts, and its lowest point anywhere (the domed seam drops at the back) | Measure the replica where the photo was measured (the lobe's underside in its front half, as seen in the front and side frames) and on the finished shape (with its cuts) |
| A redesign done blind: the first hip fix gave the lobe a fin, a slot and a see-through seam, all visible only in pictures | After every geometry change: the checks, then close-up renders from the front, side, back and the angle the user looked from, before and after side by side; then the walk replay |
| The user looked at the hip from below in the web viewer, an angle none of our standard views had | Render the angle the user reports from (a low front view here) and keep it in the before/after set |
| Pictures and numbers in the repo drifted from the design: early renders of an old robot sat in `reports/`, and the README and the viewer quoted last round's mass | Before each commit, re-render every committed picture from the current spec (`render_pose.py`, `preview.py`, the README video) and copy the quoted numbers from the reports (`reports/mass.md`, `reports/checks.md`) |

## Mechanical design

| Issue | Do instead |
|---|---|
| A roll/pitch waist (A3, H2) cannot hide under a one-piece fabric torso: 20 degrees of tilt needs ~5 cm of hem clearance | Yaw-only waist (Asimov 1); keep Figure's look with a black waist column that turns inside the pelvis block |
| A closed black skirt over the hips: the hip roll brackets swing through ~16 cm radius as the thigh flexes | Covers that ride on the moving link (hip lobes first on the hip pitch link, since iteration 21 on the hip roll link: see below); a pelvis block narrow and high enough for 97 degrees of hip flexion (its underside is Figure's arch between the legs) |
| Hip roll bracket ran through the hip pitch actuator | Roll actuator behind the pitch actuator driving backwards; bracket wraps behind and under |
| ...which left a bracket corner behind and below the roll actuator, so the hip cover had to be a box | Drive the roll FORWARD: its output plate runs down the 9 mm gap between the two actuators (it turns in its own plane, so never meets the pitch actuator); the pitch-side bracket goes over the top (the output plate only swings below the axis). Nothing behind the roll actuator, so the cover is a round pod |
| The elbow's output plate showed as a black rectangle on the outside of the arm | Put outputs on the inner face where the outside should read clean (elbow) |
| Knee flexion stopped at 92 degrees: a thigh frame behind the knee met the calf's ankle actuators | Put thigh structure in front, beside the knee pushrod |
| Shoe collar met the ankle fork at 0.67 rad | Low-cut collar below the ankle axis; the black ankle shows above the shoe, as on Figure |
| ...but a rigid shoe cannot rise in front of the ankle: at 0.9 rad of dorsiflexion the shin sweeps down to 8 cm there | Sock-shoe: the shoe's collar line drops from the heel (0.122) to the front of the ankle (0.09); the white upper above it and the angled black collar ride on the shank. Dorsiflexion 0.8 rad (the 90 degree squat uses 0.77). The sock flares toward the heel and sides, where the range leaves room |
| The round shoulder cap and round arms read as a drum on a stick | Arms oval like Figure's (0.11 deep, 0.09 wide); the cap as deep as the arm, starting at full radius inside the torso's side (no waist), the arm flaring up under it |
| Deeper forearms met the upper arm at 135 degrees of elbow flexion | Elbow limit 129 degrees |
| Every actuator's output needs a part of the child link on it | Auto-generated flange on the child's side of each actuator (0.6 x its radius: 0.75 poked out at the shoulder) |
| Static torque in two-foot stance was overstated at the ankle roll (one leg was made to hold the whole upper body's sideways moment) | Closed chain: each leg takes half the upper body as a force at its hip plus half the moment |
| 20 kg in the hands continuously needs 40 N m per elbow (RS03 rated 20) | RS04 elbows first; then (user's choice) RS06 elbows and shoulder rolls, 88 mm, to look like Figure's arms: 10 kg short carry, 3 kg continuous (4 kg needed 109 % of the RS06 rating). The smaller roll flange moved the roll bracket up to stay connected |
| Single-joint sweeps flagged a hanging arm meeting the hip as the waist turns | Collisions between links more than 3 joints apart depend on the pose: warn, do not fail; task poses check combined motion |
| Balancing a pose leaned its ankles past the joint limit (deep squat: 0.86 rad against 0.8) and only a collision gave it away | Poses check every joint against its limits after balancing. The deep squat now leans the torso 21 degrees, as a person does (less knee torque too) |
| Covers cut apart along one plane only touched: the solid union fell into 156 slivers ('covers mounted' failed) | Overlap cut pieces by 2 mm, hidden inside the bigger one; count only pieces over 0.01 cm3 (a mirrored foot left a zero-volume sliver) |
| A thin wall slipped between the sweep's 0.03 rad steps and showed only in a pose | Where covers pass close (the ankle), scan the joint every 0.01 rad per part pair |
| No Figure logo (trademark) | Our own mark: inverted triangles on the visor and chest |
| The covers either side of a joint were allowed to overlap (treated as fabric), so 13 joints could not actually move | Sweep every joint min to max with the covers across it too (only the actuator / flange contact exempt); give each joint room where Figure shows black: openings (torso for shoulder caps and neck, pelvis for the waist column), seams at the arm yaw joints, elbow and knee zones the covers stop short of, round about the joint axis wherever something turns in it |
| Folding joints (elbow, knee) with the axis in the middle of a thick limb fold only until the covers meet | Covers end short of the axis, on a slant at the knee (deeper behind); set the limit from the sweep (elbow 109, knee 123 deg) |
| Marking a joint 'exposed' to let its flange show in a seam also exempted its actuator from the skin check: the 'poke' calibration stopped being caught | `exposed='flange'` exposes the flange only; rerun every calibration after touching a check |
| Windows cut right through the shoe showed the ankle clevis plates as bars | Cut a pocket each side ('cut' geoms with targets, applied in Fusion as a combine and offline as a manifold difference); what a cut uncovers counts as showing in the skin check (calibration 'window') |
| Smaller ankle motors no longer touched the shank plates: the motors floated | The connect check caught it; clamp them between the plates |
| Push rods drawn as part of the shank would collide with the foot in the sweep | Role 'linkage': drawn, kept out of the collision checks (they move with the linkage, not with either link) |
| Shoulder roll: the arm's top sat inside the cap (round about the pitch axis), and the roll bracket left the cap through its wall; rolling the arm 160 deg swept both through the cap | Where two axes cross, give the cover a hemisphere about the crossing and cut the next link's cover as a socket round it (4 mm); take the output out ON the axis (a round hole in the cap's back never sweeps) and down behind the cap. A bracket through the cap's side needs a slot round half of it. Range from the sweep: 150 deg (past it the arm's outer top swept the cap's top) |
| Neck pitch at the jaw: the hood's underside crossed the pitch actuator and the neck at every angle, and nodding swung the chin into the neck | Pitch at the neck's base, on the yaw axis: the neck tilts with the head (one piece with the hood, as Figure's knit neck bends), its base a sphere zone about the axes' crossing; compute the torso's opening from the swept neck. The tilt range sets the collar ring: 34 deg needed 1.2-1.9 cm, 26 deg 9 mm |
| Knee covers that stopped short of the axis on slants showed a black block in a gap ("real knees with actuators won't look like this") | A hinge: a drum round about the axis on one link, between fork cheeks of the other; each cover ends in a concave socket 4 mm clear of the drum, a narrow seam on the outside of the fold (6 deg in front of the knee), an open notch on the inside (behind: 123 deg of flexion) |
| A clip plane at the knee axis never cut the thigh: its loft stopped 3 cm above the axis, and the knee showed a 3.7 cm black band in front | A clip only trims: run a part's sections past every plane meant to shape it, then look at the front view |
| Rigid covers at a fold (elbow, 109 deg): the forearm's front face sweeps 7 cm up the upper arm, and any frame above the elbow is in its way | Cups: both covers run on round the actuator, hollow round it, side walls kept. The child ends on a line through the axis; the parent closes the outside of the fold down to 6 deg from where flexion turns that line, and in front is cut to the child's front face at full flexion (4 mm clear). Carry the stator from the sector the child never reaches |
| The thigh hung under the hip lobe on a 12 mm plate and the yaw actuator's rim, 2.5 cm down ("that connection looks so weak for such a big robot"): the lobe rode on the pitch link, so the thigh rolled and yawed against it | Put the cover on the link before the last joint (the hip roll link): only that joint (yaw) moves the next cover against it, and a seam square to its axis never sweeps (3 mm). The cover then moves against the pelvis (roll): hollow it round the swept parts (convex hulls through the range, as prisms) and bevel it where the roll tips it toward the pelvis |
| Reordering the hip joints (yaw first, as Unitree H1) looked like the way to let the thigh cup the hip motor, as the knee does | Check where the new first actuator goes before reordering: yaw first puts a motor above each hip, where the torso's hem turns, and Figure's black hip band (0.86-1.0 m) has no room for one either. Kept pitch, roll, yaw (Asimov 1, G1, H2) and moved the cover to another link instead: same ranges, same kinematics, the recorded walk still valid |
| Widening the rolling lobe to clear the parts that turn inside it made it touch the hanging wrist, and its swept-volume cut broke through its outer surface in places | Move the actuator along its own axis (6 mm inboard: no kinematic change) before widening a cover; cut the swept volume in slices along the axis, each only as tall as the actuator is there (a cylinder is lower away from its centre line); find holes by sampling the uncut outer surface and testing the points against every cut |
| Domed, the thigh's top wall crossed the rim of the yaw actuator inside it (the 'covers' check caught it) | A clearance opening must clear the actuator at every height the curved wall passes: the opening through the domed top runs 1 cm deeper |
| White lines in the side view: the flat 3 mm hip seam lined up with the other hip's and showed the background through both, and the roll motor's 3 mm clearance cut through the lobe's slim back | Any flat seam is see-through seen level: curve it round the joint's axis (the thigh's top inside a sphere, `keep`; the lobe 3 mm outside it), and cap what a clearance cut reaches. Test with a green background: green inside the outline is light through a gap |
| A bevel plane along the whole lobe left a fin, seen from the front | Cut only over the length of what it must clear (the pelvis block); the full back keeps the round outline |
| Single-joint sweeps cannot see combined motions (a hip pitching, rolling and yawing at once) | Replay the recorded walk through the collision test (`tools/motion_collisions.py`, every 5th frame) before and after a change at a joint |
| An oval pad standing off a rounded arm read as a disc on a post (3/4 view) | Set the disc into a flat side (the section squarer there, n 4), the child's side wall wrapping it 3 mm clear |
| Knee pads inside the leg's width needed recesses that narrowed the shank's top | Pads outside both covers, domed, on mounts that turn in a small round hole on the axis |
| A small domed pad's sections were closer together than the wall is thick, so the skin's cavity (each end moved in by the wall) put sections out of order and the interpolation failed ('x must be strictly increasing') | When moving a loft's ends in for the cavity, drop the middle sections the moved ends pass (`geom.inner_of`) |
| A hip lobe's flat cut faces went through the hip motors and no check saw it: the sweep and skin checks take covers as 3 mm walls, open where they are cut | Check 'covers': each cover's skin closed on its flat cut faces (as Fusion draws them) against the motors of its own and the neighbouring links. Where a motor passes a cover's face, a round opening 3 mm clear of it, centred on the motor's axis |

## Motion (MuJoCo)

| Issue | Do instead |
|---|---|
| Adding bodies to an `MjSpec` that had been compiled once put the right toe on the ankle pitch link: the right ankle roll had a zero Jacobian column and never moved | Add to a fresh spec and compile once; assert each added body's parent |
| Kinematic playback with `mj_kinematics` alone left the directional light with a zero direction (a dark floor), and the MJCF's own spotlight sat over the origin | Call `mj_camlight` after `mj_kinematics`; light the viewing model on its own |
| Retargeted stance feet slid 9 cm/s and sank up to 3 cm into the floor: PICO's feet drift with no contact sensing | Lock planted feet in the IK targets (a second pass over the take) and ground the lowest sole every frame: 2 cm/s, no sinking |
| Motion from a video (2026-11-01): NLF's person detector missed the robot whenever it bent over, which is every pick-up (80 % of frames found) | The camera is fixed: find the robot by its difference from the empty room (the median of 60 frames) and pass that box to NLF as an extra detection: 99.6 % found, the bends included |
| A floor fitted to the planted feet came out tilted a degree or two; from a camera 1.3 m up, looking almost level, that put the back wall 10.6 m away on one side and 14 m on the other, and the robot grew taller with distance | Solve the horizon and the camera's height from the robot itself: in each upright frame its foot and head are at known heights, and the ratio of their distances below the horizon holds at any depth or focal length. The robot then measures 1.60-1.62 m (head joint) from 4 to 9 m |
| The camera's up vector had the wrong sign (tilting it up, not down): the robot came out 2.1 m tall | Derive it from the rule, not by eye: every ray on the horizon row is square to up, so u_z = (v_hz - c_y) when u_y = -f |
| The feet slid with the video's depth jitter: each frame NLF puts the whole body a little nearer or farther | Scale each frame about the camera (the picture is unchanged) until the lowest foot is on the floor, then pin planted feet by moving the root (`--pin`): 5 cm/s to 1 cm/s |
| The head and wrists took odd angles: a video cannot show a robot's visor direction or its wrist twist | Hold them straight (`--hold head,wrists`) and say so |
| The hips stop at 97 deg of flexion, so echo bends 65-77 deg where Figure folds over at every pick-up | Open: a hip redesign (more flexion, a slimmer rear lobe) |

## Scenes (MuJoCo)

| Issue | Do instead |
|---|---|
| A room laid out by eye put walls through the walking robot and the coffee table under its feet | Measure the room from the same video: floor points marked on the empty-room frame, traced to the floor through the calibrated camera, so the room and the motion share one frame. Then check the whole take for overlaps with `mj_geomDistance` |
| MuJoCo read no JPEG texture ("unsupported content type") | Keep sources as JPEG; write PNGs MuJoCo reads to an ignored build folder |
| A 2D texture on a box geom stretched one image over each face | Build textured boxes as meshes with texture coordinates in metres (`_box_mesh`), and rounded upholstery as superellipsoid meshes |
| Poly Haven's velvet is red and its linen blue; its walnut veneer is nearly grey | Keep the photo's detail (luminance about its mean) and recolour it to the room's colours |
| Poly Haven's API refused Python's default user agent (403) | Send a user agent (curl -A) |

## Fusion API (new here)

| Issue | Do instead |
|---|---|
| `rootComponent.name = ...` raises "root component name cannot be changed" | The root takes the document's name on save |
| Every loft showed a seam line down the front (face, torso, legs) | Start the closed splines' points at the back (t = pi) |
| Decals placed on a straight-line estimate of the face sat 2-3 mm deep | Place them on the loft surface interpolated the way the loft is (PCHIP), 0.6 mm proud; then measure in Fusion (TemporaryBRepManager difference) |
| Orthographic `camera.viewExtents` | Equals the visible height in cm (800 x 600 picture) |
| STL export units | millimetres; `make_robot_files.py` detects the scale against the spec |
| As-built joints from a sketch point | Joint axes line up with the world X/Y/Z; Fusion poses matched the offline kinematics within 1 mm |
| Rebuild time grows with the timeline (0.6 s per link in a new document, 10-15 s at ~650 features) | Rebuild into a fresh document when most links change |
| A visor cut from a head: removing bodies while iterating `splitBodyFeature.bodies` raised "Bad index" | Copy the collection to a list first; keep the piece whose centre of mass is in front of the plane. Offline the same cut is `Manifold.trim_by_plane` |
| `viewport.visualStyle` | `ShadedVisualStyle` for review renders (no edge lines) |
| A loft cut by several planes (shoe collar, a band between two lines) | `clip` takes a list of planes, applied in turn; a plane that misses the body makes `splitBodyFeatures.add` raise: keep the body |
| MCP calls time out (~60 s) while Fusion carries on: both feet in one call, all 29 joints in one call | One link per call when it has cut lofts (~45 s each); rebuild only the joints of the links rebuilt (`joints([...])`, 5 s); after a timeout, read `status()` before retrying |
| Past ~1,000 timeline features (repeated link rebuilds) every single-link call timed out, though the work still finished | Rebuild into a fresh document before the timeline passes ~900 |
| A full build alone is now ~1,800 features, and one hip lobe with 17 prism cuts took 5-10 minutes to rebuild (each cut is a sketch, an extrusion and a combine) | One link per call, then poll `status()` with read-only calls (they time out too while Fusion works; keep asking); do the offline work (renders, video, docs) meanwhile; keep cuts few and convex |
| `viewport.saveAsImageFile` returned True and wrote nothing: first while a moved joint's position was uncaptured, then for every save, even at rest (the MCP screenshot failed too) | `build.render` now reports whether the file was written; render poses offline from the spec's shapes (`tools/render_pose.py`, MuJoCo's renderer, same framing within 2 px) |
| ...and `build.render` said "written" because an older picture of the same name was there | Delete the target before saving, then check that the file exists |
| Renaming the document's appearances in one pass over `design.appearances` skipped some (renaming re-sorts the collection), so the STEP export still carried the old name | Collect the matching names first, rename, and repeat until none are left; then grep the export for the old name |
| `app.data.activeProject` raised 'InternalValidationError : group' with no project selected in the data panel | Find the project by name in `app.data.activeHub.dataProjects`; `document.saveAs(name, folder, description, tag)` into a folder made with `rootFolder.dataFolders.add`; check `doc.isSaved` and `dataFile.versionNumber` |
| The STEP's top-level product was named '(Unsaved)' while the document had never been saved | Save the document first (the root component takes its name), then export STEP and `.f3d` (`createFusionArchiveExportOptions`) |
| Fusion held about 10 GB with six full builds open, and the user needed the memory for training runs | Do the Fusion steps first (rebuild, save, export) and say when Fusion can be closed; build in one document, not a new one each round; if asked to terminate it, check the document is saved (`isModified` false) and stop the process |

## Environment (Windows)

| Issue | Do instead |
|---|---|
| `git clone` failed: "Current working directory has a path longer than allowed" (the session scratchpad path) | Clone into a short path (`C:\Users\PC\AppData\Local\Temp\figrefs`) |
| Python could not open files in the session scratchpad (the full path is over 260 characters) | Keep working files in a short path (`%TEMP%\echo`), not the scratchpad |
| Python's `zipfile` could not test the `.f3d` ('compression method not supported') | That is normal for Fusion archives: list the entries (`infolist`) to check the export, or open it in Fusion |
| `import mujoco` fails under `python -I` (it lives in the user site-packages, which `-I` skips) | `-I` for anything that reads downloaded files; plain `python` for our own sim on our own files |
| MuJoCo stand test rocked heel to toe, then leaned 5.5 degrees | Servo gains: kv = kp/6 (kp/25 was far under critical damping), kp = 8 x peak torque (servo give without gravity feed-forward). Both runs held the robot with no torque saturation |

## Repository and publishing

| Issue | Do instead |
|---|---|
| A video of someone else's robot is theirs: its frames and clips cannot be published, and X's terms forbid downloading outside its own features | Keep the clip local and ignored (`reports/video/`), fetch only the minutes needed (an HTTP range read, ffmpeg -ss over the URL), publish only echo's motion and our own renders, and say where the motion came from with no affiliation implied |
| The first push was rejected (GH007): the commits carried the owner's private e-mail | Author commits with the GitHub no-reply address (`<id>+<login>@users.noreply.github.com`); push with `GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never` so a missing credential fails instead of hanging |
| A public repo next to material we may not publish: the reference robot's video frames, comparison panels made from them, the operator's raw motion capture | Ignore them by pattern from the first commit (`reports/compare/figure_*`, `compare_*.png`, `detail_*.png`, `reports/motion/`, `reports/web/`); no reference logo (trademark): our own mark |
| A 20 MB Fusion archive was committed under the old name and renamed in the next commit, so every clone downloaded it twice | Settle names before committing large binaries. To remove one: rebuild the commits without it (`git commit-tree` on the same final tree, verify the trees identical and the file unreachable), keep a backup branch, and force-push with a lease (`--force-with-lease=main:<old>`) only with the owner's go-ahead (the permission system blocks it otherwise) |
| Renaming the robot and the repo (guna to echo) | Rename every file and mention in one commit (`git mv`, then a text replace over tracked files, then regenerate what bakes the name in: URDF, MJCF, STEP, videos), rename the repo through the GitHub API with the credential git already uses (never printed), then `git remote set-url`; GitHub redirects the old URL |
| We built on other people's work (the reference robot's specs and frames, open robots' layouts and URDFs, retargeting code from another project) without a running list of sources and licences, and had to trace each one back at the end | Keep a sources list from day one (`.claude/skills/humanoid-from-spec/references/sources.md`). Reading a design is not copying it: Asimov's, Unitree's and AgiBot's licences bind only if their files are copied. Adapted code keeps its notice (`THIRD_PARTY_NOTICES.md`, GMR's MIT notice) and names its source in the file. The project needs its own licence for others to reuse it: echo is Apache-2.0 (`LICENSE`) so anyone can |
| The user wanted to see changes before they went public | Publish the review page (the viewer with the new model, movies and renders), commit locally, and push only after the go-ahead |
