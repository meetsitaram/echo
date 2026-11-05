# Actuator interfaces: what we know and what to ask

Data: `spec/actuator_interfaces.json` and `spec/hardware_catalog.json`. Sources: RobStride's manuals (section 1.1 drawing,
section 2.3 connectors) and its STEP files of 2026-10-30. STEP was read with a small parser written for this job, not with
trimesh, cadquery or OCP. Hole patterns were checked back against a survey of the STEP: all 14 match.

## Confirmed by both drawing and STEP

| | Stator screws (bolt circle) | Hub screws (bolt circle) | Hub locating | Hub register | Outside |
|---|---|---|---|---|---|
| RS00 | front 6 M3 (50), rear 4 M3 (38) | 6 M3 (27) | 3 x 3 mm holes (24.46) | 35 | 57 dia |
| RS02 | front 9 M3 (73) | 6 M4 (24) | 3 x 4 mm holes (24) | 43 | 78.5 dia |
| RS03 | front 8 M4 (98) | 6 M4 (30.36) | 3 x 4 mm holes, 7 deep | 70 | 106 over ears |
| RS04 | both faces 10 M4 (106) | 9 M5 (36) | 3 x 6 mm pins, stand 3 proud (35.86) | 50 (STEP only) | 120 dia |
| RS06 | front 8 M3 (82) | 6 M4 (24) | 3 x 4 mm holes (24) | 52 and 37 | 88 over ears |

- **Clock angles:** every pattern is given relative to the connector, or to the cable tab on RS04. RS02's model sits off-origin.
- **RS04 is the other way up:** its output faces +Z in the STEP; the other four face -Z.
- **Hollow shafts:** none. A ray along each axis meets solid material, so cables cannot pass through the centre. The hub recesses are blind.
- **Cable exits** are radial, from the STEP, and inside the outer diameter on RS00, RS02, RS03 and RS06. Each has two
  openings 180 deg apart. RS04 differs: a 7.3 mm round stub leaves through a tab at 75 deg, with silicone seal notes on the drawing.
- **Connectors:** RS00, RS02, RS06 use XT30(2+2); RS03, RS04 use XT30 plus GH1.25 (manual 2.3).

## Drawing and STEP disagree on hub thread depth

| | Drawing | STEP bore | Design to, until answered (`spec/assumptions.json`, user's choice 2026-11-05) |
|---|---|---|---|
| RS04 9 M5 | 6.5 min | 3.5 | 6.5, assumed |
| RS06 6 M4 | 6 | 4.7 | 6, assumed |
| RS02 6 M4 | 7 min | 6.0 | 7, assumed |

These STEP bores have flat floors, and the models are shrink-wraps, so the wrap may have cut them short. It matters: if RS04's hub really is 3.5 mm, that's 0.7 x d against the 2 x d rule,
and at the knee a 6 mm screw engagement would fall to 3.
RS00 and RS03 hubs agree with their drawings (STEP is deeper).

## STEP only: do not rely on these yet

- Rear ear threads on RS03 (4, M4 size) and RS06 (4, M3 size); RS02's 6 through-holes to the rear. None is on a drawing.
- RS04's two 5 mm locating features at 3 and 183 deg: the tolerance reads as a pin, the model shows a hole.
- Bumps of unstated function: RS03 at 90 deg, RS06 at 180 deg.

## Retailer only

Nothing in the interface file. Retailer bearing figures were dropped: `hardware_catalog.json` uses maker and reference tables.
Some entries there are marked `search_excerpt`, because SKF's pages could not be opened directly. Recheck those on the SKF site.

## Missing (null in the files)

Bearing type and all ratings; screw grade and torque; thread class; flatness and run-out, on all five. Also: PA12 thread
engagement rule, open 61808, 61805 mass, M2 and M2.5 screw loads, crossed-roller moment ratings.

Without bearing ratings, a limb cannot rest on the hub alone. Until answered, assume the knee and elbow need the planned idler.

## Also noticed

- The RS spec book's tables list "Moment Inertia": RS00 0.001, RS02 0.0042, RS03 0.02, RS04 0.04, RS06 0.012 kg m2, matched
  to models by their order in the extracted text, so check the pages. `spec/actuators.json` says none is published. Not changed.
- The file in the repo's Chinese RS06 folder is the RS10P manual.

Open questions for the maker are kept in `robstride_questions.md` (draft, not sent).
