# Showcase video: the house style

How echo's showcase video is made, so the same style can be asked for again ("make the showcase video", or "same style as
docs/showcase_video.md"). Renderer: `tools/render_quality.py`. Physics: `train/replay_forces.py` with the trained
SuperMimic policy, in CPU MuJoCo.

## Format

- Portrait 1080 x 1920, 25 or 30 fps, H.264 yuv420p with faststart, under about 50 MB; decodes fully (`ffmpeg -f null`).
- Room: the living room (`tools/scene_living_room.py`, `--room living`), background only: no collisions, so motion is
  placed on open floor and echo never passes through furniture on screen.
- Lens: depth of field on in every room shot. Echo sharp, the far room gently blurred.
- Camera moves are smooth with ease in and out; no jumps. No music.

## Look

- Covered: knit grey fabric with fine lines and seams; gloss black hubs and pelvis; face as dark tinted glass reflecting
  the room, with a teal-blue glow behind it; soft key-light shadows and ambient occlusion in every crease.
- Cover off (skeleton): dark anodised motors and their housings, silver structure, screws and bearings visible.

## Storyboard (about 65 s)

| # | Section | What happens | Length |
|---|---|---|---|
| 1 | Face | Close-up; the light behind the face glass fades on at 1 s; slow orbit | ~3 s |
| 2 | Descent | Camera orbits down the body to the feet while echo does small moves: looks around, arms, folds hands, a small crouch at the legs | ~9 s |
| 3 | Ascent, covers off | Same path back up in skeleton view | ~5 s |
| 4 | Exploded view | Dolly out to the walk framing; every rigid part flies out into a cloud spread evenly through a ball (out to about 0.9 m, not all on its rim), with one or two motors drifting close past the camera, large and sharp, and back; then the covers fly in the same way until fully covered; force arrows fade in | ~8-10 s |
| 5 | Action | Short walk, crouch or low crouch, a small shove it rides out (300 N at the neck), a big one it doesn't (650 N), quick resets, the jog with shoves | ~25-30 s |
| 6 | End card | 'echo', 'Physics sim · Motion Data by Bones Studio (bones.studio)' | 2 s |

Sections 2, 3 and 5 are physics: every step simulated by the policy. Section 4 is posed and labelled 'Exploded view'.

## Captions

- On every motion section: 'Physics sim: every step simulated', 'Motion: BONES-SEED', the required credit
  'Motion Data by Bones Studio (bones.studio)', and 'Policy: SuperMimic, trained by us in MuJoCo Warp'.
- In the action: thick solid red arrows for contact forces from the solver, one per foot at its centre of pressure, depth-tested so the body hides them; the shove as a thick blue arrow with its N at the neck. No chart panel: the picture is full frame, with a small overlay readout of each foot's force (and 'Body on floor' when it happens).
- Framing in the action: the whole body head to feet with floor margin.
- Never shown: dataset clip names, Figure footage or drawings.

## Before rendering

- Run on the current model; say if the policy wasn't retrained on it.
- Any fall that isn't from a shove stays in, and is reported.
- Look at frames from every section at full size before sharing.
