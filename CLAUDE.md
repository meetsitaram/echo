# guna (a Figure 03 replica): notes for Claude

Read `LESSONS.md` first, then CDX-3R's `cad/fusion/LESSONS.md` and `docs/PIPELINE-PLAYBOOK.md`
(C:\Users\PC\projects\cdx-3r).

- **Frame and units:** metres; X forward, Y left, Z up, floor z = 0; zero pose standing, arms down, palms in. Fusion
  world = spec world (Fusion is set Z up); the Fusion API is in cm, converted only in `cad/fusion/figlib.py`.
  Fusion's "right" view looks at the robot's face.
- **Source of truth:** `spec/robot.py` (anchors at the top set the proportions) → `python -I spec/robot.py` →
  `spec/robot.json`. Never hand-edit robot.json or the Fusion document.
- **Every change:** `python -I tools/checks.py` (0 failing), then rebuild the changed links in Fusion
  (`build.link(name)`, then `build.joints()`), render, `python -I tools/compare.py`, look at the pictures.
  Masses come from the parts' materials (`python -I tools/mass_model.py`); never set a link mass by hand.
  A new check: prove it on a broken design (`--bad`) before trusting it. If Fusion saves no images
  (`build.render` reports `written: False`), `python tools/render_pose.py --views` renders the same views offline.
- **Fusion calls:** one short job per MCP call (a batch of links ~30 s max); scripts imported from
  `cad/fusion` with `importlib.reload`; check `build.status()` after builds. Rebuild in a fresh document when
  most links change (`build.new_document()`).
- **Actuators:** RobStride RS00/02/03/04/06 (`spec/actuators.json`, retailer numbers: confirm before ordering).
  Internal layout follows Asimov 1 (github.com/menloresearch/asimov-1, CERN-OHL-S-2.0).
- **References:** robot URDFs cloned to C:\Users\PC\AppData\Local\Temp\figrefs (Unitree H2/G1/H1-2, AgiBot A3/X2,
  Asimov 1); Figure 03 frames in `reports/compare/figure_*.jpg` (figure.ai reveal video).
- Don't save the Fusion document or commit unless the user asks.
