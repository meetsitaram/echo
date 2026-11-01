# Fusion 360 through MCP

Fusion is the CAD of record, but it is slow, and the MCP calls time out long before Fusion finishes. Work offline
until the spec passes, then build in short calls.

## Calling the build module

Each MCP script call imports the build module fresh:

```python
import sys, importlib
P = r'<repo>\cad\fusion'
if P not in sys.path:
    sys.path.insert(0, P)
import figlib, build
importlib.reload(figlib); importlib.reload(build)

def run(_context: str):
    print(build.link('L_hip_roll_link'))
```

| call | does |
|---|---|
| `build.new_document()` | a new Z-up design document (only for a full rebuild) |
| `build.link(name)` | deletes the link's component and rebuilds it from `spec/robot.json`: bodies, appearances, cuts |
| `build.joints([names])` | as-built revolute joints with limits; after rebuilding links, rebuild only their joints |
| `build.status()` | links, joints, timeline length, and problems (body counts against the spec, unhealthy features) |
| `build.pose({...})`, `build.show(skins=False)` | pose the model; hide the covers to see inside |
| `build.render(path, view)` | orthographic picture; reports whether the file was written |
| `build.export_stl(folder)` | one STL per link at the zero pose, for the URDF (millimetres; the writer rescales) |

## Timeouts and speed

- One link per call. A full build is about 1,800 timeline features, and in a big document one link with many cut
  prisms took 5-10 minutes.
- After a timeout Fusion carries on. Poll with read-only scripts (`"readOnly": true`) that print the link's bodies
  or `status()`. They time out too while Fusion is busy; keep asking.
- Do offline work while Fusion builds: renders, videos, docs.
- Keep cuts few and convex: each is a sketch, an extrusion and a combine.
- Rebuild only the changed links and their joints. Start a fresh document only when most links change.

## Save and export

- Save into the hub. Find the project by name in `app.data.activeHub.dataProjects` (`app.data.activeProject` can
  raise when no project is selected). Make or find a folder (`rootFolder.dataFolders.add('<name>')`), then
  `doc.saveAs(name, folder, description, '')`. Check `doc.isSaved` and `dataFile.versionNumber`.
- The root component takes the document's name only on save; before that the STEP's top product is
  "(Unsaved)". So save first, then export:
  - `exportManager.createSTEPExportOptions(path, rootComponent)` writes `cad/<name>.step`;
  - `createFusionArchiveExportOptions(path)` writes `cad/<name>.f3d`, the native design with its timeline and
    joints;
  - `export_stl` writes the URDF meshes.
- Renaming: rename appearances by collecting the matching names first, then renaming, and repeat until none are
  left. Renaming re-sorts the collection, so a single pass skips some. Then grep the STEP for the old name.

## Pictures

`viewport.saveAsImageFile` and the MCP screenshot can return success and write nothing (with the PC's screen asleep,
or a moved joint's position uncaptured). `build.render` deletes the target first and reports `written`. When Fusion
will not render, `tools/render_pose.py` draws the same shapes offline with the same framing.

## Memory

Fusion held about 10 GB with six full builds open. Build in one document, save it, do every export, and tell the
user when Fusion can be closed. Before stopping the process, check the document is saved (`isModified` false).
Unsaved "Untitled" documents are lost; Fusion offers to recover them on the next start.
