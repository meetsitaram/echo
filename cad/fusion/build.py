"""Build echo in Fusion from spec/robot.json (the scripts are the source of truth: never hand-edit
the document; change spec/robot.py, rerun the checks, rebuild the link).

One short job per MCP call (long calls time out and freeze Fusion; CDX-3R lessons 78-79):
  build.new_document()          a new Z-up design document
  build.link('pelvis')          (re)build one link as its own component, at the zero pose
  build.joints()                as-built revolute joints at the spec origins and limits
  build.status()                components, body counts, unhealthy features, duplicates
  build.pose({'L_elbow': -1})   set joint angles (rad); missing joints go to 0
  build.show(skins=False)       hide the skins to see the internals (and back)
  build.export_stl(folder)      one STL per link at the zero pose, for the URDF and the mesh checks

Frame: Fusion world = spec world (Z up, X forward, Y left); the API works in cm, the spec in m (figlib converts).
"""
import json
import os
import time

import adsk.core
import adsk.fusion

import figlib as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC = 'echo'
GLOSS, ALU, LENS = '#08090B', '#9EA3A8', '#0A1420'


def app():
    return adsk.core.Application.get()


def design():
    return adsk.fusion.Design.cast(app().activeProduct)


def spec():
    return json.load(open(os.path.join(REPO, 'spec', 'robot.json'), encoding='utf-8'))


def finish(g):
    return 'gloss' if g['color'] in (GLOSS, LENS) else 'metal' if g['color'] == ALU else 'matte'


def occurrences():
    return {o.component.name: o for o in design().rootComponent.occurrences}


def new_document():
    doc = app().documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    d = design()
    d.designType = adsk.fusion.DesignTypes.ParametricDesignType   # the root takes the document's name on save
    return f'new document, orientation {app().preferences.generalPreferences.defaultModelingOrientation} (1 = Z up)'


def link(name):
    t = time.time()
    s = spec()
    d = design()
    root = d.rootComponent
    for o in list(root.occurrences):                     # rebuild: remove the old copy first (no "name (1)")
        if o.component.name == name or o.component.name.startswith(name + ' ('):
            o.deleteMe()
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component
    comp.name = name
    b = F.Builder(comp)
    geoms = s['links'][name]['geoms']
    for g in geoms:
        if g['role'] == 'cut':
            continue
        body = b.build(g)
        body.name = g['name']
        body.appearance = F.appearance(d, g['color'], finish(g))
    for g in geoms:                                      # cuts last: windows through the bodies built above
        if g['role'] == 'cut':
            b.cut(b.build(g), [comp.bRepBodies.itemByName(n) for n in g['targets']], g['name'])
    if name == s['root']:
        occ.isGrounded = True
    else:
        occ.isGroundToParent = False                     # new occurrences come in grounded (CDX-3R lesson 24)
    return f'{name}: {comp.bRepBodies.count} bodies ({len(parts(geoms))} in spec), {time.time() - t:.1f} s'


def parts(geoms):
    """The geoms that are bodies ('cut' geoms are removed from others and leave none)."""
    return [g for g in geoms if g['role'] != 'cut']


def links(names):
    return [link(n) for n in names]


def _joint_sketch(root):
    for i in range(root.sketches.count):
        if root.sketches.item(i).name == 'joint points':
            return root.sketches.item(i)
    sk = root.sketches.add(root.xYConstructionPlane)
    sk.name = 'joint points'
    sk.isVisible = False
    return sk


def joints(only=None):
    t = time.time()
    s = spec()
    root = design().rootComponent
    occ = occurrences()
    sk = _joint_sketch(root)
    dirs = {'x': adsk.fusion.JointDirections.XAxisJointDirection,
            'y': adsk.fusion.JointDirections.YAxisJointDirection,
            'z': adsk.fusion.JointDirections.ZAxisJointDirection}
    out = []
    for j in s['joints']:
        if only and j['name'] not in only:
            continue
        old = root.asBuiltJoints.itemByName(j['name'])
        if old:
            old.deleteMe()
        sp = sk.sketchPoints.add(sk.modelToSketchSpace(F.P(j['origin'])))
        geo = adsk.fusion.JointGeometry.createByPoint(sp)
        ji = root.asBuiltJoints.createInput(occ[j['child']], occ[j['parent']], geo)
        ji.setAsRevoluteJointMotion(dirs[j['axis']])
        jt = root.asBuiltJoints.add(ji)
        jt.name = j['name']
        lim = adsk.fusion.RevoluteJointMotion.cast(jt.jointMotion).rotationLimits
        lim.isMinimumValueEnabled = True
        lim.minimumValue = j['lower']
        lim.isMaximumValueEnabled = True
        lim.maximumValue = j['upper']
        out.append(j['name'])
    return f'{len(out)} joints, {time.time() - t:.1f} s'


def pose(q):
    root = design().rootComponent
    for i in range(root.asBuiltJoints.count):
        jt = root.asBuiltJoints.item(i)
        adsk.fusion.RevoluteJointMotion.cast(jt.jointMotion).rotationValue = float(q.get(jt.name, 0.0))
    return f'posed {len(q)} joints'


def show(skins=True, internals=True):
    s = spec()
    role = {g['name']: g['role'] for l in s['links'].values() for g in l['geoms']}
    exposed = {g['name'] for l in s['links'].values() for g in l['geoms'] if g.get('exposed')}
    for j in s['joints']:
        if j['act'] and j['exposed']:
            exposed |= {f"{j['name']} flange"} if j['exposed'] == 'flange' else                 {f"{j['name']} actuator (buy) {j['act']}", f"{j['name']} flange"}
    n = 0
    for o in design().rootComponent.occurrences:
        for b in o.component.bRepBodies:
            b.isLightBulbOn = skins if role.get(b.name) in ('shell', 'decal') else (internals or b.name in exposed)
            n += 1
    return f'{n} bodies: skins {"on" if skins else "off"}, internals {"on" if internals else "off"}'


def status():
    s = spec()
    d = design()
    occ = occurrences()
    rows, problems = [], []
    for name, l in s['links'].items():
        o = occ.get(name)
        if not o:
            problems.append(f'missing {name}')
            continue
        nb = o.component.bRepBodies.count
        if nb != len(parts(l['geoms'])):
            problems.append(f'{name}: {nb} bodies, spec has {len(parts(l["geoms"]))}')
        rows.append(f'{name}:{nb}')
    extra = [n for n in occ if n not in s['links']]
    if extra:
        problems.append(f'extra components {extra}')
    tl = d.timeline
    for i in range(tl.count):
        it = tl.item(i)
        if it.healthState != adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
            problems.append(f'unhealthy: {it.name} {it.errorOrWarningMessage}')
    nj = d.rootComponent.asBuiltJoints.count
    return {'links': len(rows), 'joints': nj, 'timeline': tl.count, 'problems': problems[:30]}


EXTENT_PER_CM = 1.0       # orthographic viewExtents per cm of visible height (calibrated by tools/compare.py)
VIEWS = {
    # name: (azimuth deg from the robot's front, towards its left; elevation deg), as the reference photos
    # '34' at 60 deg: the reveal frame's silhouette widths fit best there (rms 3 cm, 2026-10-31); at 40 deg every
    # width was 10 cm too wide and the legs read heavy
    'front': (0, 0), 'side': (90, 0), 'back': (180, 0), '34': (60, 8), 'hero': (40, 8),
}


def render(path, view='side', w=800, h=600, height_cm=190.0, target_z_cm=86.0):
    """Orthographic picture of the whole robot from one of VIEWS, saved as PNG (for tools/compare.py)."""
    import math
    az, el = (math.radians(v) for v in VIEWS[view])
    vp = app().activeViewport
    vp.visualStyle = adsk.core.VisualStyles.ShadedVisualStyle   # no edge lines: they made the model look busy
    cam = vp.camera
    cam.cameraType = adsk.core.CameraTypes.OrthographicCameraType
    cam.isSmoothTransition = False
    t = adsk.core.Point3D.create(0, 0, target_z_cm)
    d = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))
    cam.target = t
    cam.eye = adsk.core.Point3D.create(600 * d[0], 600 * d[1], target_z_cm + 600 * d[2])
    cam.upVector = adsk.core.Vector3D.create(0, 0, 1)
    cam.viewExtents = height_cm * EXTENT_PER_CM
    vp.camera = cam
    vp.refresh()
    root = design().rootComponent
    root.isJointsFolderLightBulbOn = False
    root.isSketchFolderLightBulbOn = False
    root.isConstructionFolderLightBulbOn = False
    if os.path.exists(path):              # so 'written' means this save wrote it (an old file fooled the check)
        os.remove(path)
    ok = vp.saveAsImageFile(path, w, h)
    written = os.path.exists(path)
    src = os.path.join(os.path.dirname(path), 'replica_source.txt')   # tools/render_pose.py's label for compare.py
    if written and os.path.basename(path).startswith('replica_') and os.path.exists(src):
        os.remove(src)
    return f'{view}: {path} ({ok}, written: {written})'


def render_posed(q, shots):
    """render() for each (path, view, w, h, height_cm, target_z_cm) in shots, in pose q. While a moved joint's
    position is uncaptured, saveAsImageFile returns True and writes nothing (2026-10-31): capture it (a snapshot),
    render, then delete the snapshot and go back to the zero pose."""
    d = design()
    pose(q)
    snap = d.snapshots.add()
    out = [render(*s) for s in shots]
    snap.deleteMe()
    pose({})
    return out + [f'pending position after: {d.snapshots.hasPendingSnapshot}']


def export_stl(folder):
    t = time.time()
    os.makedirs(folder, exist_ok=True)
    em = design().exportManager
    out = []
    for name, o in occurrences().items():
        path = os.path.join(folder, f'{name}.stl')
        opt = em.createSTLExportOptions(o, path)
        opt.isBinaryFormat = True
        opt.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
        em.execute(opt)
        out.append(name)
    return f'{len(out)} STLs in {folder}, {time.time() - t:.1f} s'
