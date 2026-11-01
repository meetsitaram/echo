"""Fusion-side helpers: turn spec/robot.json shapes into Fusion features. Pure Python (no numpy inside Fusion).

Spec units are metres; the Fusion API uses centimetres. Convert only here (P, cm).
Lessons carried over from CDX-3R (cad/fusion/LESSONS.md there):
  - planes: offset the base plane along its own normal (dot with the base normal), never assume its sign
  - sketch points always through sketch.modelToSketchSpace
  - fillets can fail: check healthState and delete failed features so they do not break later ones
  - set body names and appearances after the feature exists
"""
import math

import adsk.core
import adsk.fusion

VI = adsk.core.ValueInput
NEW = adsk.fusion.FeatureOperations.NewBodyFeatureOperation
CM = 100.0
AX = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}
BASE = {'x': 'yZConstructionPlane', 'y': 'xZConstructionPlane', 'z': 'xYConstructionPlane'}


def P(p):
    return adsk.core.Point3D.create(p[0] * CM, p[1] * CM, p[2] * CM)


def coll(items):
    c = adsk.core.ObjectCollection.create()
    for i in items:
        c.add(i)
    return c


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def unit(a):
    n = math.sqrt(dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def add(a, b, s=1.0):
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


# ---------------------------------------------------------------- appearances

def appearance(design, hexcol, finish='matte'):
    app = adsk.core.Application.get()
    name = f'echo {finish} {hexcol}'
    a = design.appearances.itemByName(name)
    if a:
        return a
    lib = app.materialLibraries.itemByName('Fusion Appearance Library')
    want = {'gloss': 'Plastic - Glossy', 'matte': 'Plastic - Matte', 'metal': 'Aluminum - Satin'}[finish]
    base = None
    for i in range(lib.appearances.count):
        if lib.appearances.item(i).name.startswith(want):
            base = lib.appearances.item(i)
            break
    if base is None:
        base = next(lib.appearances.item(i) for i in range(lib.appearances.count)
                    if lib.appearances.item(i).name.startswith('Plastic'))
    a = design.appearances.addByCopy(base, name)
    h = hexcol.lstrip('#')
    rgb = adsk.core.Color.create(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    for pid in ('opaque_albedo', 'metal_f0', 'generic_diffuse'):
        prop = a.appearanceProperties.itemById(pid)
        if prop:
            adsk.core.ColorProperty.cast(prop).value = rgb
    return a


# ---------------------------------------------------------------- builder

class Builder:
    """Builds spec shapes as bodies of one component. Coordinates are world metres at the zero pose (the
    component's occurrence has an identity transform)."""

    def __init__(self, comp):
        self.comp = comp
        self.f = comp.features
        self._planes = {}
        self._refs = None

    # -- planes and sketches
    def plane_at(self, axis, d):
        """Plane normal to a world axis through the point d (metres) along it."""
        base = getattr(self.comp, BASE[axis])
        n = base.geometry.normal
        off = d * CM * dot(AX[axis], (n.x, n.y, n.z))
        key = (axis, round(off, 5))
        if key not in self._planes:
            if abs(off) < 1e-9:
                self._planes[key] = base
            else:
                ci = self.comp.constructionPlanes.createInput()
                ci.setByOffset(base, VI.createByReal(off))
                pl = self.comp.constructionPlanes.add(ci)
                pl.isLightBulbOn = False
                self._planes[key] = pl
        return self._planes[key]

    def plane_normal(self, o, ez):
        """A plane through o (metres) with normal ez, for tilted shapes (thumb)."""
        for ax, v in AX.items():
            if abs(abs(dot(ez, v)) - 1) < 1e-9:
                return self.plane_at(ax, dot(o, v))
        ex = unit(cross(ez, (0, 0, 1)) if abs(ez[2]) < 0.9 else cross(ez, (1, 0, 0)))
        ey = cross(ez, ex)
        if self._refs is None:
            self._refs = self.sketch(self.comp.xYConstructionPlane, 'refs')
        rs = self._refs
        pts = [rs.sketchPoints.add(rs.modelToSketchSpace(P(p))) for p in (o, add(o, ex, 0.02), add(o, ey, 0.02))]
        ci = self.comp.constructionPlanes.createInput()
        ci.setByThreePoints(*pts)
        pl = self.comp.constructionPlanes.add(ci)
        pl.isLightBulbOn = False
        return pl

    def sketch(self, plane, name):
        sk = self.comp.sketches.add(plane)
        sk.name = name
        sk.isVisible = False
        return sk

    def extrude(self, sk, profiles, normal, length, name):
        """Extrude profiles `length` metres along the world direction `normal` (sketch normal or opposite)."""
        ei = self.f.extrudeFeatures.createInput(coll(profiles), NEW)
        z = sk.transform.getAsCoordinateSystem()[3]
        d = (adsk.fusion.ExtentDirections.PositiveExtentDirection if dot((z.x, z.y, z.z), normal) > 0
             else adsk.fusion.ExtentDirections.NegativeExtentDirection)
        ei.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(VI.createByReal(length * CM)), d)
        f = self.f.extrudeFeatures.add(ei)
        f.name = name
        return f

    def fillet(self, body, r, name):
        """Round every edge of body; on failure try smaller, then give up and delete (never leave a broken
        feature in the timeline)."""
        for k in (1.0, 0.7, 0.45):
            edges = coll([body.edges.item(i) for i in range(body.edges.count)])
            fi = self.f.filletFeatures.createInput()
            try:
                fi.addConstantRadiusEdgeSet(edges, VI.createByReal(r * k * CM), True)
            except AttributeError:
                fi.edgeSetInputs.addConstantRadiusEdgeSet(edges, VI.createByReal(r * k * CM), True)
            try:
                ff = self.f.filletFeatures.add(fi)
            except RuntimeError:
                continue
            if ff.healthState == adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
                ff.name = name
                return ff.bodies.item(0)
            ff.deleteMe()
        return body

    # -- shapes
    def loft(self, g):
        li = self.f.loftFeatures.createInput(NEW)
        for s in g['sections']:
            pl = self.plane_at('z', s['z'])
            sk = self.sketch(pl, g['name'])
            pts = []
            m = 24
            for i in range(m):
                t = math.pi + 2 * math.pi * i / m        # start at the back: the spline seam shows as an edge
                c, si = math.cos(t), math.sin(t)
                x = s['cx'] + s['a'] * math.copysign(abs(c) ** (2 / s['n']), c)
                y = s['cy'] + s['b'] * math.copysign(abs(si) ** (2 / s['n']), si)
                pts.append(sk.modelToSketchSpace(P((x, y, s['z']))))
            sp = sk.sketchCurves.sketchFittedSplines.add(coll(pts))
            sp.isClosed = True
            li.loftSections.add(sk.profiles.item(0))
        li.isSolid = True
        f = self.f.loftFeatures.add(li)
        f.name = g['name']
        body = f.bodies.item(0)
        c = g.get('clip')
        planes = [] if not c else ([c] if isinstance(c, dict) else c)
        for k, cp in enumerate(planes):                  # every plane in turn
            body = self.clip(body, cp, f"{g['name']} cut{k + 1 if k else ''}")
            if cp.get('r'):
                body = self.round_rim(body, cp, [q for q in planes if q is not cp], cp['r'], f"{g['name']} rim")
        if g.get('keep'):
            body = self.keep(body, g['keep'], f"{g['name']} dome")
        return body

    def keep(self, body, ks, name):
        """Intersect body with the sphere ks = {'c', 'r'}: keep what is inside it (a domed top)."""
        tool = self.ellipsoid(dict(c=ks['c'], r=[ks['r']] * 3, name=name))
        ci = self.f.combineFeatures.createInput(body, coll([tool]))
        ci.operation = adsk.fusion.FeatureOperations.IntersectFeatureOperation
        ci.isKeepToolBodies = False
        cf = self.f.combineFeatures.add(ci)
        cf.name = name
        return cf.bodies.item(0)

    def round_rim(self, body, cp, others, r, name):
        """Fillet the rim that the clip plane cp left on body (not its edges on the other cut faces). Like
        fillet(): smaller radii on failure, then leave the rim sharp rather than keep a broken feature."""
        def on(face, q):
            if face.geometry.surfaceType != adsk.core.SurfaceTypes.PlaneSurfaceType:
                return False
            pl = adsk.core.Plane.cast(face.geometry)
            n = unit(tuple(q['n']))
            o = pl.origin
            return abs(abs(dot((pl.normal.x, pl.normal.y, pl.normal.z), n)) - 1) < 1e-6 and \
                abs(dot((o.x / CM - q['p'][0], o.y / CM - q['p'][1], o.z / CM - q['p'][2]), n)) < 1e-4
        for k in (1.0, 0.7, 0.45):
            edges = []
            for i in range(body.faces.count):
                face = body.faces.item(i)
                if not on(face, cp):
                    continue
                for e in [face.edges.item(j) for j in range(face.edges.count)]:
                    nb = [e.faces.item(j) for j in range(e.faces.count)]
                    if not any(on(f2, q) for f2 in nb for q in others):
                        edges.append(e)
            if not edges:
                return body
            fi = self.f.filletFeatures.createInput()
            try:
                fi.addConstantRadiusEdgeSet(coll(edges), VI.createByReal(r * k * CM), False)
            except AttributeError:
                fi.edgeSetInputs.addConstantRadiusEdgeSet(coll(edges), VI.createByReal(r * k * CM), False)
            try:
                ff = self.f.filletFeatures.add(fi)
            except RuntimeError:
                continue
            if ff.healthState == adsk.fusion.FeatureHealthStates.HealthyFeatureHealthState:
                ff.name = name
                return ff.bodies.item(0)
            ff.deleteMe()
        return body

    def cut(self, tool, targets, name):
        """Remove the tool body from each target body (a window through a shoe), then remove the tool."""
        for t in targets:
            if t is None:
                continue
            ci = self.f.combineFeatures.createInput(t, coll([tool]))
            ci.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
            ci.isKeepToolBodies = True
            self.f.combineFeatures.add(ci).name = name
        self.f.removeFeatures.add(tool)

    def clip(self, body, cp, name):
        """Split body with the plane cp = {'p', 'n'} and keep the part in front of it (the normal's side)."""
        p0, n = cp['p'], unit(tuple(cp['n']))
        side = lambda b: dot((b.physicalProperties.centerOfMass.x / CM - p0[0],
                              b.physicalProperties.centerOfMass.y / CM - p0[1],
                              b.physicalProperties.centerOfMass.z / CM - p0[2]), n)
        pl = self.plane_normal(p0, n)
        si = self.f.splitBodyFeatures.createInput(body, pl, True)
        try:
            sf = self.f.splitBodyFeatures.add(si)
        except RuntimeError:                     # the plane misses the body: all of it is on one side
            return body
        sf.name = name
        keep = None
        for b in [sf.bodies.item(i) for i in range(sf.bodies.count)]:     # copy: removing changes the collection
            if side(b) > 0:
                keep = b
            else:
                self.f.removeFeatures.add(b)
        return keep

    def ellipsoid(self, g):
        c, r = g['c'], g['r']
        sk = self.sketch(self.plane_at('z', c[2]), g['name'])
        to = lambda u, v: sk.modelToSketchSpace(P((c[0] + u, c[1] + v, c[2])))
        r0 = 0.05
        arc = sk.sketchCurves.sketchArcs.addByThreePoints(to(-r0, 0), to(0, r0), to(r0, 0))
        axis = sk.sketchCurves.sketchLines.addByTwoPoints(arc.startSketchPoint, arc.endSketchPoint)
        ri = self.f.revolveFeatures.createInput(sk.profiles.item(0), axis, NEW)
        ri.setAngleExtent(False, VI.createByString('360 deg'))
        body = self.f.revolveFeatures.add(ri).bodies.item(0)
        si = self.f.scaleFeatures.createInput(coll([body]), arc.centerSketchPoint, VI.createByReal(1.0))
        si.setToNonUniform(*[VI.createByReal(x / r0) for x in r])
        self.f.scaleFeatures.add(si).name = g['name']
        return self.comp.bRepBodies.item(self.comp.bRepBodies.count - 1)

    def cyl(self, g):
        """Axis 'x'/'y'/'z', or any direction (decals following a sloping surface)."""
        ax = AX[g['axis']] if isinstance(g['axis'], str) else unit(tuple(g['axis']))
        c = g['c']
        start = add(c, ax, -g['h'] / 2)
        pl = self.plane_at(g['axis'], dot(start, ax)) if isinstance(g['axis'], str) else self.plane_normal(start, ax)
        sk = self.sketch(pl, g['name'])
        sk.sketchCurves.sketchCircles.addByCenterRadius(sk.modelToSketchSpace(P(start)), g['r'] * CM)
        f = self.extrude(sk, [sk.profiles.item(0)], ax, g['h'], g['name'])
        return f.bodies.item(0)

    def rbox(self, g):
        c, s = g['c'], g['size']
        z0 = c[2] - s[2] / 2
        sk = self.sketch(self.plane_at('z', z0), g['name'])
        p0 = sk.modelToSketchSpace(P((c[0] - s[0] / 2, c[1] - s[1] / 2, z0)))
        p1 = sk.modelToSketchSpace(P((c[0] + s[0] / 2, c[1] + s[1] / 2, z0)))
        sk.sketchCurves.sketchLines.addTwoPointRectangle(p0, p1)
        body = self.extrude(sk, [sk.profiles.item(0)], AX['z'], s[2], g['name']).bodies.item(0)
        return self.fillet(body, g['fillet'], g['name'] + ' round') if g['fillet'] > 0.0005 else body

    def capsule(self, g):
        p0, p1 = g['p0'], g['p1']
        d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        L = math.sqrt(dot(d, d))
        ez = unit(d)
        start = add(p0, ez, -g['r'])                 # the round ends reach r past each end point
        sk = self.sketch(self.plane_normal(start, ez), g['name'])
        sk.sketchCurves.sketchCircles.addByCenterRadius(sk.modelToSketchSpace(P(start)), g['r'] * CM)
        body = self.extrude(sk, [sk.profiles.item(0)], ez, L + 2 * g['r'], g['name']).bodies.item(0)
        return self.fillet(body, g['r'] * 0.96, g['name'] + ' round')

    def revolve(self, g):
        """Solid of revolution about an axis-aligned line through g['c']; profile (t, r), t along the axis."""
        ax = AX[g['axis']]
        c = g['c']
        # a plane containing the axis: for a y axis, the plane x = c.x
        other = {'x': 'y', 'y': 'x', 'z': 'x'}[g['axis']]
        sk = self.sketch(self.plane_at(other, dot(c, AX[other])), g['name'])
        up = AX['z'] if g['axis'] != 'z' else AX['y']
        pt = lambda t, r: sk.modelToSketchSpace(P(add(add(c, ax, t), up, r)))
        prof = g['profile']
        inner = [pt(t, r) for t, r in prof[1:-1]]
        sp = sk.sketchCurves.sketchFittedSplines.add(coll([pt(*prof[0])] + inner + [pt(*prof[-1])]))
        axis = sk.sketchCurves.sketchLines.addByTwoPoints(sp.startSketchPoint, sp.endSketchPoint)
        ri = self.f.revolveFeatures.createInput(sk.profiles.item(0), axis, NEW)
        ri.setAngleExtent(False, VI.createByString('360 deg'))
        f = self.f.revolveFeatures.add(ri)
        f.name = g['name']
        return f.bodies.item(0)

    def prism(self, g):
        """A flat convex polygon in the plane x = x0 (points (y, z)), extruded t along +x."""
        sk = self.sketch(self.plane_at('x', g['x0']), g['name'])
        pts = [sk.modelToSketchSpace(P((g['x0'], y, z))) for y, z in g['pts']]
        lines = sk.sketchCurves.sketchLines
        first = prev = lines.addByTwoPoints(pts[0], pts[1])
        for p in pts[2:]:
            prev = lines.addByTwoPoints(prev.endSketchPoint, p)
        lines.addByTwoPoints(prev.endSketchPoint, first.startSketchPoint)
        return self.extrude(sk, [sk.profiles.item(0)], AX['x'], g['t'], g['name']).bodies.item(0)

    def build(self, g):
        body = getattr(self, {'loft': 'loft', 'ellipsoid': 'ellipsoid', 'cyl': 'cyl', 'rbox': 'rbox',
                              'capsule': 'capsule', 'revolve': 'revolve', 'prism': 'prism'}[g['kind']])(g)
        return body
