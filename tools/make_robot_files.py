"""Write the URDF and the MuJoCo model of the replica from spec/robot.json.

  python -I tools/make_robot_files.py [--meshes urdf/meshes]

Meshes: one STL per link at the zero pose in world coordinates. If urdf/meshes/fusion/<link>.stl exists (exported
from Fusion with build.export_stl) it is used, converted to metres; otherwise the link's spec shapes are meshed here.
Mass, centre of mass and inertia of each link come from its parts' materials (tools/mass_model.py, the model the
torque check uses); each joint's armature, damping and friction from the actuators that drive it (dynamics()).

Outputs: urdf/echo.urdf (ROS / Isaac), urdf/echo.xml (MuJoCo, floor, feet contacts,
position servos limited to each joint's peak torque).
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C  # noqa: E402
import geom as G  # noqa: E402
import mass_model as MM  # noqa: E402

OUT = os.path.join(G.ROOT, 'urdf')
NAME = 'echo'


def link_frames(R):
    """Each link's frame: its parent joint's origin (world-aligned). The root sits at the hip centre height."""
    f = {R.spec['root']: np.array([0.0, 0.0, 0.90])}
    for j in R.spec['joints']:
        f[j['child']] = np.array(j['origin'])
    return f


def link_mesh(spec, S, ln, fusion_dir):
    p = os.path.join(fusion_dir, f'{ln}.stl')
    if os.path.exists(p):
        m = trimesh.load(p, force='mesh')
        # Fusion writes STL in cm or mm depending on the document; detect from the spec extent
        ref = S.all[ln].bounding_box()
        span = max(ref[3] - ref[0], ref[4] - ref[1], ref[5] - ref[2])
        mspan = float(np.max(m.extents))
        scale = min((0.001, 0.01, 1.0), key=lambda s: abs(np.log(mspan * s / span)))
        m.apply_scale(scale)
        return m, 'fusion'
    mesh = S.all[ln].to_mesh()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts), 'spec'


def dynamics(spec, j):
    """(armature kg m2, damping N m s/rad, friction N m) at joint j: the driving modules' rotor inertia reflected
    through their gear and any linkage ratio (armature = n x rotor x (gear x ratio)^2), their viscous damping
    (x ratio^2) and Coulomb friction (x ratio). spec/actuators.json, estimates until the datasheets."""
    c = j['capacity']
    a = spec['actuators'][c['model']]
    n, k = c['count'], c['ratio']
    return n * a['rotor_kgm2'] * (a['gear'] * k) ** 2, n * a['damping_nms'] * k * k, n * a['friction_nm'] * k


def inertia_old(spec, S, ln):
    """(mass, com, 3x3 inertia about the com) in world coordinates at the zero pose: the hand-budget model
    before 2026-10-31 (actuators at their centres, the rest of the budget spread over the link's solids)."""
    A = spec['actuators']
    l = spec['links'][ln]
    parts = []                                           # (mass, com, inertia about own com)
    for g in l['geoms']:
        if g['role'] == 'actuator':
            m = A[g['name'].split()[-1]]['mass']
            r, h = g['r'], g['h']
            ia, ir = 0.5 * m * r * r, m * (3 * r * r + h * h) / 12
            I = np.diag([ir, ir, ir])
            k = 'xyz'.index(g['axis'])
            I[k, k] = ia
            parts.append((m, np.array(g['c']), I))
    rest = l['budget'] - sum(p[0] for p in parts)
    others = C.union([S.geom[(ln, i)] for i, g in enumerate(l['geoms']) if g['role'] != 'actuator'])
    if others is not None and rest > 1e-6:
        mm = others.to_mesh()
        tm = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
        tm.density = rest / tm.volume
        parts.append((rest, tm.center_mass, tm.moment_inertia))
    mt = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / mt
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = c - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mt, com, I


def capacity(spec, j):
    c = j['capacity']
    a = spec['actuators'][c['model']]
    k = c['count'] * c['ratio']
    return a['peak_nm'] * k, a['noload_rads'] / c['ratio']


def fmt(v):
    return ' '.join(f'{x:.6g}' for x in v)


# Arm against pelvis and upper leg: only these collide with each other (bit 4, apart from the floor's bit 1 and the
# terrain's bit 2), so the policy can't pass its hands through its hips (2026-11-04: a soles-only sim let it, 65 cm3).
# Spheres fitted to each link's hull: boxes and capsules from the mesh bounds overlapped at the zero pose (15 contacts)
# and read 10 cm deep where the meshes overlap by a few cm. supermimic's reward reads the same spheres by name.
SELF_ARM = {'elbow_link': 3, 'wrist_yaw_link': 3, 'wrist_pitch_link': 2, 'hand': 5}
SELF_BODY = {'hip_pitch_link': 4, 'hip_roll_link': 8, 'hip_yaw_link': 8}
SELF_PELVIS = 6
SELF_BIT = 4
# Penalty only (2026-11-04): spheres the reward measures but the physics never tests, so they cost it nothing. Leg against
# leg (the hip spheres above count too) and hand and forearm against the torso and shanks. Same fit as above.
# Leg spheres stand up to 12 mm proud of the hull: the walks' real touches are a few cm3, and spheres kept inside a thigh's flat
# inner face missed them. The legs stand 44 mm apart, so the zero pose stays clear; a near miss reads as a light touch.
SELF_LEG = {'hip_yaw_link': 12, 'knee_link': 12, 'foot': 8}
LEG_PROUD = 0.012
SELF_TORSO = 24
# Floor contact for the rest of the body (2026-11-04), so kneeling and falls land on something. Floor-only: bit 8 is set on
# the floor's contype and is these geoms' only affinity, so they meet nothing else, soles and terrain included. Fitted to the
# link's hull. Capsules and a sphere give 1-2 contacts each on a plane, a box 4. The knee_link capsule is both knee and shin.
FLOOR_BIT = 8
FLOOR_LINKS = {'knee_link': 'capsule', 'elbow_link': 'capsule', 'wrist_yaw_link': 'capsule', 'hand': 'capsule'}
FLOOR_BODY = {'pelvis': 'capsule', 'torso': 'box', 'head': 'sphere'}


def hull_points(m, seed=1):
    """Up to 8000 points filling the mesh's convex hull."""
    rng = np.random.default_rng(seed)
    hull = ConvexHull(m.vertices)
    lo, hi = m.bounds
    pts = rng.uniform(lo, hi, (60000, 3))
    return pts[(hull.equations[:, :3] @ pts.T + hull.equations[:, 3:]).max(0) <= 0][:8000]


def fit_spheres(m, k, seed=1):
    """k spheres over the mesh's convex hull: k-means of points filling the hull, each sphere reaching most (80 %) of
    its cluster, so it covers the link without standing far proud of it. [(centre, radius)] in the mesh's frame."""
    pts = hull_points(m, seed)
    c, lab = kmeans2(pts, k, seed=seed, minit='++')
    return [(c[i], float(np.percentile(np.linalg.norm(pts[lab == i] - c[i], axis=1), 80))) for i in range(k) if (lab == i).any()]


def fit_spheres_inside(m, k, proud=0.0, seed=1):
    """fit_spheres, each radius also capped at its centre's distance to the hull plus proud: a broad flat link (the torso, 3 mm
    from the elbow) would otherwise bulge past its own sides and read as touching at the zero pose."""
    hull = ConvexHull(m.vertices)
    return [(c, min(r, proud + float(-(hull.equations[:, :3] @ c + hull.equations[:, 3]).max()))) for c, r in fit_spheres(m, k, seed)]


def sphere_geoms(kind, ln, m, k, attr, ind):
    return [f'{ind}<geom name="selfcol_{kind}_{ln}_{i}" type="sphere" pos="{fmt(c)}" size="{r:.4f}" {attr} group="3"/>'
            for i, (c, r) in enumerate(fit_spheres_inside(m, k, LEG_PROUD if kind == 'leg' else 0.0))]


def self_collision_geoms(ln, m, ind):
    """Named selfcol_<arm, body, leg or torso>_<link>_<i>."""
    legs = sphere_geoms('leg', ln, m, SELF_LEG[ln[2:]], 'contype="0" conaffinity="0"', ind) if ln[2:] in SELF_LEG else []
    return legs + trunk_geoms(ln, m, ind)


def trunk_geoms(ln, m, ind):
    if ln[2:] in SELF_ARM:
        kind, k, attr = 'arm', SELF_ARM[ln[2:]], f'contype="{SELF_BIT}" conaffinity="0" condim="1"'
    elif ln == 'pelvis' or ln[2:] in SELF_BODY:
        kind, k, attr = 'body', SELF_PELVIS if ln == 'pelvis' else SELF_BODY[ln[2:]], f'contype="0" conaffinity="{SELF_BIT}" condim="1"'
    elif ln == 'torso':
        kind, k, attr = 'torso', SELF_TORSO, 'contype="0" conaffinity="0"'
    else:
        return []
    return sphere_geoms(kind, ln, m, k, attr, ind)


def floor_geoms(ln, m, ind):
    """One floor-only primitive named floorcol_<link>, fitted to the link's hull, or nothing. A capsule lies along the
    longest side of the mesh's bounds and reaches 90 % of the hull's points around that line."""
    shape = FLOOR_BODY.get(ln) or FLOOR_LINKS.get(ln[2:])
    if not shape:
        return []
    attr = f'name="floorcol_{ln}" contype="0" conaffinity="{FLOOR_BIT}" group="3"'
    lo, hi = m.bounds
    if shape == 'box':
        return [f'{ind}<geom {attr} type="box" pos="{fmt((lo + hi) / 2)}" size="{fmt((hi - lo) / 2)}"/>']
    pts = hull_points(m)
    c = pts.mean(0)
    if shape == 'sphere':
        return [f'{ind}<geom {attr} type="sphere" pos="{fmt(c)}" size="{np.percentile(np.linalg.norm(pts - c, axis=1), 90):.4f}"/>']
    k = int(np.argmax(hi - lo))
    off = pts - c
    off[:, k] = 0.0
    r = float(np.percentile(np.linalg.norm(off, axis=1), 90))
    a, b = c.copy(), c.copy()
    a[k], b[k] = lo[k] + r, hi[k] - r
    return [f'{ind}<geom {attr} type="capsule" fromto="{fmt(a)} {fmt(b)}" size="{r:.4f}"/>']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion', default=os.path.join(OUT, 'meshes', 'fusion'))
    a = ap.parse_args()
    spec = G.load()
    R = G.Robot(spec)
    S = C.Solids(spec)
    frames = link_frames(R)
    mesh_dir = os.path.join(OUT, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    src, meshes = {}, {}
    inert = {ln: v[:3] for ln, v in MM.link_inertials(spec, S).items()}   # from the materials
    for ln in spec['links']:
        m, src[ln] = link_mesh(spec, S, ln, a.fusion)
        m.apply_translation(-frames[ln])                 # mesh in the link frame
        m.export(os.path.join(mesh_dir, f'{ln}.stl'))
        meshes[ln] = m

    # ---------------- URDF
    u = [f'<?xml version="1.0"?>', f'<robot name="{NAME}">',
         '  <!-- generated by tools/make_robot_files.py from spec/robot.json; do not edit -->']
    for ln in R.order:
        mt, com, I = inert[ln]
        c = com - frames[ln]
        u += [f'  <link name="{ln}">',
              f'    <inertial><origin xyz="{fmt(c)}" rpy="0 0 0"/><mass value="{mt:.4f}"/>',
              f'      <inertia ixx="{I[0, 0]:.6g}" ixy="{I[0, 1]:.6g}" ixz="{I[0, 2]:.6g}" iyy="{I[1, 1]:.6g}" '
              f'iyz="{I[1, 2]:.6g}" izz="{I[2, 2]:.6g}"/></inertial>',
              f'    <visual><geometry><mesh filename="meshes/{ln}.stl"/></geometry></visual>']
        if ln.endswith('_foot'):
            sole = [g for g in spec['links'][ln]['geoms'] if 'sole' in g['name']][0]
            sole = sole.get('contact', sole)              # a shaped outsole carries its contact box
            u.append(f'    <collision><origin xyz="{fmt(np.array(sole["c"]) - frames[ln])}"/>'
                     f'<geometry><box size="{fmt(sole["size"])}"/></geometry></collision>')
        else:
            u.append(f'    <collision><geometry><mesh filename="meshes/{ln}.stl"/></geometry></collision>')
        u.append('  </link>')
    for j in spec['joints']:
        eff, vel = capacity(spec, j)
        u += [f'  <joint name="{j["name"]}" type="revolute">',
              f'    <parent link="{j["parent"]}"/><child link="{j["child"]}"/>',
              f'    <origin xyz="{fmt(frames[j["child"]] - frames[j["parent"]])}" rpy="0 0 0"/>',
              f'    <axis xyz="{fmt(G.AXIS[j["axis"]])}"/>',
              f'    <limit lower="{j["lower"]}" upper="{j["upper"]}" effort="{eff:.1f}" velocity="{vel:.2f}"/>',
              f'    <dynamics damping="{dynamics(spec, j)[1]:.4g}" friction="{dynamics(spec, j)[2]:.4g}"/>',
              '  </joint>']
    u.append('</robot>')
    with open(os.path.join(OUT, f'{NAME}.urdf'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(u))

    # ---------------- MJCF: visual meshes, sole boxes for contact, position servos at peak torque
    kids = {}
    for j in spec['joints']:
        kids.setdefault(j['parent'], []).append(j)
    x = [f'<mujoco model="{NAME}">',
         '  <!-- generated by tools/make_robot_files.py from spec/robot.json; do not edit -->',
         '  <compiler angle="radian" meshdir="meshes"/>',
         '  <option timestep="0.002" integrator="implicitfast"/>',
         '  <default><joint damping="0.1" armature="0.005"/>',   # each joint sets its own (dynamics())
         '    <geom contype="0" conaffinity="0" group="2" rgba="0.75 0.76 0.78 1"/></default>',
         '  <asset>']
    x += [f'    <mesh name="{ln}" file="{ln}.stl"/>' for ln in spec['links']]
    x += ['  </asset>', '  <worldbody>',
          f'    <geom name="floor" type="plane" size="5 5 0.1" contype="{1 | FLOOR_BIT}" conaffinity="1" group="0" '
          'rgba="0.9 0.9 0.9 1"/>',
          '    <light pos="1 1 3" dir="-0.3 -0.3 -1"/>']

    def body(ln, parent_frame, ind):
        mt, com, I = inert[ln]
        pos = frames[ln] - parent_frame
        s = '  ' * ind
        out = [f'{s}<body name="{ln}" pos="{fmt(pos)}">']
        if ln == spec['root']:
            out.append(f'{s}  <freejoint name="root"/>')
        else:
            j = R.parent_joint[ln]
            arm, damp, fric = dynamics(spec, j)
            out.append(f'{s}  <joint name="{j["name"]}" axis="{fmt(G.AXIS[j["axis"]])}" '
                       f'range="{j["lower"]} {j["upper"]}" armature="{arm:.4g}" damping="{damp:.4g}" '
                       f'frictionloss="{fric:.4g}"/>')
        ev, evec = np.linalg.eigh(I)
        q = trimesh.transformations.quaternion_from_matrix(
            np.vstack([np.hstack([evec * np.sign(np.linalg.det(evec)), np.zeros((3, 1))]), [0, 0, 0, 1]]))
        out.append(f'{s}  <inertial pos="{fmt(com - frames[ln])}" quat="{fmt(q)}" mass="{mt:.4f}" '
                   f'diaginertia="{fmt(np.maximum(ev, 1e-6))}"/>')
        out.append(f'{s}  <geom type="mesh" mesh="{ln}"/>')
        if ln.endswith('_foot'):
            sole = [g for g in spec['links'][ln]['geoms'] if 'sole' in g['name']][0]
            sole = sole.get('contact', sole)
            out.append(f'{s}  <geom name="{ln}_sole" type="box" pos="{fmt(np.array(sole["c"]) - frames[ln])}" '
                       f'size="{fmt(np.array(sole["size"]) / 2)}" contype="1" conaffinity="1" group="0" '
                       f'friction="1.0 0.02 0.001" rgba="0.2 0.2 0.2 1"/>')
        out += self_collision_geoms(ln, meshes[ln], s + '  ')
        out += floor_geoms(ln, meshes[ln], s + '  ')
        for j in kids.get(ln, []):
            out += body(j['child'], frames[ln], ind + 1)
        out.append(f'{s}</body>')
        return out

    x += body(spec['root'], np.zeros(3), 2)
    x += ['  </worldbody>', '  <actuator>']
    for j in spec['joints']:
        eff, _ = capacity(spec, j)
        kp = min(max(8 * eff, 20.0), 1500.0)
        # kv = kp/6: with kp/25 the standing robot rocked heel to toe (whole body ~50 kg m^2 about the ankles,
        # far under critical damping); kp = 4 x peak leaned 5.5 deg under its own weight (servo give, no gravity
        # feed-forward). 2026-10-31
        x.append(f'    <position name="{j["name"]}" joint="{j["name"]}" kp="{kp:.0f}" kv="{kp / 6:.1f}" '
                 f'forcerange="{-eff:.1f} {eff:.1f}"/>')
    x += ['  </actuator>', '</mujoco>']
    with open(os.path.join(OUT, f'{NAME}.xml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(x))
    n_f = sum(v == 'fusion' for v in src.values())
    print(f'{NAME}.urdf + .xml: {len(spec["links"])} links ({n_f} meshes from Fusion, '
          f'{len(src) - n_f} from the spec), total mass {sum(v[0] for v in inert.values()):.2f} kg')


if __name__ == '__main__':
    main()
