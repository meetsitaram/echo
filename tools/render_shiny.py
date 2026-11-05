"""The robot with its cover off in polished metal: an OpenGL pass reflecting a studio of softboxes, with contact shadows from the depth.
Polish is for looking at, not a surface finish we have chosen. The camera is named by where it stands: front is +x, the robot's right is -y.

  python tools/render_shiny.py --out reports/parts/skeleton_shiny.png
"""
from __future__ import annotations

import argparse
import ctypes
import math
import sys
from pathlib import Path

import glfw
import mujoco
import numpy as np
import OpenGL.GL as gl
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402
import geom as G  # noqa: E402
from render_motion import XML  # noqa: E402

BG = (232, 231, 228)
ROLES = ('struct', 'actuator', 'linkage', 'hardware', 'bearing')
ELECTRONICS = {'battery', 'battery lower', 'compute'}
BEND = {'L_hip_pitch': -0.35, 'L_knee': 0.75, 'L_ankle_pitch': -0.35, 'R_hip_pitch': 0.1, 'R_knee': 0.1,
        'L_shoulder_pitch': 0.25, 'R_shoulder_pitch': -0.25, 'L_elbow': -0.5, 'R_elbow': -0.5}
# name, camera direction from the target as (front, left), height, distance, target height
VIEWS = [('Front right', (1.0, -0.8), 0.15, 2.45, 0.86), ('Side', (0.0, -1.0), 0.05, 2.45, 0.86),
         ('Back left', (-1.0, 0.8), 0.15, 2.45, 0.86), ('Legs', (1.0, -0.75), 0.1, 1.45, 0.50)]
# (tint rgb, metal). Metal reflects the studio; the rest is painted.
LOOK = {'struct': ((0.93, 0.94, 0.96), 1.0), 'black': ((0.13, 0.135, 0.15), 1.0), 'actuator': ((0.17, 0.175, 0.19), 1.0), 'linkage': ((0.80, 0.81, 0.84), 1.0),
        'hardware': ((0.42, 0.43, 0.46), 1.0), 'bearing': ((0.72, 0.73, 0.75), 1.0), 'electronics': ((0.07, 0.08, 0.10), 0.35)}

VS = """#version 330 core
layout(location=0) in vec3 p; layout(location=1) in vec3 n; layout(location=2) in vec3 tint; layout(location=3) in float metal;
uniform mat4 vp; out vec3 P; out vec3 N; out vec3 T; out float M;
void main(){ P=p; N=n; T=tint; M=metal; gl_Position=vp*vec4(p,1.0); }"""
FS = """#version 330 core
in vec3 P; in vec3 N; in vec3 T; in float M; uniform vec3 eye; out vec4 c;
float box(float a,float a0,float a1,float s){ return smoothstep(a0-s,a0+s,a)*(1.0-smoothstep(a1-s,a1+s,a)); }
vec3 env(vec3 r){
  float e=asin(clamp(r.z,-1.0,1.0)); float az=atan(r.y,r.x);
  float g=mix(0.16,0.34,smoothstep(-1.4,0.0,e));
  g=mix(g,0.46,smoothstep(0.0,0.5,e)); g=mix(g,0.30,smoothstep(0.7,1.5,e));
  g+=0.95*box(az,-0.95,-0.25,0.03)*box(e,-0.15,1.0,0.04);
  g+=0.7*box(az,0.55,1.25,0.04)*box(e,0.0,0.9,0.05);
  g+=0.6*box(az,2.5,3.1,0.05)*box(e,-0.3,0.8,0.05);
  g+=0.6*box(az,-2.7,-2.0,0.05)*box(e,-0.1,0.9,0.05);
  g+=0.45*box(e,1.1,1.45,0.05);
  g-=0.20*box(az,-0.14,0.1,0.015)*box(e,-0.7,1.1,0.02);
  return vec3(g,g*1.01,g*1.03);
}
void main(){
  vec3 n=normalize(N); vec3 v=normalize(eye-P); if(dot(n,v)<0.0) n=-n;
  vec3 r=reflect(-v,n);
  float f=pow(1.0-max(dot(n,v),0.0),5.0);
  vec3 refl=env(r)*T+vec3(f*0.35);
  vec3 l=normalize(vec3(0.4,-0.5,0.8));
  vec3 paint=T*(0.35+0.65*max(dot(n,l),0.0))+0.25*env(r)*(0.05+0.95*f);
  vec3 col=mix(paint,refl,M);
  col=pow(clamp(col/(1.0+0.15*col),0.0,1.0),vec3(1.0/2.2));
  c=vec4(col,1.0);
}"""


def posed_meshes():
    spec = G.load()
    S = C.Solids(spec)
    m0 = mujoco.MjModel.from_xml_path(str(XML))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for j, q in BEND.items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = q
    mujoco.mj_kinematics(m, d)
    verts = []
    count = {}
    for (ln, i), solid in S.geom.items():
        g = spec['links'][ln]['geoms'][i]
        if solid is None or g.get('role') not in ROLES:
            continue
        mesh = solid.to_mesh()
        v, f = np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, int)
        if len(f) < 4:
            continue
        b = m.body(ln).id
        R = d.xmat[b].reshape(3, 3)
        v = (v - d0.xpos[b]) @ R.T + d.xpos[b]  # body frames are world-aligned at zero
        t = trimesh.Trimesh(v, f, process=True)
        s = trimesh.graph.smooth_shade(t, angle=math.radians(35))  # keeps machined edges sharp
        tint, metal = LOOK['electronics'] if g['name'] in ELECTRONICS else LOOK[g['role']]
        tri = s.vertices[s.faces].reshape(-1, 3)
        nor = s.vertex_normals[s.faces].reshape(-1, 3)
        rgb = np.tile(tint, (len(tri), 1)).astype(float)
        if g.get('role') == 'struct' and g.get('finish'):      # dark is a motor and its housing, silver is structure: the part's own finish, masked where it has zones
            # zones are in the zero pose: read them on the unposed solid's triangles
            z0 = np.asarray(mesh.vert_properties[:, :3], float)
            t0 = trimesh.Trimesh(z0, f, process=True)
            s0 = trimesh.graph.smooth_shade(t0, angle=math.radians(35))
            names = G.finish_of(g, s0.vertices[s0.faces].mean(1))
            rgb[np.repeat(names == 'black', 3)] = LOOK['black'][0]
        verts.append(np.hstack((tri, nor, rgb, np.full((len(tri), 1), metal))))
        count[g['role']] = count.get(g['role'], 0) + 1
    return np.vstack(verts).astype(np.float32), count


def perspective(fovy, aspect, n, f):
    t = 1 / math.tan(fovy / 2)
    return np.array([[t / aspect, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]], float)


def look_at(eye, tgt):
    z = eye - tgt
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = x, y, z
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def compile_program():
    def sh(kind, src):
        s = gl.glCreateShader(kind)
        gl.glShaderSource(s, src)
        gl.glCompileShader(s)
        if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(s))
        return s
    p = gl.glCreateProgram()
    gl.glAttachShader(p, sh(gl.GL_VERTEX_SHADER, VS))
    gl.glAttachShader(p, sh(gl.GL_FRAGMENT_SHADER, FS))
    gl.glLinkProgram(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    data, count = posed_meshes()
    W, H, SS = 900, 1500, 2
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, False)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(64, 64, 'x', None, None)
    glfw.make_context_current(win)
    prog = compile_program()
    vao, vbo = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
    stride = data.shape[1] * 4
    for loc, size, off in ((0, 3, 0), (1, 3, 3), (2, 3, 6), (3, 1, 9)):
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, False, stride, ctypes.c_void_p(off * 4))
    w, h = W * SS, H * SS
    ms_fbo, ms_col, ms_dep = gl.glGenFramebuffers(1), gl.glGenRenderbuffers(1), gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, ms_col)
    gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, 4, gl.GL_RGBA8, w, h)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, ms_dep)
    gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, 4, gl.GL_DEPTH_COMPONENT32F, w, h)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, ms_fbo)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, ms_col)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, ms_dep)
    fbo, col, dep = gl.glGenFramebuffers(1), gl.glGenRenderbuffers(1), gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, col)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, w, h)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, dep)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32F, w, h)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, col)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, dep)
    gl.glUseProgram(prog)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, w, h)
    n_v = len(data)
    root = data[:, :2].mean(0)
    sheet = Image.new('RGB', (W * len(VIEWS), H + 90), BG)
    font, small = ImageFont.load_default(size=40), ImageFont.load_default(size=32)
    near, far = 0.1, 20.0
    for k, (title, (fx, fy), up, dist, zt) in enumerate(VIEWS):
        dirn = np.array([fx, fy, up], float)
        dirn /= np.linalg.norm(dirn)
        tgt = np.array([root[0], root[1], zt])
        eye = tgt + dirn * dist
        vp = perspective(math.radians(45), W / H, near, far) @ look_at(eye, tgt)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, ms_fbo)
        gl.glClearColor(BG[0] / 255, BG[1] / 255, BG[2] / 255, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(prog, 'vp'), 1, True, vp.astype(np.float32))
        gl.glUniform3fv(gl.glGetUniformLocation(prog, 'eye'), 1, eye.astype(np.float32))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, n_v)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, ms_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo)
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        rgb = np.frombuffer(gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE), np.uint8).reshape(h, w, 3)[::-1].astype(float)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, ms_fbo)
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_DEPTH_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        zb = np.frombuffer(gl.glReadPixels(0, 0, w, h, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT), np.float32).reshape(h, w)[::-1].astype(float)
        mask = zb < 1.0
        # Shadows where parts sit close behind others: true depth against its local blur.
        z = 2 * near * far / (far + near - (2 * zb - 1) * (far - near))
        z_f = np.where(mask, z, dist + 1.0)
        occ = np.clip((z_f - ndimage.gaussian_filter(z_f, 22 * SS)) / 0.05, 0, 1) * mask
        rgb[mask] = (rgb * (1 - 0.55 * occ[..., None]))[mask]
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        ImageDraw.Draw(im).text((30, 24), title, fill=(50, 50, 55), font=font)
        sheet.paste(im, (k * W, 0))
    glfw.terminate()
    note = f'Cover off, from the spec: {count.get("actuator", 0)} motors, {count.get("struct", 0)} structure shapes, {count.get("hardware", 0)} screws, {count.get("bearing", 0)} bearings. Polished look for viewing, not a chosen finish. Wiring not modelled; most structure above the knee is placeholder.'
    ImageDraw.Draw(sheet).text((30, H + 26), note, fill=(150, 40, 30), font=small)
    sheet.save(a.out)
    print(a.out, sheet.size, count, n_v // 3, 'triangles')


if __name__ == '__main__':
    main()
