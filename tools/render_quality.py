"""Product-quality OpenGL renders of echo, stills and replay movies. Everything here is look only: the spec is not touched.

Materials: tinted dark glass visor with an optional soft inner glow, glossy black hubs and shells, a procedural knit on the grey garment
(in each part's body frame, so it stays on the body in motion), matte shoes and rubber hands. Studio: seamless light backdrop, a faint
floor reflection, soft floor contact shadow. Camera names and framing follow train/render_forces.py.

  python tools/render_quality.py --sheet reports/parts/quality_covered.png
  python tools/render_quality.py --sheet reports/parts/quality_skeleton.png --skeleton
  python tools/render_quality.py --closeup head --glow blue --out head_blue.png
  python tools/render_quality.py --rec replay.npz --out clip.mp4 --start 2 --dur 5 --title "Relaxed walk" --tag "boneseed clip, used in training"

MuJoCo camera convention, as render_forces.py: the camera looks along azimuth; it stands at lookat - distance * (cos el cos az, cos el sin az, sin el).
So 'hero' at heading + 135 degrees stands at heading - 45 degrees: the robot's front right. Elevation -6 is above the target.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import pickle
import sys
import tempfile
from pathlib import Path

import glfw
import imageio
import mujoco
import numpy as np
import OpenGL.GL as gl
import OpenGL.images as _gli
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402
import geom as G  # noqa: E402
from render_motion import VIEWS as MJ_VIEWS, XML, smooth_track  # noqa: E402

_gli.COMPONENT_COUNTS[gl.GL_RG] = 2
KNIT, VISOR, GLOSS, MATTE, RUBBER, METAL, FLAT = 0, 1, 2, 3, 4, 5, 6
SKELETON = ('struct', 'actuator', 'linkage', 'hardware', 'bearing')
ELECTRONICS = {'battery', 'battery lower', 'compute'}
METALS = {'struct': (0.93, 0.94, 0.96), 'actuator': (0.11, 0.115, 0.13), 'linkage': (0.80, 0.81, 0.84), 'hardware': (0.42, 0.43, 0.46), 'bearing': (0.72, 0.73, 0.75)}
BEND = {'L_hip_pitch': -0.10, 'L_knee': 0.20, 'L_ankle_pitch': -0.10, 'R_hip_pitch': -0.10, 'R_knee': 0.20, 'R_ankle_pitch': -0.10, 'L_elbow': -0.25, 'R_elbow': -0.25}
# name, camera direction from the target as (front, left), height, distance, target height. Front is +x, the robot's right is -y.
SHEET = [('Front right', (1.0, -0.8), 0.15, 2.75, 0.88), ('Side', (0.0, -1.0), 0.05, 2.75, 0.88), ('Back left', (-1.0, 0.8), 0.15, 2.75, 0.88), ('Legs', (1.0, -0.75), 0.1, 1.45, 0.50)]
GLOWS = {'teal': (0.05, 0.62, 0.90), 'blue': (0.10, 0.32, 1.00), 'green': (0.08, 0.95, 0.42), 'off': (0.0, 0.0, 0.0)}
RGB = ((30, 170, 70), (235, 105, 10))
N_PER_M = 500.0  # metres of arrow per newton is half render_forces.py's: per contact point, capped at ARROW_MAX
ARROW_MAX = 0.85

VS = """#version 330 core
layout(location=0) in vec3 p; layout(location=1) in vec3 n; layout(location=2) in vec3 col; layout(location=3) in vec2 mp;
uniform mat4 vp; uniform mat4 model; uniform float mirror; uniform vec3 eye; uniform float bias;
out vec3 P; out vec3 N; out vec3 LP; out vec3 LN; out vec3 T; flat out int M; flat out float ID; flat out float LAYER;
void main(){
  vec4 w=model*vec4(p,1.0); P=w.xyz; N=mat3(model)*n; LP=p; LN=n; T=col; M=int(mp.x+0.5); ID=mp.y;
  LAYER = (M==FLATID)?0.0:fract(mp.y)*10.0;
  vec4 q=w; if(mirror>0.5) q.z=-q.z; else q.xyz+=normalize(eye-q.xyz)*LAYER*bias;  // decals sit a hair in front of the panel they lie on, in metres
  gl_Position=vp*q;
}""".replace('FLATID', str(FLAT))

FS = """#version 330 core
in vec3 P; in vec3 N; in vec3 LP; in vec3 LN; in vec3 T; flat in int M; flat in float ID; flat in float LAYER;
uniform vec3 eye; uniform vec3 key; uniform float envyaw; uniform mat3 modelR; uniform float mirror; uniform float ss;
uniform sampler2DShadow shadow; uniform mat4 lightvp; uniform vec3 flatcol;
uniform vec3 glow; uniform vec3 vmin; uniform vec3 vmax; uniform float skeleton; uniform float useroom; uniform sampler2D roomenv; uniform float roomgain; uniform float keyk;
out vec4 c;
const float PI2=6.28318531;
float box(float a,float a0,float a1,float s){ return smoothstep(a0-s,a0+s,a)*(1.0-smoothstep(a1-s,a1+s,a)); }
// The studio, turned with the camera. rough widens every edge. Softboxes run brighter than 1: tone mapping and Fresnel bring them down.
float env(vec3 r,float rough,float room){
  float e=asin(clamp(r.z,-1.0,1.0)); float az=atan(r.y,r.x)-envyaw; az=atan(sin(az),cos(az));
  float s=0.02+rough*0.9;
  float g=mix(0.34,0.50,smoothstep(-1.2,0.0,e));       // pale floor and backdrop
  g=mix(g,0.62,smoothstep(0.0,0.5,e)); g=mix(g,0.32,smoothstep(0.7,1.5,e)); g*=room;
  g+=6.0*box(az,-1.25,-0.45,s)*box(e,-0.05,0.95,s);    // the big key box, its cross bar a window in the glass
  g-=2.2*box(az,-0.88,-0.82,s)*box(e,-0.05,0.95,s);
  g+=3.2*box(az,0.35,0.62,s)*box(e,-0.25,0.85,s);      // a tall strip on the other side
  g+=2.6*box(az,2.45,3.05,s)*box(e,-0.2,0.8,s);        // two kickers behind
  g+=2.6*box(az,-2.95,-2.3,s)*box(e,-0.1,0.9,s);
  float cy=cos(envyaw),sy=sin(envyaw); vec2 rr=vec2(cy*r.x+sy*r.y,-sy*r.x+cy*r.y);
  g+=2.4*box(rr.x,-0.55,0.15,s)*box(rr.y,-0.28,0.28,s)*step(0.0,r.z);   // overhead
  return g;
}
vec3 eqsample(vec3 d,float lod){ vec2 uv=vec2(atan(d.y,d.x)/PI2+0.5,asin(clamp(d.z,-1.0,1.0))/3.14159265+0.5); return textureLod(roomenv,uv,lod).rgb; }
// in the room, reflections and fill come from the room itself: a low-res image of it taken from where echo stands
vec3 env3(vec3 r,float rough,float room){ return useroom>0.5?eqsample(r,rough*9.0)*mix(1.0,room,0.35):vec3(env(r,rough,room)); }
vec3 hemi3(vec3 n,float lo,float hi){ return useroom>0.5?eqsample(n,6.2)*roomgain:vec3(mix(lo,hi,0.5+0.5*n.z)); }
float lit_shadow(vec3 n){
  float g=1.0-max(dot(n,key),0.0);
  vec4 q=lightvp*vec4(P+n*(0.003+0.012*g),1.0); vec3 u=q.xyz*0.5+0.5;
  if(u.x<0.0||u.x>1.0||u.y<0.0||u.y>1.0) return 1.0;
  float s=0.0; const float R=1.6/2048.0;
  for(int i=-2;i<=2;i++) for(int j=-2;j<=2;j++) s+=texture(shadow,vec3(u.xy+vec2(i,j)*R,u.z-0.00025));
  return s/25.0;
}
// Knit: stitches at about 2.5 mm, courses each 7.5 mm that still read when the whole body is in the frame. Each fades before it can alias.
const float PF=0.0025, PM=0.0075;
float stitch(vec2 ab){
  float r=ab.y/PF; float shift=0.5*mod(floor(r+0.5),2.0);
  return (0.5+0.5*cos(PI2*r))*(0.72+0.28*cos(PI2*(ab.x/PF+shift)));
}
float knit_height(vec3 lp,vec3 ln,out float fine){
  vec3 w=pow(abs(ln),vec3(6.0)); w/=w.x+w.y+w.z;
  float fw=max(length(dFdx(lp)),length(dFdy(lp)))*ss;     // metres per picture pixel, obliquity included
  float ff=1.0-smoothstep(PF/4.0,PF/2.6,fw); float fm=1.0-smoothstep(PM/4.5,PM/2.6,fw);
  float hf=0.0,hm=0.0;
  if(ff>0.0) hf=w.x*stitch(lp.yz)+w.y*stitch(lp.xz)+w.z*stitch(lp.yx);
  hm=w.x*cos(PI2*lp.z/PM)+w.y*cos(PI2*lp.z/PM)+w.z*cos(PI2*lp.x/PM);
  fine=ff;
  return ff*(hf-0.5)*0.75 + fm*(0.5*hm)*0.32*(1.0-0.6*ff);
}
vec3 bump(vec3 n,float h,float k){
  vec3 dx=dFdx(P),dy=dFdy(P); vec3 r1=cross(dy,n),r2=cross(n,dx); float det=dot(dx,r1);
  vec3 g=sign(det)*(dFdx(h)*r1+dFdy(h)*r2);
  return normalize(abs(det)*n-k*g);
}
float tonemap1(float x){ const float a=0.72; return x<a?x:a+(1.0-a)*(1.0-exp(-(x-a)/(1.0-a))); }
vec3 tonemap(vec3 v){ return vec3(tonemap1(v.r),tonemap1(v.g),tonemap1(v.b)); }
void main(){
  vec3 n=normalize(N); vec3 realeye=mirror>0.5?vec3(eye.xy,-eye.z):eye; vec3 v=normalize(realeye-P);
  if(dot(n,v)<0.0) n=-n;
  float nv=max(dot(n,v),0.0);
  float sh=mirror>0.5?1.0:lit_shadow(n);
  vec3 col;
  if(M==FLATMAT){
    col=flatcol*(0.55+0.55*max(dot(n,normalize(vec3(key.xy,1.0))),0.0)+0.35*pow(max(dot(reflect(-v,n),normalize(vec3(key.xy,1.0))),0.0),24.0));   // shaded like a solid, a little brighter than lit paint
  } else if(M==KNITMAT){
    float fine; float h=knit_height(LP,normalize(LN),fine);
    vec3 nb=bump(n,h,0.00042);
    float nvb=max(dot(nb,v),0.0);
    float wrap=max((dot(nb,key)+0.30)/1.30,0.0);
    float hemi=0.5+0.5*nb.z;
    vec3 diff=hemi3(nb,0.38,0.54)+vec3(1.0,0.97,0.92)*keyk*0.56*wrap*mix(0.12,1.0,sh);
    vec3 alb=T*(1.0-0.22*clamp(0.5-h,0.0,1.0));
    float sheen=pow(1.0-nvb,3.5)*0.24*mix(0.5,1.0,hemi);
    col=alb*diff+vec3(sheen);
  } else if(M==VISORMAT){
    vec3 r=reflect(-v,n);
    float f=0.07+0.42*pow(1.0-nv,5.0);
    vec3 tint=vec3(0.86,0.95,1.0);
    vec3 refl=env3(r,0.015,0.30)*f*tint;
    // the glow lives behind the glass: seen through it, sampled a little deeper along the view, with no source to find
    vec3 vl=transpose(modelR)*v; vec3 q=LP-vl*0.03;
    vec3 ce=0.5*(vmin+vmax); vec3 sz=vmax-vmin;
    float gz=(q.z-(ce.z+0.08*sz.z))/(0.11*sz.z); float gy=(q.y-ce.y)/(0.36*sz.y);
    float g=exp(-gz*gz)*exp(-gy*gy*gy*gy)+0.10*exp(-0.22*gz*gz)*exp(-0.8*gy*gy);
    col=vec3(0.003,0.0035,0.0045)+glow*g*(1.0-f)+refl;
  } else if(M==GLOSSMAT){
    vec3 r=reflect(-v,n);
    float f=0.04+0.55*pow(1.0-nv,5.0);
    col=T*(0.25+0.5*max(dot(n,key),0.0)*sh)+env3(r,0.07,0.35)*f;
  } else if(M==RUBBERMAT){
    vec3 r=reflect(-v,n);
    float f=0.025+0.14*pow(1.0-nv,5.0);
    vec3 diff=hemi3(n,0.5,0.8)+vec3(0.6*max(dot(n,key),0.0)*sh);
    col=T*diff+env3(r,0.45,0.6)*f;
  } else if(M==METALMAT){
    vec3 r=reflect(-v,n);
    float f=pow(1.0-nv,5.0);
    col=(T*env3(r,0.06,1.0)*0.72+vec3(f*0.3))*mix(0.55,1.0,sh);
  } else {
    float wrap=max((dot(n,key)+0.25)/1.25,0.0);
    vec3 diff=hemi3(n,0.38,0.54)*0.86+vec3(1.0,0.97,0.92)*keyk*0.44*wrap*mix(0.12,1.0,sh);
    float sheen=pow(1.0-nv,4.0)*0.08;
    col=T*diff+vec3(sheen);
  }
  col=pow(tonemap(col),vec3(1.0/2.2));
  if(mirror>0.5){
    if(P.z<0.0) discard;
    float a=exp(-P.z/0.32); c=vec4(col*a,a);
  } else c=vec4(col,1.0);
}""".replace('FLATMAT', str(FLAT)).replace('KNITMAT', str(KNIT)).replace('VISORMAT', str(VISOR)).replace('GLOSSMAT', str(GLOSS)).replace('RUBBERMAT', str(RUBBER)).replace('METALMAT', str(METAL))

SHADOW_FS = "#version 330 core\nvoid main(){}"
ID_VS = """#version 330 core
layout(location=0) in vec3 p; layout(location=3) in vec2 mp; uniform mat4 vp; uniform mat4 model; uniform mat4 view; uniform vec3 eye;
flat out float ID; flat out float K; out float Z;
void main(){ vec4 w=model*vec4(p,1.0); ID=floor(mp.y); K=(int(mp.x+0.5)==0)?1.0:0.0; Z=-(view*w).z;
  w.xyz+=normalize(eye-w.xyz)*fract(mp.y)*10.0*0.001; gl_Position=vp*w; }"""
ID_FS = """#version 330 core
flat in float ID; flat in float K; in float Z; out vec4 c; void main(){ c=vec4(K>0.5?ID:-ID,Z,0.0,1.0); }"""

AO_FS = """#version 330 core
in vec2 u; out vec4 c; uniform sampler2D gbuf; uniform vec2 res; uniform float tanv; uniform float aspect; uniform vec2 radii;
vec3 vp(ivec2 t){ float z=texelFetch(gbuf,clamp(t,ivec2(0),ivec2(res)-1),0).g; vec2 d=(vec2(t)+0.5)/res*2.0-1.0; return vec3(d.x*tanv*aspect*z,d.y*tanv*z,-z); }
float hash(vec2 p){ return fract(52.9829189*fract(dot(p,vec2(0.06711056,0.00583715)))); }
// Horizon-style occlusion in view space: how far the geometry around a point rises above its surface, within R metres.
float ao(vec3 P,vec3 n,ivec2 t,float R,float focal,float jit){
  float pr=max(R*focal/(-P.z),3.0); float sum=0.0; const int D=12,S=6;
  for(int i=0;i<D;i++){
    float a=(float(i)+jit)/float(D)*6.2831853; vec2 dir=vec2(cos(a),sin(a));
    for(int j=0;j<S;j++){
      float r=(float(j)+0.3+0.7*fract(jit*7.0+float(i)*0.37))/float(S)*pr;
      vec3 v=vp(t+ivec2(dir*r))-P; float d2=dot(v,v); if(d2<1e-9) continue;
      float co=dot(v,n)/sqrt(d2)-0.06;
      sum+=max(co,0.0)*max(1.0-d2/(R*R),0.0);
    }
  }
  return clamp(5.5*sum/float(D*S),0.0,1.0);
}
void main(){
  ivec2 t=ivec2(u*res); vec3 P=vp(t);
  if(-P.z>500.0){ c=vec4(0.0); return; }
  const int s=2;
  vec3 xp=vp(t+ivec2(s,0))-P, xn=P-vp(t-ivec2(s,0)), yp=vp(t+ivec2(0,s))-P, yn=P-vp(t-ivec2(0,s));
  vec3 dx=abs(xp.z)<abs(xn.z)?xp:xn; vec3 dy=abs(yp.z)<abs(yn.z)?yp:yn;
  vec3 n=normalize(cross(dx,dy)); if(n.z<0.0) n=-n;
  float focal=res.y/(2.0*tanv); float jit=hash(gl_FragCoord.xy);
  c=vec4(ao(P,n,t,radii.x,focal,jit),ao(P,n,t,radii.y,focal,fract(jit+0.5)),0.0,1.0);
}"""
BG_VS = """#version 330 core
const vec2 q[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1)); out vec2 u;
void main(){ u=q[gl_VertexID]*0.5+0.5; gl_Position=vec4(q[gl_VertexID],0.9999,1.0); }"""
BACKDROP = """
vec3 backdrop(vec2 u){
  float y=u.y; float g=mix(0.885,0.925,smoothstep(0.0,0.5,y)); g=mix(g,0.860,smoothstep(0.55,1.0,y));
  float x=abs(u.x-0.5)*2.0; g-=0.022*x*x;
  return vec3(g,g,g*1.002);
}"""
BG_FS = "#version 330 core\nin vec2 u; out vec4 c;" + BACKDROP + "\nvoid main(){ c=vec4(backdrop(u),1.0); }"
FLOOR_VS = """#version 330 core
layout(location=0) in vec3 p; uniform mat4 vp; out vec3 P; void main(){ P=p; gl_Position=vp*vec4(p,1.0); }"""
FLOOR_FS = "#version 330 core\nin vec3 P; out vec4 c; uniform vec2 res; uniform vec3 eye; uniform sampler2D mirrortex; uniform sampler2D floorshadow; uniform vec4 shadowbox; uniform float marks;" + BACKDROP + """
void main(){
  vec2 u=gl_FragCoord.xy/res; vec3 col=backdrop(u);
  vec2 su=(P.xy-shadowbox.xy)/shadowbox.zw+0.5;
  float near=1.0-smoothstep(1.2,2.9,length(P.xy-shadowbox.xy));
  // faint marks each half metre: the camera tracks the robot, and on a plain floor a walk would read as a treadmill
  if(marks>0.5){
    vec2 q=abs(fract(P.xy/0.5+0.5)-0.5)*0.5; vec2 fw=fwidth(P.xy)*1.2;
    vec2 ln=1.0-smoothstep(vec2(0.0012),vec2(0.0012)+fw,q);
    float m=max(ln.x*(1.0-smoothstep(0.014,0.014+fw.y,q.y)),ln.y*(1.0-smoothstep(0.014,0.014+fw.x,q.x)));
    col*=1.0-0.07*m*(1.0-smoothstep(2.0,5.5,length(P.xy-shadowbox.xy)));
  }
  float s=(su.x>0.0&&su.x<1.0&&su.y>0.0&&su.y<1.0)?texture(floorshadow,su).r:0.0;
  col*=1.0-s*near;
  vec3 vd=normalize(eye-P); float R=0.16+0.34*pow(1.0-max(vd.z,0.0),4.0);
  vec4 m=textureLod(mirrortex,u,1.6);
  col=col*(1.0-m.a*R)+m.rgb*R;
  c=vec4(col,1.0);
}"""


def srgb_to_linear(hex_color):
    h = hex_color.lstrip('#')
    v = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(v < 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def part_material(g, skeleton):
    """(material, linear colour, layer) for one spec part; None to leave it out."""
    role, name, col = g.get('role'), g['name'], g.get('color', '#808080')
    if role == 'cut' or 'crease' in name:  # the thigh creases lie half buried in the thigh and read as scratches; the seams do their job
        return None
    if skeleton == 'both' and role in SKELETON:
        skeleton = True
    elif skeleton == 'both':
        skeleton = False
    if skeleton:
        if role not in SKELETON:
            return None
        if name in ELECTRONICS:
            return GLOSS, np.array([0.02, 0.022, 0.026]), 0
        # a part's finish decides when the spec gives one (black anodised or silver); otherwise its role
        finish = str(g.get('finish') or '').lower()
        tint = METALS['actuator'] if 'black' in finish or 'anodi' in finish and 'clear' not in finish else METALS['struct'] if 'silver' in finish else METALS[role]
        return METAL, np.array(tint), 0
    if role not in ('knit', 'shell', 'decal'):
        # inside the garment: the covered view keeps them, so a gap shows what is really there, but dark so they never compete
        return (None if role == 'hardware' else (MATTE, srgb_to_linear(col) * 0.35, 0))
    layer = (3 if 'crease' in name or ' seam' in name else 1) if role == 'decal' else 0
    if name == 'visor':
        return VISOR, srgb_to_linear(col), layer
    if role == 'knit' or (name == 'head' and col.upper() == '#BCC0C4'):
        return KNIT, srgb_to_linear(col), layer
    if 'palm' in name or 'cuff' in name or 'wrist' in name or any(w in name for w in ('index', 'middle', 'ring', 'little', 'thumb')) and ' hub ' not in name:
        return RUBBER, srgb_to_linear(col), layer
    if col.upper() in ('#08090B', '#0A1420'):
        return GLOSS, srgb_to_linear(col), layer
    return MATTE, srgb_to_linear(col), layer


def load_parts(skeleton):
    """Every part in its body's frame: {body: Nx11 float array}, cached since the solids take a while. Body frames are world-aligned at zero."""
    root = Path(__file__).resolve().parents[1]
    key = hashlib.sha1((root / 'spec' / 'robot.json').read_bytes() + Path(__file__).read_bytes().split(b'def load_parts(')[0].split(b'def part_material(')[-1] + Path(__file__).read_bytes().split(b'def load_parts(')[1].split(b'def perspective(')[0]).hexdigest()[:12]
    cache = Path(tempfile.gettempdir()) / f'echo_quality_{key}_{skeleton}.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    spec = G.load()
    S = C.Solids(spec)
    m0 = mujoco.MjModel.from_xml_path(str(XML))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    out, vis, pid, count, groups, pieces = {}, None, 1, {}, {}, {}
    for (ln, i), solid in S.geom.items():
        g = spec['links'][ln]['geoms'][i]
        got = None if solid is None else part_material(g, skeleton)
        if got is None:
            continue
        mat, col, layer = got
        mesh = solid.to_mesh()
        v, f = np.asarray(mesh.vert_properties[:, :3], float) - d0.xpos[m0.body(ln).id], np.asarray(mesh.tri_verts, int)
        if len(f) < 4:
            continue
        s = trimesh.graph.smooth_shade(trimesh.Trimesh(v, f, process=True), angle=math.radians(40 if mat != VISOR else 60))  # keeps machined edges sharp
        tri, nor = s.vertices[s.faces].reshape(-1, 3), s.vertex_normals[s.faces].reshape(-1, 3)
        if mat == VISOR:
            vis = np.array([tri.min(0), tri.max(0)])
        group = g['name'].replace('knee gaiter', 'thigh').replace('elbow gaiter', 'upper arm').replace('upper arm lower', 'upper arm')
        seam_id = groups.setdefault(group, pid)
        # the id's fraction carries the depth layer: 0.1 per step
        arr = np.hstack((tri, nor, np.tile(col, (len(tri), 1)), np.full((len(tri), 1), mat), np.full((len(tri), 1), seam_id + layer * 0.1)))
        start = sum(len(x) for x in out.get(ln, []))
        pieces.setdefault(ln, []).append((start, len(arr), tri.mean(0), g.get('role') in ('knit', 'shell', 'decal'), float(np.linalg.norm(tri - tri.mean(0), axis=1).max()), g.get('role')))
        out.setdefault(ln, []).append(arr.astype(np.float32))
        count[g.get('role')] = count.get(g.get('role'), 0) + 1
        pid += 1
    res = {'parts': {k: np.vstack(v) for k, v in out.items()}, 'visor': vis, 'count': count, 'pieces': pieces}
    cache.write_bytes(pickle.dumps(res))
    return res


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


def ortho(cx, cy, cz, right, up, fwd, half, depth):
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = right / half, up / half, fwd / depth
    M[:3, 3] = -M[:3, :3] @ np.array([cx, cy, cz])
    return M


def prim_meshes():
    """Unit cylinder, cone and disc along +z, in the same vertex layout, for the arrows."""
    def layout(mesh):
        s = trimesh.graph.smooth_shade(mesh, angle=math.radians(50))
        tri, nor = s.vertices[s.faces].reshape(-1, 3), s.vertex_normals[s.faces].reshape(-1, 3)
        return np.hstack((tri, nor, np.ones((len(tri), 3)), np.full((len(tri), 1), FLAT), np.zeros((len(tri), 1)))).astype(np.float32)
    cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=20)
    cyl.apply_translation([0, 0, 0.5])
    cone = trimesh.creation.cone(radius=1.0, height=1.0, sections=20)
    return {'cyl': layout(cyl), 'cone': layout(cone)}


def compile_program(vs, fs):
    def sh(kind, src):
        s = gl.glCreateShader(kind)
        gl.glShaderSource(s, src)
        gl.glCompileShader(s)
        if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(s))
        return s
    p = gl.glCreateProgram()
    gl.glAttachShader(p, sh(gl.GL_VERTEX_SHADER, vs))
    gl.glAttachShader(p, sh(gl.GL_FRAGMENT_SHADER, fs))
    gl.glLinkProgram(p)
    if not gl.glGetProgramiv(p, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(p))
    return p


def rt(w, h, internal, depth=True, samples=0):
    fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    for att, fmt, want in ((gl.GL_COLOR_ATTACHMENT0, internal, True), (gl.GL_DEPTH_ATTACHMENT, gl.GL_DEPTH_COMPONENT32F, depth)):
        if not want:
            continue
        rb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rb)
        if samples:
            gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples, fmt, w, h)
        else:
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, fmt, w, h)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, att, gl.GL_RENDERBUFFER, rb)
    return fbo


_ROOMS, _ENVS = {}, {}


class Room:
    """tools/scene_living_room.py through MuJoCo's own renderer, from the same camera as the GL pass, with depth."""

    def __init__(self, W, H):
        import scene_living_room as SL
        ms = mujoco.MjSpec()
        SL.add_living_room(ms)
        ms.visual.global_.offwidth, ms.visual.global_.offheight = max(W, 2048), max(H, 2048)
        ms.visual.quality.offsamples = 8
        self.m = ms.compile()
        self.m.stat.extent = 15.0
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.r = mujoco.Renderer(self.m, H, W)
        self.W, self.H = W, H

    def _cam(self, eye, tgt):
        dv = np.asarray(tgt, float) - np.asarray(eye, float)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = tgt
        cam.distance = float(np.linalg.norm(dv))
        cam.azimuth = math.degrees(math.atan2(dv[1], dv[0]))
        cam.elevation = math.degrees(math.asin(dv[2] / np.linalg.norm(dv)))
        return cam

    def render(self, eye, tgt, fovy):
        self.m.vis.global_.fovy = fovy
        cam = self._cam(eye, tgt)
        self.r.update_scene(self.d, cam)
        col = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(self.d, cam)
        dep = self.r.render().copy()
        self.r.disable_depth_rendering()
        return col, dep

    def equirect(self, pos, n=256):
        """The room as seen from pos, 2n x n, from six square views. Linear, with the window light given back its range."""
        rr = mujoco.Renderer(self.m, n, n)
        self.m.vis.global_.fovy = 90.0
        faces = []
        for az, el in ((0, 0), (90, 0), (180, 0), (270, 0), (0, 89.9), (0, -89.9)):
            a, e = math.radians(az), math.radians(el)
            f = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            right = np.cross(f, [0, 0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, f)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = np.asarray(pos) + f
            cam.distance, cam.azimuth, cam.elevation = 1.0, az, el
            rr.update_scene(self.d, cam)
            faces.append((f, right, up, rr.render().astype(np.float32) / 255))
        rr.close()
        h, w = n, 2 * n
        lat = ((np.arange(h) + 0.5) / h - 0.5) * math.pi
        lon = ((np.arange(w) + 0.5) / w - 0.5) * 2 * math.pi
        lon, lat = np.meshgrid(lon, lat)
        D = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), -1)
        out = np.zeros((h, w, 3), np.float32)
        best = np.full((h, w), -2.0)
        for f, right, up, img in faces:
            z = D @ f
            take = z > best
            best = np.where(take, z, best)
            x = (D @ right) / np.maximum(z, 1e-4)
            y = (D @ up) / np.maximum(z, 1e-4)
            col = np.clip(((x + 1) / 2 * n).astype(int), 0, n - 1)
            row = np.clip(((1 - y) / 2 * n).astype(int), 0, n - 1)
            out[take] = img[row[take], col[take]]
        lin = out ** 2.2
        lum = lin.max(-1, keepdims=True)
        return lin * (1 + 5.0 * np.clip((lum - 0.55) / 0.45, 0, 1) ** 2)

    def close(self):
        self.r.close()


class Studio:
    def __init__(self, W, H, SS=2, skeleton=False, glow='teal', glow_gain=0.07, marks=True, room=None, envpos=(4.6, 0.0, 1.2)):
        self.W, self.H, self.SS, self.skeleton, self.marks = W, H, SS, skeleton, marks
        self.w, self.h = W * SS, H * SS
        # one MuJoCo room renderer per picture size and one room image per spot, shared by every Studio in the process
        if room:
            if (W, H) not in _ROOMS:
                _ROOMS[(W, H)] = Room(W, H)
            self.room = _ROOMS[(W, H)]
            key = tuple(np.round(envpos, 3))
            if key not in _ENVS:
                _ENVS[key] = self.room.equirect(envpos)
            self.env_img = _ENVS[key]
        else:
            self.room, self.env_img = None, None
        c = GLOWS.get(glow)
        if c is None:
            c = tuple(srgb_to_linear(glow))
        self.glow = np.array(c, float) * glow_gain
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        P = load_parts(skeleton)
        self.count = P['count']
        self.foot = {b: P['parts'][b][::3, :3].astype(float) for b in ('L_foot', 'R_foot')} if not skeleton or 'L_foot' in P['parts'] else {}
        vis = P['visor'] if P['visor'] is not None else np.zeros((2, 3))
        self.vmin, self.vmax = vis[0], vis[1]
        prims = prim_meshes()
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.win = glfw.create_window(64, 64, 'x', None, None)
        glfw.make_context_current(self.win)
        self.prog = compile_program(VS, FS)
        self.depth_prog = compile_program(VS.replace('out vec3 P; out vec3 N; out vec3 LP; out vec3 LN; out vec3 T; flat out int M; flat out float ID; flat out float LAYER;', 'vec3 P,N,LP,LN,T; int M; float ID,LAYER;').replace('LAYER*bias', '0.0'), SHADOW_FS)
        self.id_prog = compile_program(ID_VS, ID_FS)
        self.bg_prog = compile_program(BG_VS, BG_FS)
        self.floor_prog = compile_program(FLOOR_VS, FLOOR_FS)
        self.ao_prog = compile_program(BG_VS, AO_FS)
        chunks, self.ranges, off = [], {}, 0
        self.pieces = []   # (body, first vertex, count, centroid in the body frame, is a cover)
        for ln, arr in P['parts'].items():
            self.ranges[ln] = (off, len(arr))
            for o_, n_, c_, cov_, sz_, role_ in P['pieces'][ln]:
                self.pieces.append((ln, off + o_, n_, np.asarray(c_, float), cov_, sz_, role_))
            chunks.append(arr)
            off += len(arr)
        self.prim = {}
        for k, arr in prims.items():
            self.prim[k] = (off, len(arr))
            chunks.append(arr)
            off += len(arr)
        data = np.vstack(chunks)
        self.vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        stride = data.shape[1] * 4
        for loc, size, o in ((0, 3, 0), (1, 3, 3), (2, 3, 6), (3, 2, 9)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, False, stride, ctypes.c_void_p(o * 4))
        self.floor_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.floor_vao)
        fb = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, fb)
        self.floor_buf = fb
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 12 * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 12, None)
        self.empty_vao = gl.glGenVertexArrays(1)
        self.ms = rt(self.w, self.h, gl.GL_RGBA8, samples=4)
        self.res = rt(self.w, self.h, gl.GL_RGBA8, depth=False)
        self.idfbo = rt(self.w, self.h, gl.GL_RG32F, depth=True)
        self.idtex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.idtex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RG32F, self.w, self.h, 0, gl.GL_RG, gl.GL_FLOAT, None)
        for pn, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, pn, v)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.idfbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.idtex, 0)
        self.aofbo = rt(W, H, gl.GL_RG32F, depth=False)
        # shadow map from the key light
        self.SM = 2048
        self.smtex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.smtex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT32F, self.SM, self.SM, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE), (gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL),
                     (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
        self.smfbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.smfbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.smtex, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        # lowest height over the floor, from below
        self.FS_N = 512
        self.hfbo = rt(self.FS_N, self.FS_N, gl.GL_RGBA8)
        self.fstex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fstex)
        for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
        self.envtex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.envtex)
        for pn, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, pn, v)
        if self.env_img is not None:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, self.env_img.shape[1], self.env_img.shape[0], 0, gl.GL_RGB, gl.GL_FLOAT, self.env_img)
        else:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, 2, 1, 0, gl.GL_RGB, gl.GL_FLOAT, np.ones((1, 2, 3), np.float32))
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        # the mirrored robot, kept apart so the floor can take it soft
        self.mw, self.mh = W, H
        self.mirtex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mirtex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.mw, self.mh, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
        self.mirfbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.mirfbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.mirtex, 0)
        rb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32F, self.mw, self.mh)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, rb)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def pose_zero(self, bend=None):
        d, m = self.data, self.model
        d.qpos[:] = m.qpos0
        for j, q in (bend or {}).items():
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = q
        mujoco.mj_kinematics(m, d)
        low = min(((d.xmat[m.body(b).id].reshape(3, 3) @ self.foot[b].T).T[:, 2] + d.xpos[m.body(b).id][2]).min() for b in self.foot)
        d.qpos[2] -= low  # the soles on the floor
        mujoco.mj_kinematics(m, d)

    def place(self, x, y, heading):
        """Move the posed robot on the floor."""
        d, m = self.data, self.model
        d.qpos[0], d.qpos[1] = x, y
        d.qpos[3:7] = (math.cos(heading / 2), 0, 0, math.sin(heading / 2))
        mujoco.mj_kinematics(m, d)

    def bodies(self):
        out = []
        for ln, (o, n) in self.ranges.items():
            b = self.model.body(ln).id
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = self.data.xmat[b].reshape(3, 3), self.data.xpos[b]
            out.append((M, o, n))
        return out

    def _set_m4(self, prog, name, M):
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(prog, name), 1, True, np.asarray(M, np.float32))

    def _draw(self, prog, draws):
        gl.glBindVertexArray(self.vao)
        loc = gl.glGetUniformLocation(prog, 'model')
        rl = gl.glGetUniformLocation(prog, 'modelR')
        for M, o, n in draws:
            gl.glUniformMatrix4fv(loc, 1, True, np.asarray(M, np.float32))
            if rl >= 0:
                gl.glUniformMatrix3fv(rl, 1, True, np.asarray(M[:3, :3], np.float32))
            gl.glDrawArrays(gl.GL_TRIANGLES, o, n)

    def arrow_draws(self, a, b, width, rgba):
        """A thin shaft and head from a to b, as one flat colour."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-4:
            return []
        z = d / L
        x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        hl = min(L * 0.45, width * 5)

        def mk(p, r, h):
            M = np.eye(4)
            M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x * r, y * r, z * h, p
            return M
        return [(mk(a, width, L - hl), *self.prim['cyl'], rgba), (mk(a + z * (L - hl), width * 2.4, hl), *self.prim['cone'], rgba)]

    def disc_draw(self, p, r, rgba):
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = (r, 0, 0), (0, r, 0), (0, 0, 0.003), p
        return [(M, *self.prim['cyl'], rgba)]

    def body_mats(self):
        out = {}
        for ln in self.ranges:
            b = self.model.body(ln).id
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = self.data.xmat[b].reshape(3, 3), self.data.xpos[b]
            out[ln] = M
        return out

    def render(self, eye, tgt, fovy_deg, overlays=(), t=0.0, focus=None, draws=None):
        """draws: (model matrix, first vertex, count) to draw in place of the posed bodies, e.g. an exploded view.
        focus: in the room, the distance kept sharp; the room blurs away from it."""
        W, H, w, h = self.W, self.H, self.w, self.h
        self.focus = focus
        eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
        if self.room:
            room_col, room_dep = self.room.render(eye, tgt, fovy_deg)
        glfw.make_context_current(self.win)
        near, far = 0.05, 30.0
        V = look_at(eye, tgt)
        VP = perspective(math.radians(fovy_deg), W / H, near, far) @ V
        pts = np.array([M[:3, 3] for M, _, _ in self.bodies()])
        draws = self.bodies() if draws is None else draws
        cen = (pts.min(0) + pts.max(0)) / 2
        cam_az = math.atan2(eye[1] - tgt[1], eye[0] - tgt[0])
        # the studio moves with the camera, with a slow drift: the softboxes sweep across the glass even in a straight walk
        envyaw = cam_az + 0.30 * math.sin(0.55 * t)
        ka = cam_az + math.radians(62)
        key = np.array([math.cos(ka), math.sin(ka), 1.35])
        if self.room:
            key = np.array([0.75, 0.10, 0.62])   # the windows at the fireplace wall
            envyaw = 0.0
        key /= np.linalg.norm(key)
        # shadow map
        fwd = -key
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        half = 1.7 + 0.5 * np.ptp(pts, axis=0).max() * 0.5
        L = ortho(*cen, right, up, fwd, half, 4.0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.smfbo)
        gl.glViewport(0, 0, self.SM, self.SM)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.depth_prog)
        self._set_m4(self.depth_prog, 'vp', L)
        gl.glUniform1f(gl.glGetUniformLocation(self.depth_prog, 'mirror'), 0.0)
        self._draw(self.depth_prog, draws)
        # floor shadow: a view from under the floor gives each spot's lowest height, blurred tight for contact and wide for the soft pool
        box = 7.0 if self.room else 5.0
        cx, cy = float(cen[0]), float(cen[1])
        B = np.eye(4)
        B[0, 0], B[1, 1], B[0, 3], B[1, 3] = 2 / box, 2 / box, -2 * cx / box, -2 * cy / box
        B[2, 2], B[2, 3] = 2 / 3.0, -1.0  # z 0..3 m to depth; the nearest depth wins, so the lowest part
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.hfbo)
        gl.glViewport(0, 0, self.FS_N, self.FS_N)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT)
        self._set_m4(self.depth_prog, 'vp', B)
        self._draw(self.depth_prog, draws)
        zb = np.frombuffer(gl.glReadPixels(0, 0, self.FS_N, self.FS_N, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT), np.float32).reshape(self.FS_N, self.FS_N)
        hh = zb.astype(float) * 3.0
        hit = zb < 0.9999
        tex = 5.0 / box
        contact = ndimage.gaussian_filter(np.where(hit, np.exp(-hh / 0.05), 0.0), 2.2 * tex)
        mid = ndimage.gaussian_filter(np.where(hit, np.exp(-hh / 0.22), 0.0), 6.0 * tex)
        pool = ndimage.gaussian_filter(np.where(hit, np.exp(-hh / 0.9), 0.0), 16.0 * tex)
        sh = np.clip(0.42 * contact + 0.30 * mid + 0.34 * pool, 0, 0.62).astype(np.float32)
        sun = None
        if self.room:   # the window light's long shadow, thrown toward the room's near end
            Bs = B.copy()
            Bs[0, 2], Bs[1, 2] = -1.25 * 2 / box, -0.17 * 2 / box
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT)
            self._set_m4(self.depth_prog, 'vp', Bs)
            self._draw(self.depth_prog, draws)
            zs = np.frombuffer(gl.glReadPixels(0, 0, self.FS_N, self.FS_N, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT), np.float32).reshape(self.FS_N, self.FS_N)
            sun = ndimage.gaussian_filter((zs < 0.9999).astype(float), 5.0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fstex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, self.FS_N, self.FS_N, 0, gl.GL_RED, gl.GL_FLOAT, sh)
        # common uniforms
        p = self.prog
        gl.glUseProgram(p)
        gl.glUniform3fv(gl.glGetUniformLocation(p, 'eye'), 1, eye.astype(np.float32))
        gl.glUniform3fv(gl.glGetUniformLocation(p, 'key'), 1, key.astype(np.float32))
        gl.glUniform1f(gl.glGetUniformLocation(p, 'envyaw'), envyaw)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'bias'), 0.001)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'ss'), float(self.SS))
        gl.glUniform3fv(gl.glGetUniformLocation(p, 'glow'), 1, self.glow.astype(np.float32))
        gl.glUniform3fv(gl.glGetUniformLocation(p, 'vmin'), 1, np.asarray(self.vmin, np.float32))
        gl.glUniform3fv(gl.glGetUniformLocation(p, 'vmax'), 1, np.asarray(self.vmax, np.float32))
        gl.glUniform3f(gl.glGetUniformLocation(p, 'flatcol'), 1, 1, 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.smtex)
        gl.glUniform1i(gl.glGetUniformLocation(p, 'shadow'), 0)
        self._set_m4(p, 'lightvp', L)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'useroom'), 1.0 if self.room else 0.0)
        # the room's image is display-referred: scale its average to the fill level the studio uses
        gl.glUniform1f(gl.glGetUniformLocation(p, 'roomgain'), 0.40 / float(np.clip(self.env_img, 0, 1.5).mean()) if self.room else 1.0)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'keyk'), 0.55 if self.room else 1.0)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.envtex)
        gl.glUniform1i(gl.glGetUniformLocation(p, 'roomenv'), 2)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self._set_m4(p, 'vp', VP)
        # the mirrored robot: same real shading, seen from the real eye through the floor. eye is flipped by the shader's mirror branch.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.mirfbo)
        gl.glViewport(0, 0, self.mw, self.mh)
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'mirror'), 1.0)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'ss'), float(W) / self.mw * 1.0)
        if not self.room:
            self._draw(p, draws)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mirtex)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        # beauty
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ms)
        gl.glViewport(0, 0, w, h)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if not self.room:
            gl.glDepthMask(False)
            gl.glUseProgram(self.bg_prog)
            gl.glBindVertexArray(self.empty_vao)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
            gl.glDepthMask(True)
        # floor plane around the robot
        gl.glUseProgram(self.floor_prog)
        F = 30.0
        q = np.array([[cx - F, cy - F, 0], [cx + F, cy - F, 0], [cx - F, cy + F, 0], [cx + F, cy + F, 0]], np.float32)
        gl.glBindVertexArray(self.floor_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.floor_buf)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, q.nbytes, q)
        fp = self.floor_prog
        self._set_m4(fp, 'vp', VP)
        gl.glUniform2f(gl.glGetUniformLocation(fp, 'res'), float(w), float(h))
        gl.glUniform3fv(gl.glGetUniformLocation(fp, 'eye'), 1, eye.astype(np.float32))
        gl.glUniform4f(gl.glGetUniformLocation(fp, 'shadowbox'), cx, cy, box, box)
        gl.glUniform1f(gl.glGetUniformLocation(fp, 'marks'), 1.0 if self.marks else 0.0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mirtex)
        gl.glUniform1i(gl.glGetUniformLocation(fp, 'mirrortex'), 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fstex)
        gl.glUniform1i(gl.glGetUniformLocation(fp, 'floorshadow'), 1)
        if not self.room:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.smtex)
        # robot
        gl.glUseProgram(p)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'mirror'), 0.0)
        gl.glUniform1f(gl.glGetUniformLocation(p, 'ss'), float(self.SS))
        self._draw(p, draws)
        fc = gl.glGetUniformLocation(p, 'flatcol')
        for M, o, n, rgba in overlays:   # depth-tested: a leg in front hides them
            gl.glUniform3f(fc, *rgba)
            self._draw(p, [(M, o, n)])
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.ms)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.res)
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.res)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        rgba = np.frombuffer(gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE), np.uint8).reshape(h, w, 4)[::-1]
        rgb = rgba[..., :3]
        # part ids and true depth, one sample a pixel, for seams and contact shading
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.idfbo)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0, 1000.0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.id_prog)
        self._set_m4(self.id_prog, 'vp', VP)
        self._set_m4(self.id_prog, 'view', V)
        gl.glUniform3fv(gl.glGetUniformLocation(self.id_prog, 'eye'), 1, eye.astype(np.float32))
        self._draw(self.id_prog, draws)
        for M, o, n, _ in overlays:
            self._draw(self.id_prog, [(M, o, n)])
        idz = np.frombuffer(gl.glReadPixels(0, 0, w, h, gl.GL_RG, gl.GL_FLOAT), np.float32).reshape(h, w, 2)[::-1]
        ids, z = idz[..., 0], idz[..., 1]
        # ambient occlusion from that depth, at picture size, on the graphics card
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.aofbo)
        gl.glViewport(0, 0, W, H)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self.ao_prog)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.idtex)
        gl.glUniform1i(gl.glGetUniformLocation(self.ao_prog, 'gbuf'), 0)
        gl.glUniform2f(gl.glGetUniformLocation(self.ao_prog, 'res'), float(w), float(h))
        gl.glUniform1f(gl.glGetUniformLocation(self.ao_prog, 'tanv'), math.tan(math.radians(fovy_deg) / 2))
        gl.glUniform1f(gl.glGetUniformLocation(self.ao_prog, 'aspect'), W / H)
        gl.glUniform2f(gl.glGetUniformLocation(self.ao_prog, 'radii'), 0.022, 0.10)
        gl.glBindVertexArray(self.empty_vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        self.ao = np.frombuffer(gl.glReadPixels(0, 0, W, H, gl.GL_RG, gl.GL_FLOAT), np.float32).reshape(H, W, 2)[::-1]
        if self.room:
            # the room gets corners and creases too: MuJoCo's lights leave its walls flat
            rd = np.zeros((H, W, 2), np.float32)
            rd[..., 1] = room_dep[::-1]
            if not hasattr(self, 'roomdtex'):
                self.roomdtex = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.roomdtex)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.roomdtex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RG32F, W, H, 0, gl.GL_RG, gl.GL_FLOAT, rd)
            gl.glUniform2f(gl.glGetUniformLocation(self.ao_prog, 'res'), float(W), float(H))
            gl.glUniform2f(gl.glGetUniformLocation(self.ao_prog, 'radii'), 0.10, 0.70)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
            self.room_ao = np.frombuffer(gl.glReadPixels(0, 0, W, H, gl.GL_RG, gl.GL_FLOAT), np.float32).reshape(H, W, 2)[::-1]
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.smtex)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if self.room:
            return self._composite_room(rgb, rgba[..., 3], ids, z, np.linalg.norm(eye - tgt), fovy_deg, room_col, room_dep, eye, tgt, sh, sun, (cx, cy, box))
        return self._composite(rgb, ids, z, np.linalg.norm(eye - tgt), fovy_deg)

    def _composite_room(self, rgb, alpha, ids, z, dist, fovy, col, dep, eye, tgt, sh, sun, sbox):
        W, H = self.W, self.H
        # echo goes in front of the room wherever its depth is nearer. A rug's 3 mm is not allowed to cut the soles.
        zr = np.asarray(Image.fromarray(dep.astype(np.float32), 'F').resize((self.w, self.h), Image.BILINEAR))
        vis = (z < zr + 0.02)
        a = np.asarray(Image.fromarray((alpha * vis).astype(np.uint8)).resize((W, H), Image.LANCZOS), float)[..., None] / 255
        rob = np.asarray(self._composite(rgb * vis[..., None], ids, z, dist, fovy), float)
        # the floor under the robot, from the room's depth
        fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        tv = math.tan(math.radians(fovy) / 2)
        xs = ((np.arange(W) + 0.5) / W * 2 - 1) * tv * W / H
        ys = (1 - (np.arange(H) + 0.5) / H * 2) * tv
        xs, ys = np.meshgrid(xs, ys)
        Pw = eye + dep[..., None] * (fwd + xs[..., None] * right + ys[..., None] * up)
        cx, cy, box = sbox
        coords = np.stack((((Pw[..., 1] - cy) / box + 0.5) * self.FS_N, ((Pw[..., 0] - cx) / box + 0.5) * self.FS_N))
        on = np.abs(Pw[..., 2]) < 0.03
        s1 = ndimage.map_coordinates(sh.astype(np.float32), coords, order=1, cval=0.0) * on
        s2 = ndimage.map_coordinates(sun.astype(np.float32), coords, order=1, cval=0.0) * on
        room = (col.astype(float) / 255) ** 2.2
        room = room * np.array([1.03, 1.0, 0.95])                                   # warmer
        ra, rb = ndimage.gaussian_filter(self.room_ao[..., 0], 1.2), ndimage.gaussian_filter(self.room_ao[..., 1], 3.0)
        room = room * (1 - 0.65 * ra[..., None]) * (1 - 0.75 * rb[..., None])
        room = room * np.clip(1 - 1.5 * s1[..., None], 0.12, 1) * (1 - 0.42 * s2[..., None])
        # daylight is uneven: brighter toward the windows, falling off into the near end of the room, and the highlights bloom a little
        lit = np.clip(0.72 + 0.075 * Pw[..., 0], 0.62, 1.20)
        room = room * lit[..., None]
        hi = np.clip(room - 0.75, 0, None)
        room = room + 0.5 * ndimage.gaussian_filter(hi, (24 * H / 1000, 24 * H / 1000, 0))
        u, v = (np.arange(W) + 0.5) / W * 2 - 1, (np.arange(H) + 0.5) / H * 2 - 1
        u, v = np.meshgrid(u, v)
        room = room * (1 - 0.22 * np.clip(u * u + v * v - 0.25, 0, 1.6))[..., None]
        room = room ** 1.08 * 0.94
        room = 255 * np.clip(room, 0, 1) ** (1 / 2.2)
        if getattr(self, 'focus', None):
            # a lens: sharp at echo, the room softening with distance from that plane (circle of confusion in pixels)
            coc = np.clip(5.5 * W / 1080 * np.abs(1 - self.focus / np.maximum(dep, 0.05)), 0, 5.5 * W / 1080)
            sig = np.array([0.0, 1.0, 2.0, 3.4, 5.2]) * W / 1080
            stack = [room] + [ndimage.gaussian_filter(room, (s_, s_, 0)) for s_ in sig[1:]]
            pos = np.interp(coc, sig, np.arange(len(sig)))
            lo = np.floor(pos).astype(int)
            fr = (pos - lo)[..., None]
            hi = np.minimum(lo + 1, len(sig) - 1)
            S_ = np.stack(stack)
            yy, xx = np.indices(lo.shape)
            room = S_[lo, yy, xx] * (1 - fr) + S_[hi, yy, xx] * fr
        return Image.fromarray(np.clip(rob + (1 - a) * room, 0, 255).astype(np.uint8))

    def _composite(self, rgb, ids, z, dist, fovy):
        W, H, SS = self.W, self.H, self.SS
        im = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((W, H), Image.LANCZOS), float)
        mask = z < 500
        k = 4
        px_per_m = self.h / (2 * dist * math.tan(math.radians(fovy) / 2)) / k
        small = ndimage.gaussian_filter(self.ao[..., 0], 1.1)   # the jitter's noise, smoothed
        large = ndimage.gaussian_filter(self.ao[..., 1], 1.8)
        # seams: where one knit panel meets another at the same depth
        knit = ids > 0
        seam = np.zeros_like(knit)
        for ax in (0, 1):
            a0 = np.take(ids, range(0, ids.shape[ax] - 1), ax)
            a1 = np.take(ids, range(1, ids.shape[ax]), ax)
            z0 = np.take(z, range(0, z.shape[ax] - 1), ax)
            z1 = np.take(z, range(1, z.shape[ax]), ax)
            e = (a0 != a1) & (a0 > 0) & (a1 > 0) & (np.abs(z0 - z1) < 0.012)
            if ax == 0:
                seam[:-1] |= e
            else:
                seam[:, :-1] |= e
        s = np.asarray(Image.fromarray((seam * 255).astype(np.uint8)).resize((W, H), Image.BOX), float) / 255
        # panels that nearly coincide flip ids in specks: a real seam is a long line, so only long connected runs count
        lab, nlab = ndimage.label(ndimage.binary_dilation(s > 0, iterations=2))
        if nlab:
            longest = 0.09 * px_per_m * k * W / self.w  # 9 cm in picture pixels
            keep = np.zeros(nlab + 1, bool)
            for j, sl in enumerate(ndimage.find_objects(lab), 1):
                keep[j] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= longest
            s = s * keep[lab]
        s = np.clip(ndimage.gaussian_filter(s, 0.6) * 2.4, 0, 1)
        im = im * (1 - 0.82 * small[..., None]) * (1 - 0.60 * large[..., None]) * (1 - 0.40 * s[..., None])
        return Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))

    def close(self):
        glfw.destroy_window(self.win)


def still_camera(fx, fy, up, dist, tgt):
    d = np.array([fx, fy, up], float)
    d /= np.linalg.norm(d)
    tgt = np.asarray(tgt, float)
    return tgt + d * dist, tgt


def closeup_target(st, which):
    d, m = st.data, st.model
    if which == 'head':
        p = d.xpos[m.body('head').id].copy()
        return ('Head', (1.0, -0.75), 0.06, 0.62, (p[0] + 0.02, p[1], p[2] + 0.11))
    side = 'L' if which == 'knee' else which[0]
    p = d.xpos[m.body(f'{side}_knee_link').id].copy()
    return ('Knee', (1.0, -0.9), 0.1, 0.55, tuple(p))


def sheet(a):
    W, H = 900, 1500
    st = Studio(W, H, SS=3, skeleton=a.skeleton, glow=a.glow, glow_gain=a.glow_gain, marks=False)
    st.pose_zero(BEND)
    root = st.data.xpos[st.model.body('pelvis').id]
    views = [(n, f, u, dd, (root[0], root[1], zt)) for n, f, u, dd, zt in SHEET]
    views += [closeup_target(st, 'head'), closeup_target(st, 'knee')]
    out = Image.new('RGB', (W * len(views), H), (234, 234, 232))
    font = ImageFont.load_default(size=40)
    for k, (title, (fx, fy), up, dist, tgt) in enumerate(views):
        eye, tgt = still_camera(fx, fy, up, dist, tgt)
        im = st.render(eye, tgt, 45.0)
        ImageDraw.Draw(im).text((30, 24), title, fill=(50, 50, 55), font=font)
        out.paste(im, (k * W, 0))
    st.close()
    out.save(a.sheet)
    print(a.sheet, out.size, st.count)


def closeup(a):
    W, H = 1200, 1500
    st = Studio(W, H, SS=3, skeleton=a.skeleton, glow=a.glow, glow_gain=a.glow_gain, marks=False)
    st.pose_zero(BEND)
    _, (fx, fy), up, dist, tgt = closeup_target(st, a.closeup)
    eye, tgt = still_camera(fx, fy, up, dist, tgt)
    st.render(eye, tgt, 45.0, t=a.t).save(a.out)
    st.close()
    print(a.out)


ROOM_SPOT = (4.6, 0.05)


def knee_panels(a):
    """The right knee straight, part bent and fully bent, standing on the left leg: the gaps and folds where shadow matters."""
    W, H = 900, 1100
    st = Studio(W, H, SS=3, skeleton=a.skeleton, glow=a.glow, glow_gain=a.glow_gain, marks=False)
    out = Image.new('RGB', (W * 3, H))
    font = ImageFont.load_default(size=36)
    for k, ang in enumerate((0.0, 0.6, 2.15)):
        st.pose_zero({'R_knee': ang, 'R_hip_pitch': -ang / 2, 'R_ankle_pitch': -ang / 2, 'L_knee': 0.05, 'L_hip_pitch': -0.025, 'L_ankle_pitch': -0.025, 'L_elbow': -0.25, 'R_elbow': -0.25})
        p = st.data.xpos[st.model.body('R_knee_link').id]
        eye, tgt = still_camera(0.75, -1.0, 0.12, 0.72, p)
        im = st.render(eye, tgt, 45.0)
        ImageDraw.Draw(im).text((24, 20), f'Knee {ang:.2f} rad', fill=(50, 50, 55), font=font)
        out.paste(im, (k * W, 0))
    st.close()
    out.save(a.knees)
    print(a.knees)


def room_stills(a):
    """Echo standing on the rug: the camera at its front right, its back to the fireplace wall."""
    az = math.radians(194)
    heading = az + math.radians(45)
    px, py = ROOM_SPOT
    d = np.array([math.cos(az), math.sin(az)])
    st = None
    for name, (W, H), dist, eye_z, tz, fovy, ez in (('hero', (1600, 1000), 3.1, 1.35, 0.92, 45.0, 1.3), ('wide', (1920, 1080), 5.7, 1.3, 1.15, 45.0, 1.3),
                                                    ('portrait', (1080, 1920), 2.95, 1.35, 0.95, 45.0, 1.3), ('head', (1200, 1500), 0.8, 0.0, 0.0, 40.0, 1.5)):
        if a.only and name not in a.only.split(','):
            continue
        st = Studio(W, H, SS=3, glow=a.glow, glow_gain=a.glow_gain, room=a.room, envpos=(px, py, ez))
        st.pose_zero(BEND)
        st.place(px, py, heading)
        if name == 'head':
            tgt = st.data.xpos[st.model.body('head').id] + [0, 0, 0.08]
            eye = tgt + np.array([d[0] * dist, d[1] * dist, 0.03])
        else:
            tgt = np.array([px, py, tz])
            eye = np.array([px + d[0] * dist, py + d[1] * dist, eye_z])
        st.render(eye, tgt, fovy).save(f'{a.room_out}_{name}.png')
        st.close()
    print(a.room_out, 'done')


def chart_portrait(t, fz, weight, i, w, h, window=6.0, other=None):
    """The foot-force strip from render_forces.py, on a pale panel that matches the studio."""
    im = Image.new('RGB', (w, h), (247, 247, 246))
    dr = ImageDraw.Draw(im)
    big, small = ImageFont.load_default(size=38), ImageFont.load_default(size=28)
    dr.text((30, 18), f'Left foot {fz[i, 0]:4.0f} N', fill=RGB[0], font=big)
    dr.text((w // 2 + 10, 18), f'Right foot {fz[i, 1]:4.0f} N', fill=RGB[1], font=big)
    x0, x1, y0, y1 = 130, w - 30, h - 34, 86
    top = 1.8 * weight
    t1 = t[i]
    t0 = t1 - window * 0.8
    X = lambda v: x0 + (v - t0) / window * (x1 - x0)  # noqa: E731
    Y = lambda v: y0 + np.clip(v / top, 0, 1) * (y1 - y0)  # noqa: E731
    yw = Y(weight)
    dr.line([(x0, y0), (x1, y0)], fill=(120, 120, 120), width=2)
    dr.line([(x0, yw), (x1, yw)], fill=(90, 90, 90), width=2)
    dr.text((14, yw - 16), 'Weight', fill=(60, 60, 60), font=small)
    dr.text((66, y0 - 16), '0 N', fill=(60, 60, 60), font=small)
    lo = max(0, int(np.searchsorted(t, t0)))
    hi = int(np.searchsorted(t, min(t1, t0 + window)))
    for s in (0, 1):
        pts = [(float(X(t[k])), float(Y(fz[k, s]))) for k in range(lo, max(lo + 2, hi))]
        dr.line(pts, fill=RGB[s], width=5)
    dr.line([(X(t1), y0), (X(t1), y1)], fill=(30, 30, 30), width=2)
    if other is not None and other[i] > 20:
        dr.text((x0 + 10, 76), f'Body on floor {other[i]:4.0f} N', fill=(190, 30, 30), font=big)
    return im


def movie(a):
    d = np.load(a.rec, allow_pickle=True)
    t, q = d['t'], d['q']
    src = round(1 / (t[1] - t[0]))
    F, weight = d['force'], float(d['weight'])
    W_, H_, STRIP_ = 1080, 1560, 360
    root, qw = d['root_pos'].copy(), d['root_quat_wxyz'].copy()
    if a.room:
        # the clip's first frame moved to the walk's start in the room, turned to the walk's heading. The room is only a backdrop.
        k0 = int(a.start * round(1 / (t[1] - t[0])))
        x0, y0, h0 = (float(v) for v in a.walk.split(','))
        k1 = min(len(t) - 1, k0 + int((a.dur or 5.0) * round(1 / (t[1] - t[0]))))
        dy = math.radians(h0) - math.atan2(root[k1, 1] - root[k0, 1], root[k1, 0] - root[k0, 0])
        c_, s_ = math.cos(dy), math.sin(dy)
        p0 = root[k0, :2].copy()
        xy = root[:, :2] - p0
        root[:, 0], root[:, 1] = x0 + c_ * xy[:, 0] - s_ * xy[:, 1], y0 + s_ * xy[:, 0] + c_ * xy[:, 1]
        qz = np.array([math.cos(dy / 2), 0, 0, math.sin(dy / 2)])
        w0, v0 = qz[0], qz[1:]
        w1, v1 = qw[:, 0], qw[:, 1:]
        qw = np.column_stack((w0 * w1 - v1 @ v0, w0 * v1 + w1[:, None] * v0 + np.cross(v0, v1)))
        a.forces = False
    st = Studio(W_, H_, SS=2, skeleton=a.skeleton, glow=a.glow, glow_gain=a.glow_gain, room=a.room)
    m, data = st.model, st.data
    adr = [m.jnt_qposadr[m.joint(str(n)).id] for n in d['joint_names']]
    i0 = int(a.start * src)
    i1 = len(t) if a.dur is None else min(len(t), int((a.start + a.dur) * src))
    fps = 25
    frames = np.arange(i0, i1, max(1, round(src / fps)))
    az, el, _ = MJ_VIEWS['hero']
    yaw = np.unwrap(np.arctan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2)))
    heading = smooth_track(yaw[:, None], 1.0 * src)[:, 0]
    fovy = float(m.vis.global_.fovy)
    # The framing of render_forces.py: each frame's bodies, then a window that holds the distance before the body needs the room.
    # Added here: the captions get the top CAP of the picture to themselves, so the head never runs under the type.
    CAP, FOOT = 0.15, 0.05
    centre, need = np.zeros((len(t), 3)), np.zeros(len(t))
    tan_v = math.tan(math.radians(fovy / 2))
    for k in range(len(t)):
        data.qpos[:3], data.qpos[3:7] = root[k], qw[k]
        data.qpos[adr] = q[k]
        mujoco.mj_kinematics(m, data)
        x = data.xpos[1:]
        ang = math.radians(math.degrees(heading[k]) + az)
        side = x[:, 0] * -math.sin(ang) + x[:, 1] * math.cos(ang)
        centre[k] = (x.min(0) + x.max(0)) / 2
        top = x[:, 2].max() + 0.3  # body origins: the head's is at the neck, its crown about 30 cm above
        low = max(0.0, min(x[:, 2].min(), root[k, 2] - 0.1) - 0.1)
        centre[k, 2] = (low + top) / 2 + 0.05
        need[k] = max((top - low + 0.3) / (2 * tan_v) / (1 - CAP - FOOT), (side.max() - side.min() + 0.45) / (2 * tan_v * W_ / H_))
    win = int(0.5 * src)
    need = np.array([need[max(0, k - win):k + win + 1].max() for k in range(len(t))])
    need = smooth_track(np.maximum(need, 2.75)[:, None], 0.4 * src)[:, 0]
    centre = smooth_track(centre, 0.25 * src)
    centre[:, 2] += (CAP - FOOT) * need * tan_v
    w = imageio.get_writer(a.out, fps=fps, codec='libx264', quality=None, pixelformat='yuv420p', macro_block_size=8, ffmpeg_params=['-crf', '15', '-preset', 'veryfast', '-movflags', '+faststart'])
    big, small = ImageFont.load_default(size=46), ImageFont.load_default(size=30)
    has_contacts = 'contact_pos' in d.files
    for i in frames:
        data.qpos[:3], data.qpos[3:7] = root[i], qw[i]
        data.qpos[adr] = q[i]
        mujoco.mj_kinematics(m, data)
        over = []
        if a.forces and has_contacts:
            for p, f in zip(d['contact_pos'][i], d['contact_force'][i]):
                if np.isfinite(p).all():
                    over += st.disc_draw(p + [0, 0, 0.001], 0.016, (0.98, 0.45, 0.08))
                    if np.linalg.norm(f) > 2:
                        fn = np.linalg.norm(f)
                        over += st.arrow_draws(p, p + f / fn * min(fn / N_PER_M, ARROW_MAX), 0.0036, (0.82, 0.07, 0.06))
        neck_n = 0.0
        if 'shove_force' in d.files:
            back = d['shove_force'][max(0, i - int(0.6 * src)):i + 1]
            hit = np.flatnonzero(np.abs(back).sum(1) > 0)
            if len(hit):
                f = back[hit[-1]]
                neck_n = float(np.linalg.norm(f))
                tip = d['shove_point'][i]
                if a.forces:
                    over += st.arrow_draws(tip - f / neck_n * (0.04 + min(0.18 + neck_n / 2000.0, 0.5)), tip - f / neck_n * 0.04, 0.009, (0.05, 0.32, 0.85))
        ang = math.radians(math.degrees(heading[i]) + az)
        e = math.radians(el)
        fwd = np.array([math.cos(e) * math.cos(ang), math.cos(e) * math.sin(ang), math.sin(e)])
        eye, look = centre[i] - need[i] * fwd, centre[i]
        if a.room:   # a tripod at the room's near end, panning: a tracked camera would go through the wall
            eye, look = np.array([float(v) for v in a.eye.split(',')]), np.array([centre[i, 0], centre[i, 1], 0.92])
        img = st.render(eye, look, 45.0 if a.room else fovy, overlays=over, t=float(t[i]))
        if a.room:   # a pale panel so the captions read over the room
            ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).rounded_rectangle((20, 20, 860, 146 if not a.forces else 180), 14, fill=(245, 245, 243, 215))
            img = Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB')
        dr = ImageDraw.Draw(img)
        dr.text((40, 36), a.title, fill=(28, 30, 34), font=big)
        dr.text((40, 94), 'Physics sim: every step simulated', fill=(70, 74, 80), font=small)
        if a.forces:
            dr.text((40, 132), 'Red arrows: contact forces from the solver', fill=(70, 74, 80), font=small)
        if a.tag:
            x0, y0, x1, y1 = dr.textbbox((40, H_ - 62), a.tag, font=small)
            dr.rounded_rectangle((x0 - 14, y0 - 10, x1 + 14, y1 + 10), 10, fill=(44, 44, 40))
            dr.text((40, H_ - 62), a.tag, fill=(255, 215, 110), font=small)
        if neck_n:
            dr.text((40, 176), f'SHOVE {neck_n:.0f} N at the neck', fill=(16, 84, 200), font=big)
        strip = chart_portrait(t, F[..., 2], weight, i, W_, STRIP_, other=d['other_n'])
        frame = Image.new('RGB', (W_, H_ + STRIP_))
        frame.paste(img, (0, 0))
        frame.paste(strip, (0, H_))
        w.append_data(np.asarray(frame))
    w.close()
    st.close()
    print(a.out, len(frames), 'frames')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--covered', action='store_true', help='the garment on (default)')
    ap.add_argument('--skeleton', action='store_true', help='cover off, in polished metal')
    ap.add_argument('--sheet', help='write the 6-view sheet here')
    ap.add_argument('--closeup', choices=('head', 'knee'), help='one close-up, to --out')
    ap.add_argument('--rec', help='a train/replay_forces.py npz: renders a movie to --out')
    ap.add_argument('--out')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--title', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--forces', action=argparse.BooleanOptionalAction, default=True, help='contact-force arrows on the movie')
    ap.add_argument('--glow', default='teal', help='light behind the visor glass: teal, blue, green, off or #RRGGBB')
    ap.add_argument('--glow-gain', type=float, default=0.07)
    ap.add_argument('--t', type=float, default=0.0, help='studio time for a still')
    ap.add_argument('--knees', help='write three knee close-ups (straight, 0.6, 2.15 rad) here')
    ap.add_argument('--room', choices=('living',), help='the living room (tools/scene_living_room.py) in place of the studio; background only, not simulated')
    ap.add_argument('--room-out', help='with --room and no --rec: write the stills to <this>_hero.png, _wide, _portrait, _head')
    ap.add_argument('--only', help='room stills to make, comma separated')
    ap.add_argument('--walk', default='7.4,-0.7,180', help='room movie: where the clip starts, x,y,heading in degrees')
    ap.add_argument('--eye', default='-1.2,1.5,1.25', help='room movie: the tripod')
    a = ap.parse_args()
    if a.knees:
        knee_panels(a)
    elif a.room and not a.rec:
        room_stills(a)
    elif a.rec:
        movie(a)
    elif a.closeup:
        closeup(a)
    else:
        sheet(a)


if __name__ == '__main__':
    main()
