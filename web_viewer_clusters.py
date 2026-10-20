"""Self-contained web viewer for combined and per-camera clustered points."""
from __future__ import annotations

import json
import base64
import os
import tempfile
from pathlib import Path

from live_backprojection import _pathsetup  # noqa: F401
from single_frame_backprojection.web_viewer import _mesh_to_binary, _read_vendor_js


def build_cluster_viewer_html(mesh, cluster_outputs: dict, window_name="Clustered defects") -> str:
    data = {}
    combined = []
    for camera, result in cluster_outputs.items():
        points = []
        for cluster in result.get("clusters", []):
            p = cluster.get("point_mm", [0, 0, 0])
            item = {"camera": camera, "cluster_id": cluster.get("cluster_id"),
                    "point": p, "n": cluster.get("n_observations", 0),
                    "spread": cluster.get("spread_mm", 0)}
            points.append(item); combined.append(item)
        data[camera] = points
    data = {"All cameras": combined, **data}
    mesh_b64 = base64.b64encode(_mesh_to_binary(mesh)).decode("ascii")
    three = _read_vendor_js("three.min.js")
    controls = _read_vendor_js("OrbitControls.js")
    template = """<!doctype html><html><head><meta charset='utf-8'><title>__TITLE__</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#11141a;color:#e5e7eb;font:13px Segoe UI,Arial}
#bar{height:54px;background:#161a22;border-bottom:1px solid #2d3342;display:flex;align-items:center;gap:14px;padding:0 18px}
#canvas{position:absolute;top:54px;bottom:0;left:0;right:0}canvas{width:100%;height:100%;display:block}
select{background:#222936;color:#e5e7eb;border:1px solid #465166;padding:5px}.info{color:#94a3b8}</style></head>
<body><div id='bar'><b>__TITLE__</b><label>Camera <select id='camera'></select></label><span id='info' class='info'></span></div>
<div id='canvas'><canvas id='view'></canvas></div><script>__THREE__</script><script>__CONTROLS__</script>
<script>const MESH='__MESH__';const DATA=__DATA__;
let scene,camera,renderer,orbit,meshObj,points=[];const canvas=document.getElementById('view');
function buildStudioEnvironment(renderer){const pmrem=new THREE.PMREMGenerator(renderer);pmrem.compileEquirectangularShader();const envScene=new THREE.Scene();const room=new THREE.Mesh(new THREE.SphereGeometry(60,32,16),new THREE.MeshStandardMaterial({color:0x94a3b8,side:THREE.BackSide,roughness:1,metalness:0}));envScene.add(room);const key=new THREE.PointLight(0xffffff,1.4,140,2);key.position.set(25,45,30);envScene.add(key);const fill=new THREE.PointLight(0x93c5fd,.7,140,2);fill.position.set(-35,15,20);envScene.add(fill);const rim=new THREE.PointLight(0x38bdf8,.5,140,2);rim.position.set(-15,-25,-35);envScene.add(rim);const rt=pmrem.fromScene(envScene,.035);pmrem.dispose();return rt.texture;}
function decodeMesh(b64){const bin=atob(b64),bytes=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);const buffer=bytes.buffer,view=new DataView(buffer),nVerts=view.getUint32(4,true),nTris=view.getUint32(8,true);let offset=12;const positions=new Float32Array(buffer,offset,nVerts*3);offset+=nVerts*3*4;const normals=new Float32Array(buffer,offset,nVerts*3);offset+=nVerts*3*4;const indices=new Uint32Array(buffer,offset,nTris*3);return {positions,normals,indices};}
function addCarMesh(){const d=decodeMesh(MESH),g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(d.positions,3));g.setAttribute('normal',new THREE.BufferAttribute(d.normals,3));g.setIndex(new THREE.BufferAttribute(d.indices,1));meshObj=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x64748b,metalness:.12,roughness:.75,side:THREE.DoubleSide,flatShading:false}));scene.add(meshObj);g.computeBoundingBox();return g.boundingBox.clone();}
function init(){const w=window.innerWidth,h=window.innerHeight-54;renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(window.devicePixelRatio);renderer.setSize(w,h);renderer.outputEncoding=THREE.sRGBEncoding;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;scene=new THREE.Scene();scene.background=new THREE.Color(0x11141a);camera=new THREE.PerspectiveCamera(45,w/h,1,50000);orbit=new THREE.OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.dampingFactor=.08;scene.environment=buildStudioEnvironment(renderer);scene.add(new THREE.AmbientLight(0xffffff,.12));let d1=new THREE.DirectionalLight(0xffffff,.55);d1.position.set(2500,4500,3500);scene.add(d1);let d2=new THREE.DirectionalLight(0x93c5fd,.25);d2.position.set(-2500,2000,2500);scene.add(d2);let d3=new THREE.DirectionalLight(0x38bdf8,.2);d3.position.set(-1500,1000,-3500);scene.add(d3);let db=new THREE.DirectionalLight(0x334155,.2);db.position.set(0,-3000,0);scene.add(db);const bbox=addCarMesh(),center=bbox.getCenter(new THREE.Vector3()),size=bbox.getSize(new THREE.Vector3());camera.position.set(center.x+size.x*1.4,center.y-size.y*1.4,center.z+size.z*1.1);camera.lookAt(center);orbit.target.copy(center);orbit.update();window.onresize=resize;resize();}
function resize(){renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix();}
function draw(name){points.forEach(p=>scene.remove(p));points=[];let rows=DATA[name]||[];rows.forEach(x=>{let s=new THREE.Mesh(new THREE.SphereGeometry(7,20,20),new THREE.MeshStandardMaterial({color:0x00cc66,emissive:0x00cc66,emissiveIntensity:.35,roughness:.3,metalness:.2}));s.position.set(...x.point);scene.add(s);points.push(s)});document.getElementById('info').textContent=rows.length+' clusters';}
function animate(){requestAnimationFrame(animate);orbit.update();renderer.render(scene,camera)}
init();let sel=document.getElementById('camera');Object.keys(DATA).forEach(k=>{let o=document.createElement('option');o.textContent=k;o.value=k;sel.appendChild(o)});sel.onchange=()=>draw(sel.value);draw('All cameras');animate();
</script></body></html>"""
    return (template.replace("__TITLE__", window_name).replace("__THREE__", three)
            .replace("__CONTROLS__", controls).replace("__MESH__", mesh_b64)
            .replace("__DATA__", json.dumps(data)))


def show_cluster_results(mesh, cluster_outputs: dict, out_path: str | None = None, **kwargs) -> str:
    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="cluster_viewer_", suffix=".html"); os.close(fd)
    Path(out_path).write_text(build_cluster_viewer_html(mesh, cluster_outputs, **kwargs), encoding="utf-8")
    return out_path

