"""Browser-based (Three.js) 3D viewer for the tracking-target back-projection
result -- replaces `pipeline/viewer3d.py`'s Open3D window.

Why: `pipeline/mesh_utils.py` already fixed the mesh's *geometry* (crease-
angle-aware vertex normals, triangle-winding repair -- see that module's
docstring), but the Open3D legacy `draw_geometries()` window still showed a
faceted/banded-looking body because that renderer has no PBR materials and no
image-based lighting -- a handful of hard directional lights on a fixed-
function renderer looks flat next to a proper renderer no matter how correct
the normals feeding it are.

This project already has proof a different renderer fixes this:
`projection_tracking_targets - Copy/viewer_3d.html`, a Three.js viewer for
the sticker-fitting tool, renders the same kind of mesh with
`MeshStandardMaterial` plus a baked PMREM "studio environment" for image-
based lighting (see `_build_studio_environment_js` below) and its own
docstrings/history confirm this is what made it look "flawless" where
Open3D's viewer showed banding. This module ports that technique into a
purpose-built, self-contained (offline, no CDN) HTML viewer for this
project's back-projection results: the car mesh plus, per tracking target, a
purple sphere at the target's real-world measured position and a green
sphere at its back-projected estimate, with a thin gray line between them,
plus a blue camera marker with an RGB axis frame.

"Measured position" prefers the raw Creaform/C-Track reading
(`RayHit.raw_target_point`, before it's nudged onto the nearest mesh
surface) over the surface-fit ground truth (`RayHit.gt_point`) whenever a
raw measurement was supplied -- falling back to the surface-fit point only
for hits with none, so nothing just disappears. Note this makes the drawn
line a different (arguably more honest) comparison than the pipeline's own
`error_mm` metric, which is always measured against the surface-fit point,
not the raw one.

Deliberately keeps the mesh's native CAD-frame millimeter coordinates as-is
(no axis remap): the old tool's X/Y/Z swap was specific to its own BIP_Flat/
Three.js convention and would be an unverified new source of bugs here.
"""

import base64
import html
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def _read_vendor_js(filename: str) -> str:
    with open(os.path.join(_VENDOR_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def _mesh_to_binary(mesh: o3d.geometry.TriangleMesh) -> bytes:
    """Pack a mesh's (already mm-scaled, already crease-normal-shaded)
    vertices/normals/triangles into a compact binary blob for the browser to
    decode directly into a Three.js BufferGeometry -- no per-vertex JSON
    parsing, no re-deriving normals in JS (which would throw away the
    crease-aware normals and reintroduce the banding `mesh_utils.py` fixed).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.uint32)
    header = b"CAR3" + struct.pack("<II", len(verts), len(tris))
    return header + verts.tobytes() + normals.tobytes() + tris.tobytes()


def _finite_point_or_none(p):
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        return None
    return p.tolist()


def _build_scene_data(hits, camera_position, camera_rotation, sphere_radius_mm, camera_axis_len_mm) -> dict:
    targets = []
    for h in hits:
        estimated = _finite_point_or_none(h.estimated_point) if h.hit else None
        # Prefer the raw Creaform/C-Track measurement (the actual physical
        # reading) over the surface-fit ground truth (that same measurement
        # nudged onto the mesh surface -- see backproject_targets.py) when
        # both are available; fall back to the surface-fit point only for
        # hits with no raw measurement supplied, so nothing just disappears.
        target = _finite_point_or_none(getattr(h, "raw_target_point", None))
        if target is None:
            target = _finite_point_or_none(h.gt_point)
        if target is None and estimated is None:
            continue
        error_mm = (float(np.linalg.norm(np.array(target) - np.array(estimated)))
                    if target is not None and estimated is not None else None)
        targets.append({"index": h.point_line, "target": target, "estimated": estimated, "error_mm": error_mm})

    cam_pos = _finite_point_or_none(camera_position)
    cam_rot = None
    if cam_pos is not None and camera_rotation is not None:
        R = np.asarray(camera_rotation, dtype=np.float64)
        if R.shape == (3, 3) and np.all(np.isfinite(R)):
            cam_rot = R.tolist()

    return {
        "targets": targets,
        "sphere_radius_mm": float(sphere_radius_mm),
        "camera_axis_len_mm": float(camera_axis_len_mm),
        "camera_position": cam_pos,
        "camera_rotation": cam_rot,
    }


def _cad_alignment_line(hits) -> str:
    """One-line summary of the CAD-tracking-model alignment error -- i.e.
    the error computed upstream in `projection_tracking_targets`'
    fit_stickers_to_mesh.py, not this pipeline's own camera-calibration
    error_mm: how far each Creaform-scanned sticker sits from the CAD
    surface once the calibrated body->CAD transform is applied, before
    that raw scan position gets nudged onto the nearest mesh point to
    become `gt_point`. That per-target displacement is exactly
    `|raw_target_point - gt_point|` (see backproject_targets.run()'s
    `raw_points3d_path` docstring) -- computed fresh here from `hits`
    rather than threaded through from the caller, since every hit already
    carries both points.
    """
    displacements = []
    for h in hits:
        raw = _finite_point_or_none(getattr(h, "raw_target_point", None))
        gt = _finite_point_or_none(h.gt_point)
        if raw is None or gt is None:
            continue
        displacements.append(float(np.linalg.norm(np.array(raw) - np.array(gt))))

    if not displacements:
        return "CAD alignment (Creaform scan -> CAD surface): no raw scan data supplied (--raw-points3d)"

    arr = np.array(displacements)
    return (
        f"CAD alignment (Creaform scan -> CAD surface, {len(arr)} targets): "
        f"median={np.median(arr):.2f}mm mean={arr.mean():.2f}mm max={arr.max():.2f}mm"
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>__TITLE__</title>
  <style>
    html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
      background: #11141a; font-family: 'Segoe UI', Arial, sans-serif; color: #eceff4; }
    #header { position: absolute; top: 0; left: 0; right: 0; height: 74px;
      background: rgba(22, 26, 34, 0.95); display: flex; align-items: center;
      justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #2d3342; z-index: 10; }
    #title-block { display: flex; flex-direction: column; align-items: flex-start;
      justify-content: center; gap: 3px; overflow: hidden; }
    #title { font-weight: bold; font-size: 16px; color: #60a5fa; white-space: nowrap; }
    #summary { font-size: 12px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #alignment-summary { font-size: 12px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #legend { display: flex; gap: 16px; font-size: 12px; color: #cbd5e1; white-space: nowrap; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    #hint { font-size: 11px; color: #64748b; white-space: nowrap; }
    #canvas-wrap { position: absolute; top: 74px; left: 0; right: 0; bottom: 0; }
    canvas { display: block; width: 100%; height: 100%; }
    #controls { position: absolute; left: 16px; bottom: 16px; z-index: 10;
      background: rgba(22, 26, 34, 0.9); border: 1px solid #2d3342; border-radius: 8px;
      padding: 10px 14px; font-size: 12px; color: #cbd5e1; display: flex; flex-direction: column; gap: 8px; }
    #controls label { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
    #controls input[type="range"] { width: 120px; }
    #detail-panel { position: absolute; right: 16px; bottom: 16px; z-index: 10; max-width: 260px;
      background: rgba(22, 26, 34, 0.9); border: 1px solid #2d3342; border-radius: 8px;
      padding: 10px 14px; font-size: 12px; color: #cbd5e1; }
    #detail-panel .dp-title { font-weight: bold; color: #eceff4; margin-bottom: 4px; }
  </style>
</head>
<body>
  <div id="header">
    <div id="title-block">
      <span id="title">__TITLE__</span>
      <span id="summary">__SUMMARY__</span>
      <span id="alignment-summary">__ALIGNMENT_LINE__</span>
    </div>
    <div id="legend">
      <span><span class="swatch" style="background:#9933e6;"></span>Target (measured)</span>
      <span><span class="swatch" style="background:#00cc00;"></span>Estimate</span>
      <span><span class="swatch" style="background:#1a4dff;"></span>Camera</span>
      <span><span class="swatch" style="background:#e5e7eb;"></span>CAD origin</span>
    </div>
    <div id="hint">Rotate: Left drag&nbsp;&nbsp;|&nbsp;&nbsp;Pan: Right drag&nbsp;&nbsp;|&nbsp;&nbsp;Zoom: Wheel</div>
  </div>
  <div id="canvas-wrap"><canvas id="webgl-canvas"></canvas></div>
  <div id="controls">
    <label><input type="checkbox" id="hide-mesh-checkbox"> Hide car mesh</label>
    <label>Mesh opacity <input type="range" id="opacity-slider" min="0" max="100" value="100"></label>
    <label>Tag size <input type="range" id="tag-size-slider" min="0" max="300" value="100"></label>
  </div>
  <div id="detail-panel">Click a target or estimate point for its coordinates.</div>

  <script>__THREE_JS__</script>
  <script>__ORBIT_CONTROLS_JS__</script>
  <script>
    const CAR_MESH_B64 = "__MESH_B64__";
    const SCENE_DATA = __SCENE_JSON__;

    (function () {
      let scene, camera, renderer, controls;
      let carMesh, carMaterial;

      // Bakes a small procedural "studio room" (soft walls + a few colored
      // point lights) into a PMREM environment map -- the same trick glTF/
      // Babylon.js viewers use for a clean default look without shipping an
      // external HDRI. Gives the car's PBR material smooth, gradient
      // reflections instead of the flat/faceted look a handful of hard
      // directional lights gives a dense mesh (this is the actual fix for
      // the banding an Open3D window shows on the same mesh).
      function buildStudioEnvironment(renderer) {
        const pmremGenerator = new THREE.PMREMGenerator(renderer);
        pmremGenerator.compileEquirectangularShader();

        const envScene = new THREE.Scene();
        const roomGeo = new THREE.SphereGeometry(60, 32, 16);
        const roomMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8, side: THREE.BackSide, roughness: 1.0, metalness: 0.0,
        });
        envScene.add(new THREE.Mesh(roomGeo, roomMat));

        const key = new THREE.PointLight(0xffffff, 1.4, 140, 2);
        key.position.set(25, 45, 30);
        envScene.add(key);
        const fill = new THREE.PointLight(0x93c5fd, 0.7, 140, 2);
        fill.position.set(-35, 15, 20);
        envScene.add(fill);
        const rim = new THREE.PointLight(0x38bdf8, 0.5, 140, 2);
        rim.position.set(-15, -25, -35);
        envScene.add(rim);

        const envRT = pmremGenerator.fromScene(envScene, 0.035);
        pmremGenerator.dispose();
        return envRT.texture;
      }

      function decodeMesh(b64) {
        const binStr = window.atob(b64);
        const bytes = new Uint8Array(binStr.length);
        for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
        const buffer = bytes.buffer;
        const view = new DataView(buffer);
        const nVerts = view.getUint32(4, true);
        const nTris = view.getUint32(8, true);
        let offset = 12;
        const positions = new Float32Array(buffer, offset, nVerts * 3);
        offset += nVerts * 3 * 4;
        const normals = new Float32Array(buffer, offset, nVerts * 3);
        offset += nVerts * 3 * 4;
        const indices = new Uint32Array(buffer, offset, nTris * 3);
        return { positions, normals, indices };
      }

      // Car mesh, using the crease-aware normals decoded straight out of the
      // binary -- geometry.computeVertexNormals() must never be called here,
      // it would blend across real design creases and reintroduce banding.
      function addCarMesh() {
        const { positions, normals, indices } = decodeMesh(CAR_MESH_B64);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        geo.setIndex(new THREE.BufferAttribute(indices, 1));
        carMaterial = new THREE.MeshStandardMaterial({
          color: 0x64748b, metalness: 0.12, roughness: 0.75, side: THREE.DoubleSide, flatShading: false,
          transparent: false, opacity: 1.0,
        });
        carMesh = new THREE.Mesh(geo, carMaterial);
        scene.add(carMesh);
        geo.computeBoundingBox();
        return geo.boundingBox.clone();
      }

      // `transparent` starts false (matches the default, fully-opaque
      // slider position) and is only switched on below 100% -- leaving it
      // permanently true even at opacity 1.0 would move the mesh into
      // Three.js's back-to-front-sorted transparent render queue for no
      // reason, which is both slower and occasionally sorts oddly against
      // the target spheres.
      function setupControls() {
        const opacitySlider = document.getElementById('opacity-slider');
        const hideCheckbox = document.getElementById('hide-mesh-checkbox');
        const tagSizeSlider = document.getElementById('tag-size-slider');
        opacitySlider.addEventListener('input', () => {
          const opacity = parseInt(opacitySlider.value, 10) / 100;
          carMaterial.opacity = opacity;
          carMaterial.transparent = opacity < 1.0;
          carMaterial.needsUpdate = true;
        });
        hideCheckbox.addEventListener('change', () => {
          carMesh.visible = !hideCheckbox.checked;
        });
        tagSizeSlider.addEventListener('input', () => {
          const factor = parseInt(tagSizeSlider.value, 10) / 100;
          labelEntries.forEach(({ sprite, center }) => {
            sprite.scale.set(LABEL_BASE_W * factor, LABEL_BASE_H * factor, 1);
            sprite.position.set(center[0], center[1] + LABEL_OFFSET_MM * factor, center[2]);
          });
        });
      }

      function addSphere(center, radiusMm, colorHex, bbox) {
        const geo = new THREE.SphereGeometry(radiusMm, 20, 20);
        const mat = new THREE.MeshStandardMaterial({
          color: colorHex, emissive: colorHex, emissiveIntensity: 0.35, roughness: 0.3, metalness: 0.2,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(center[0], center[1], center[2]);
        scene.add(mesh);
        bbox.expandByPoint(mesh.position);
        return mesh;
      }

      function addLine(a, b, colorHex) {
        const geo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(a[0], a[1], a[2]),
          new THREE.Vector3(b[0], b[1], b[2]),
        ]);
        scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: colorHex })));
      }

      // Small always-on billboard showing just the target's index (its
      // points3d line number, shared by its target and estimate sphere) --
      // kept lightweight and permanently visible, unlike the full
      // coordinate readout below which only appears on click (with 90+
      // targets, always-on coordinate text for every point would be
      // unreadable clutter).
      const LABEL_BASE_W = 45;
      const LABEL_BASE_H = 22.5;

      function createIndexLabel(text, colorHex) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const colorCss = '#' + colorHex.toString(16).padStart(6, '0');
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.strokeStyle = colorCss;
        ctx.lineWidth = 5;
        const r = 16, x = 4, y = 4, w = 120, h = 56;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        // Shrink the font for longer labels (e.g. "CAD Origin") so they
        // still fit the same fixed-width canvas as a short "#12" tag.
        const fontPx = text.length > 4 ? Math.max(16, Math.floor(34 * 4 / text.length)) : 34;
        ctx.font = `bold ${fontPx}px "Segoe UI", Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 33);
        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        const mat = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false, transparent: true });
        const sprite = new THREE.Sprite(mat);
        sprite.scale.set(LABEL_BASE_W, LABEL_BASE_H, 1);
        return sprite;
      }

      function fmtCoord(p) {
        return `(${p[0].toFixed(1)}, ${p[1].toFixed(1)}, ${p[2].toFixed(1)}) mm`;
      }

      const LABEL_BASE_W = 45;
      const LABEL_BASE_H = 22.5;
      const LABEL_OFFSET_MM = 30;
      const clickableMeshes = []; // sphere meshes eligible for click-to-reveal-coordinates
      const labelEntries = []; // { sprite, center } for every tag -- lets the "Tag size" slider rescale/reposition them all together

      function addLabel(center, text, colorHex) {
        const label = createIndexLabel(text, colorHex);
        label.position.set(center[0], center[1] + LABEL_OFFSET_MM, center[2]);
        scene.add(label);
        labelEntries.push({ sprite: label, center });
      }

      function setupControls() {
        const hideCheckbox = document.getElementById('hide-mesh-checkbox');
        hideCheckbox.addEventListener('change', () => {
          if (carMesh) carMesh.visible = !hideCheckbox.checked;
        });

        const opacitySlider = document.getElementById('opacity-slider');
        opacitySlider.addEventListener('input', () => {
          if (carMaterial) {
            const val = parseInt(opacitySlider.value, 10) / 100;
            carMaterial.opacity = val;
            carMaterial.transparent = val < 1.0;
          }
        });

        const tagSlider = document.getElementById('tag-size-slider');
        tagSlider.addEventListener('input', () => {
          const factor = parseInt(tagSlider.value, 10) / 100;
          labelEntries.forEach(({ sprite, center }) => {
            sprite.scale.set(LABEL_BASE_W * factor, LABEL_BASE_H * factor, 1);
            sprite.position.set(center[0], center[1] + LABEL_OFFSET_MM * factor, center[2]);
          });
        });
      }

      function addTaggedSphere(center, radiusMm, colorHex, bbox, index, kind, errorMm) {
        const sphere = addSphere(center, radiusMm, colorHex, bbox);
        sphere.userData = { index, kind, coord: center, errorMm };
        clickableMeshes.push(sphere);
        addLabel(center, '#' + index, colorHex);
      }

      function addTargets(bbox) {
        const r = SCENE_DATA.sphere_radius_mm;
        SCENE_DATA.targets.forEach((t) => {
          if (t.target) addTaggedSphere(t.target, r, 0x9933e6, bbox, t.index, 'target', t.error_mm);
          if (t.estimated) {
            addTaggedSphere(t.estimated, r, 0x00cc00, bbox, t.index, 'estimate', t.error_mm);
            if (t.target) addLine(t.target, t.estimated, 0x999999);
          }
        });
      }

      // CAD frame origin (0,0,0) -- an RGB (X/Y/Z) axis frame plus a small
      // sphere and label, exactly like the target/camera markers, so users
      // can orient themselves relative to the mesh's own coordinate frame.
      function addCadOrigin(bbox) {
        const origin = [0, 0, 0];
        const size = Math.max(SCENE_DATA.sphere_radius_mm * 2, 15);
        addSphere(origin, size, 0xe5e7eb, bbox);
        const axisLen = SCENE_DATA.camera_axis_len_mm;
        addLine(origin, [axisLen, 0, 0], 0xff3b30);
        addLine(origin, [0, axisLen, 0], 0x34c759);
        addLine(origin, [0, 0, axisLen], 0x007aff);
        bbox.expandByPoint(new THREE.Vector3(axisLen, axisLen, axisLen));
        addLabel(origin, 'CAD Origin', 0xe5e7eb);
      }

      // Click-to-reveal: index labels stay on permanently, but the full
      // coordinate readout for a point only shows up in the detail panel
      // once you click that point -- see createIndexLabel's comment for why.
      function setupPicking() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const panel = document.getElementById('detail-panel');
        renderer.domElement.addEventListener('click', (ev) => {
          const rect = renderer.domElement.getBoundingClientRect();
          mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
          raycaster.setFromCamera(mouse, camera);
          const hit = raycaster.intersectObjects(clickableMeshes)[0];
          if (!hit) return;
          const { index, kind, coord, errorMm } = hit.object.userData;
          const label = kind === 'target' ? 'Target (measured)' : 'Estimate';
          const errorLine = errorMm == null
            ? 'Error vs. pair: n/a (no estimate for this target)'
            : `Error vs. pair: ${errorMm.toFixed(2)}mm`;
          panel.innerHTML = `<div class="dp-title">#${index} -- ${label}</div>${fmtCoord(coord)}<br>${errorLine}`;
        });
      }

      function addCameraMarker(bbox) {
        if (!SCENE_DATA.camera_position) return;
        const pos = SCENE_DATA.camera_position;
        addSphere(pos, SCENE_DATA.sphere_radius_mm * 1.5, 0x1a4dff, bbox);
        if (!SCENE_DATA.camera_rotation) return;

        const R = SCENE_DATA.camera_rotation; // row-major 3x3, world_from_cam
        const axisLen = SCENE_DATA.camera_axis_len_mm;
        const origin = new THREE.Vector3(pos[0], pos[1], pos[2]);
        const axisColors = [0xff3b30, 0x34c759, 0x007aff]; // camera X, Y, Z
        for (let col = 0; col < 3; col++) {
          const dir = new THREE.Vector3(R[0][col], R[1][col], R[2][col]).normalize();
          const end = origin.clone().addScaledVector(dir, axisLen);
          addLine([origin.x, origin.y, origin.z], [end.x, end.y, end.z], axisColors[col]);
          bbox.expandByPoint(end);
        }
      }

      function frameCamera(bbox) {
        if (bbox.isEmpty()) {
          bbox.set(new THREE.Vector3(-500, -500, -500), new THREE.Vector3(500, 500, 500));
        }
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        const radius = Math.max(size.x, size.y, size.z, 100) * 0.5;
        const dist = (radius / Math.tan((camera.fov * Math.PI / 180) / 2)) * 1.6;
        camera.position.set(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.6);
        camera.near = 1;
        camera.far = dist * 20 + radius * 4;
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
      }

      function onWindowResize() {
        const w = window.innerWidth, h = window.innerHeight - 74;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      }

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }

      function init() {
        const canvas = document.getElementById('webgl-canvas');
        const w = window.innerWidth, h = window.innerHeight - 74;

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(w, h);
        renderer.outputEncoding = THREE.sRGBEncoding;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x11141a);

        camera = new THREE.PerspectiveCamera(45, w / h, 1, 50000);
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        scene.environment = buildStudioEnvironment(renderer);

        scene.add(new THREE.AmbientLight(0xffffff, 0.12));
        const dir1 = new THREE.DirectionalLight(0xffffff, 0.55); dir1.position.set(2500, 4500, 3500); scene.add(dir1);
        const dir2 = new THREE.DirectionalLight(0x93c5fd, 0.25); dir2.position.set(-2500, 2000, 2500); scene.add(dir2);
        const dir3 = new THREE.DirectionalLight(0x38bdf8, 0.2); dir3.position.set(-1500, 1000, -3500); scene.add(dir3);
        const dirBounce = new THREE.DirectionalLight(0x334155, 0.2); dirBounce.position.set(0, -3000, 0); scene.add(dirBounce);

        const bbox = addCarMesh();
        addTargets(bbox);
        addCameraMarker(bbox);
        addCadOrigin(bbox);
        frameCamera(bbox);
        setupControls();
        setupPicking();

        window.addEventListener('resize', onWindowResize);
        animate();
      }

      init();
    })();
  </script>
</body>
</html>
"""


def build_viewer_html(
    mesh: o3d.geometry.TriangleMesh,
    hits,
    camera_position=None,
    camera_rotation=None,
    sphere_radius_mm: float = 3.0,
    camera_axis_len_mm: float = 100.0,
    window_name: str = "Tracking-Target Back-Projection",
    summary_line: str = "",
) -> str:
    """Build the full, self-contained HTML document text for one
    back-projection run's browser viewer.

    Args:
        mesh: car mesh, already scaled to millimeters (e.g. from
            `mesh_utils.load_car_mesh`).
        hits: list of `camera_model.RayHit`.
        camera_position: (3,) mm, world/CAD frame; draws a camera marker if given.
        camera_rotation: (3,3) world_from_cam rotation; draws an axis frame if given.
        summary_line: short plain-text status line shown next to the title
            (e.g. hit-rate/RMS), typically built from the same report dict
            `backproject_targets.run()` returns.
    """
    mesh_b64 = base64.b64encode(_mesh_to_binary(mesh)).decode("ascii")
    scene_data = _build_scene_data(hits, camera_position, camera_rotation, sphere_radius_mm, camera_axis_len_mm)
    alignment_line = _cad_alignment_line(hits)

    doc = _HTML_TEMPLATE
    doc = doc.replace("__TITLE__", html.escape(window_name))
    doc = doc.replace("__SUMMARY__", html.escape(summary_line))
    doc = doc.replace("__ALIGNMENT_LINE__", html.escape(alignment_line))
    doc = doc.replace("__THREE_JS__", _read_vendor_js("three.min.js"))
    doc = doc.replace("__ORBIT_CONTROLS_JS__", _read_vendor_js("OrbitControls.js"))
    doc = doc.replace("__MESH_B64__", mesh_b64)
    doc = doc.replace("__SCENE_JSON__", json.dumps(scene_data))
    return doc


def show(
    mesh: o3d.geometry.TriangleMesh,
    hits,
    camera_position=None,
    camera_rotation=None,
    sphere_radius_mm: float = 3.0,
    camera_axis_len_mm: float = 100.0,
    out_path: str = None,
    window_name: str = "Tracking-Target Back-Projection",
    summary_line: str = "",
) -> str:
    """Write the viewer HTML (to `out_path`, or a fresh temp file if not
    given) and open it in the system's default browser. Returns the path
    written to.
    """
    doc = build_viewer_html(
        mesh, hits,
        camera_position=camera_position,
        camera_rotation=camera_rotation,
        sphere_radius_mm=sphere_radius_mm,
        camera_axis_len_mm=camera_axis_len_mm,
        window_name=window_name,
        summary_line=summary_line,
    )

    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="backprojection_viewer_", suffix=".html")
        os.close(fd)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    import webbrowser
    webbrowser.open(Path(out_path).resolve().as_uri())
    return out_path
