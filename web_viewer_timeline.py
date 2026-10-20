"""Stage 3: per-frame web viewer.

Extends pipeline/web_viewer.py's established Three.js / self-contained-HTML
pattern (same dark theme, same binary mesh embedding, same vendored
three.min.js/OrbitControls.js, same studio-environment lighting) with a
frame-scrubber: the car mesh is loaded/embedded once, and a slider swaps
which frame's detected-point spheres + camera marker are shown.

No frame-scrubbing concept exists anywhere else in this project yet -- this
is new UI built strictly on the existing rendering/embedding techniques
(pipeline/web_viewer.py), not a new tech stack, per the project's own
"strictly use the same verified method" requirement for the mesh side, and
"like the ones previously in the folder" for the viewer style.
"""
import base64
import html
import json
import os
import tempfile
from pathlib import Path

from live_backprojection import _pathsetup  # noqa: F401

import numpy as np

from single_frame_backprojection.web_viewer import _mesh_to_binary, _read_vendor_js

# The complete tracking model (with hood) -- final model for auto threshold + manual + non-web GUI.
TRACKING_MODEL_PATH = os.path.join(_pathsetup.PROJECT_ROOT, "data", "new_car_300926_whood.txt")


def _load_tracking_model_points(path: str = TRACKING_MODEL_PATH):
    """Returns a list of {"point_line": int, "point_mm": [x,y,z]} for every
    row in the tracking-model points3d file (1-indexed line = point_line,
    only the first 3 columns -- X Y Z in mm -- are used)."""
    raw = np.loadtxt(path)
    points_mm = raw[:, :3]
    return [{"point_line": i + 1, "point_mm": points_mm[i].tolist()} for i in range(len(points_mm))]


def _build_timeline_scene_data(stage2_output: dict, sphere_radius_mm: float = 3.0,
                                camera_axis_len_mm: float = 100.0,
                                tracking_model_path: str = TRACKING_MODEL_PATH) -> dict:
    """Turns Stage 2's per-frame hit records into the JS-side SCENE_DATA: a
    frame_id-sorted list, each with its own hit points (green "detected
    point" spheres -- no ground-truth target sphere/comparison line exists
    for a live detection) and its own camera marker. Misses (hit=false,
    point_mm=null) are filtered out here so the JS side never needs a
    null-check per point. Also includes the static (frame-independent)
    tracking-model reference points, shown via an optional toggle."""
    frames = []
    for f in sorted(stage2_output["frames"], key=lambda fr: fr["frame_id"]):
        points = [h["point_mm"] for h in f["hits"] if h["hit"]]
        frames.append({
            "frame_id": f["frame_id"],
            "frame_name": f["frame_name"],
            "points": points,
            "num_hit": len(points),
            "num_total": len(f["hits"]),
            "camera_position": f["camera_position_world_mm"],
            "camera_rotation": f["camera_rotation_world_cam"],
        })
    tracking_model_points = _load_tracking_model_points(tracking_model_path) if tracking_model_path else []
    return {
        "frames": frames,
        "sphere_radius_mm": float(sphere_radius_mm),
        "camera_axis_len_mm": float(camera_axis_len_mm),
        "tracking_model_points": tracking_model_points,
    }


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
    #legend { display: flex; gap: 16px; font-size: 12px; color: #cbd5e1; white-space: nowrap; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    #hint { font-size: 11px; color: #64748b; white-space: nowrap; }
    #canvas-wrap { position: absolute; top: 74px; left: 0; right: 0; bottom: 0; }
    canvas { display: block; width: 100%; height: 100%; }
    #controls { position: absolute; left: 16px; bottom: 16px; z-index: 10;
      background: rgba(22, 26, 34, 0.9); border: 1px solid #2d3342; border-radius: 8px;
      padding: 10px 14px; font-size: 12px; color: #cbd5e1; display: flex; flex-direction: column; gap: 8px; }
    #controls label { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
    #controls input[type="range"] { width: 160px; }
    #frame-readout { color: #94a3b8; }
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
    </div>
    <div id="legend">
      <span><span class="swatch" style="background:#00cc00;"></span>Detected point</span>
      <span><span class="swatch" style="background:#ff8c00;"></span>Tracking model (reference)</span>
      <span><span class="swatch" style="background:#1a4dff;"></span>Camera</span>
      <span><span class="swatch" style="background:#e5e7eb;"></span>CAD origin</span>
    </div>
    <div id="hint">Rotate: Left drag&nbsp;&nbsp;|&nbsp;&nbsp;Pan: Right drag&nbsp;&nbsp;|&nbsp;&nbsp;Zoom: Wheel</div>
  </div>
  <div id="canvas-wrap"><canvas id="webgl-canvas"></canvas></div>
  <div id="controls">
    <label><input type="checkbox" id="hide-mesh-checkbox"> Hide car mesh</label>
    <label>Mesh opacity <input type="range" id="opacity-slider" min="0" max="100" value="100"></label>
    <label>Frame <input type="range" id="frame-slider" min="0" max="0" value="0" step="1">
      <span id="frame-readout"></span></label>
    <label><input type="checkbox" id="play-checkbox"> Auto-play (10fps)</label>
    <label><input type="checkbox" id="tracking-model-checkbox"> Show tracking model (reference)</label>
    <label>Detected sphere size <input type="range" id="estimate-sphere-slider" min="0" max="300" value="100"></label>
    <label>Detected tag size <input type="range" id="estimate-tag-slider" min="0" max="300" value="100"></label>
    <label>Tracking model sphere size <input type="range" id="tracking-sphere-slider" min="0" max="300" value="100"></label>
    <label>Tracking model tag size <input type="range" id="tracking-tag-slider" min="0" max="300" value="100"></label>
  </div>
  <div id="detail-panel">Click a detected point for its coordinates.</div>

  <script>__THREE_JS__</script>
  <script>__ORBIT_CONTROLS_JS__</script>
  <script>
    const CAR_MESH_B64 = "__MESH_B64__";
    const SCENE_DATA = __SCENE_JSON__;

    (function () {
      let scene, camera, renderer, controls;
      let carMesh, carMaterial;

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

      function setupStaticControls() {
        const opacitySlider = document.getElementById('opacity-slider');
        const hideCheckbox = document.getElementById('hide-mesh-checkbox');
        opacitySlider.addEventListener('input', () => {
          const opacity = parseInt(opacitySlider.value, 10) / 100;
          carMaterial.opacity = opacity;
          carMaterial.transparent = opacity < 1.0;
          carMaterial.needsUpdate = true;
        });
        hideCheckbox.addEventListener('change', () => {
          carMesh.visible = !hideCheckbox.checked;
        });
      }

      function addSphere(center, radiusMm, colorHex, bbox) {
        const geo = new THREE.SphereGeometry(radiusMm, 20, 20);
        const mat = new THREE.MeshStandardMaterial({
          color: colorHex, emissive: colorHex, emissiveIntensity: 0.35, roughness: 0.3, metalness: 0.2,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(center[0], center[1], center[2]);
        if (bbox) bbox.expandByPoint(mesh.position);
        return mesh;
      }

      function addLine(a, b, colorHex) {
        const geo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(a[0], a[1], a[2]),
          new THREE.Vector3(b[0], b[1], b[2]),
        ]);
        return new THREE.Line(geo, new THREE.LineBasicMaterial({ color: colorHex }));
      }

      const LABEL_BASE_W = 45;
      const LABEL_BASE_H = 22.5;
      const LABEL_OFFSET_MM = 30;
      const labelEntries = []; // { sprite, center } for the CAD-origin label only (permanent, fixed size --
                                 // not adjustable by the size sliders below, which only cover
                                 // detected/tracking-model points)

      // Independent size factors (1.0 = 100%, the baked-in base size) for the two adjustable
      // point groups. Persisted here (not just read off the slider at creation time) so a
      // frame switch (which rebuilds the detected-point spheres/labels from scratch) keeps
      // whatever size the user last set, rather than resetting to 100%.
      let estimateSphereFactor = 1.0;
      let estimateTagFactor = 1.0;
      let trackingSphereFactor = 1.0;
      let trackingTagFactor = 1.0;

      function applyLabelTransform(sprite, factor) {
        const center = sprite.userData.center;
        sprite.scale.set(LABEL_BASE_W * factor, LABEL_BASE_H * factor, 1);
        sprite.position.set(center[0], center[1] + LABEL_OFFSET_MM * factor, center[2]);
      }

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

      // ---- Per-frame content: rebuilt on every slider move ----
      let currentFrameIdx = 0;
      let frameSpheres = [];   // detected-point spheres for the CURRENT frame only
      let frameLabels = [];    // matching index labels
      let cameraMarkerGroup = null; // THREE.Group holding this frame's camera sphere + RGB axis
      const clickableMeshes = []; // spheres eligible for click-to-reveal; mutated per frame switch

      function clearFrameObjects() {
        frameSpheres.forEach((s) => {
          scene.remove(s);
          const idx = clickableMeshes.indexOf(s);
          if (idx !== -1) clickableMeshes.splice(idx, 1);
        });
        frameLabels.forEach((l) => scene.remove(l));
        frameSpheres = [];
        frameLabels = [];
        if (cameraMarkerGroup) {
          scene.remove(cameraMarkerGroup);
          cameraMarkerGroup = null;
        }
      }

      function addCameraMarkerInto(group, pos, rot, bbox) {
        if (!pos) return;
        const sphere = addSphere(pos, SCENE_DATA.sphere_radius_mm * 1.5, 0x1a4dff, bbox);
        group.add(sphere);
        if (!rot) return;
        const axisLen = SCENE_DATA.camera_axis_len_mm;
        const origin = new THREE.Vector3(pos[0], pos[1], pos[2]);
        const axisColors = [0xff3b30, 0x34c759, 0x007aff];
        for (let col = 0; col < 3; col++) {
          const dir = new THREE.Vector3(rot[0][col], rot[1][col], rot[2][col]).normalize();
          const end = origin.clone().addScaledVector(dir, axisLen);
          group.add(addLine([origin.x, origin.y, origin.z], [end.x, end.y, end.z], axisColors[col]));
          if (bbox) bbox.expandByPoint(end);
        }
      }

      function showFrame(idx) {
        currentFrameIdx = idx;
        clearFrameObjects();
        const f = SCENE_DATA.frames[idx];
        const r = SCENE_DATA.sphere_radius_mm;

        f.points.forEach((p, i) => {
          const sphere = addSphere(p, r, 0x00cc00, null);
          sphere.scale.setScalar(estimateSphereFactor);
          scene.add(sphere);
          sphere.userData = { coord: p };
          frameSpheres.push(sphere);
          clickableMeshes.push(sphere);

          const label = createIndexLabel('#' + (i + 1), 0x00cc00);
          label.userData = { center: p };
          applyLabelTransform(label, estimateTagFactor);
          scene.add(label);
          frameLabels.push(label);
        });

        cameraMarkerGroup = new THREE.Group();
        scene.add(cameraMarkerGroup);
        addCameraMarkerInto(cameraMarkerGroup, f.camera_position, f.camera_rotation, null);

        document.getElementById('frame-readout').textContent =
          `${f.frame_name} (${idx + 1}/${SCENE_DATA.frames.length}) -- ${f.num_hit}/${f.num_total} hit`;
      }

      function setupTimelineControls() {
        const slider = document.getElementById('frame-slider');
        slider.max = SCENE_DATA.frames.length - 1;
        slider.addEventListener('input', () => showFrame(parseInt(slider.value, 10)));

        const playCheckbox = document.getElementById('play-checkbox');
        let playTimer = null;
        playCheckbox.addEventListener('change', () => {
          if (playCheckbox.checked) {
            playTimer = setInterval(() => {
              const next = (parseInt(slider.value, 10) + 1) % SCENE_DATA.frames.length;
              slider.value = next;
              showFrame(next);
            }, 100); // 100ms = 10fps, matching the settled live-detection rate
          } else if (playTimer) {
            clearInterval(playTimer);
            playTimer = null;
          }
        });
      }

      // CAD frame origin -- permanent, not per-frame.
      function addCadOrigin(bbox) {
        const origin = [0, 0, 0];
        const size = Math.max(SCENE_DATA.sphere_radius_mm * 2, 15);
        scene.add(addSphere(origin, size, 0xe5e7eb, bbox));
        const axisLen = SCENE_DATA.camera_axis_len_mm;
        scene.add(addLine(origin, [axisLen, 0, 0], 0xff3b30));
        scene.add(addLine(origin, [0, axisLen, 0], 0x34c759));
        scene.add(addLine(origin, [0, 0, axisLen], 0x007aff));
        bbox.expandByPoint(new THREE.Vector3(axisLen, axisLen, axisLen));
        const label = createIndexLabel('CAD Origin', 0xe5e7eb);
        label.position.set(origin[0], origin[1] + LABEL_OFFSET_MM, origin[2]);
        scene.add(label);
        labelEntries.push({ sprite: label, center: origin });
      }

      // Static (frame-independent) tracking-model reference points -- built
      // once at init, hidden by default, toggled via the "Show tracking
      // model" checkbox. Does NOT contribute to the camera-framing bbox
      // (stays hidden until the user asks for it, so it shouldn't silently
      // change the initial view), and its spheres are intentionally NOT
      // added to clickableMeshes -- they're a visual reference only, not a
      // per-point inspectable detection.
      let trackingModelGroup = null;
      let trackingModelSpheres = [];
      let trackingModelLabels = [];

      function addTrackingModel() {
        trackingModelGroup = new THREE.Group();
        trackingModelGroup.visible = false;
        const baseR = SCENE_DATA.sphere_radius_mm * 0.7; // slightly smaller than detection spheres, so overlaps stay legible
        SCENE_DATA.tracking_model_points.forEach((p) => {
          const sphere = addSphere(p.point_mm, baseR, 0xff8c00, null);
          sphere.scale.setScalar(trackingSphereFactor);
          trackingModelGroup.add(sphere);
          trackingModelSpheres.push(sphere);

          const label = createIndexLabel('#' + p.point_line, 0xff8c00);
          label.userData = { center: p.point_mm };
          applyLabelTransform(label, trackingTagFactor);
          trackingModelGroup.add(label);
          trackingModelLabels.push(label);
        });
        scene.add(trackingModelGroup);
      }

      function setupTrackingModelControls() {
        const checkbox = document.getElementById('tracking-model-checkbox');
        checkbox.addEventListener('change', () => {
          if (trackingModelGroup) trackingModelGroup.visible = checkbox.checked;
        });
      }

      function setupSizeControls() {
        const estSphereSlider = document.getElementById('estimate-sphere-slider');
        const estTagSlider = document.getElementById('estimate-tag-slider');
        const trkSphereSlider = document.getElementById('tracking-sphere-slider');
        const trkTagSlider = document.getElementById('tracking-tag-slider');

        estSphereSlider.addEventListener('input', () => {
          estimateSphereFactor = parseInt(estSphereSlider.value, 10) / 100;
          frameSpheres.forEach((s) => s.scale.setScalar(estimateSphereFactor));
        });
        estTagSlider.addEventListener('input', () => {
          estimateTagFactor = parseInt(estTagSlider.value, 10) / 100;
          frameLabels.forEach((label) => applyLabelTransform(label, estimateTagFactor));
        });
        trkSphereSlider.addEventListener('input', () => {
          trackingSphereFactor = parseInt(trkSphereSlider.value, 10) / 100;
          trackingModelSpheres.forEach((s) => s.scale.setScalar(trackingSphereFactor));
        });
        trkTagSlider.addEventListener('input', () => {
          trackingTagFactor = parseInt(trkTagSlider.value, 10) / 100;
          trackingModelLabels.forEach((label) => applyLabelTransform(label, trackingTagFactor));
        });
      }

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
          const { coord } = hit.object.userData;
          panel.innerHTML = `<div class="dp-title">Detected point</div>${fmtCoord(coord)}`;
        });
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
        addCadOrigin(bbox);
        addTrackingModel();
        frameCamera(bbox);   // camera view framed from the car mesh's own bbox only --
                              // never re-framed on frame switch, so the view stays stable while scrubbing
        setupStaticControls();
        setupTimelineControls();
        setupTrackingModelControls();
        setupSizeControls();
        setupPicking();
        showFrame(0);

        window.addEventListener('resize', onWindowResize);
        animate();
      }

      init();
    })();
  </script>
</body>
</html>
"""


def build_timeline_viewer_html(mesh, stage2_output: dict, sphere_radius_mm: float = 3.0,
                                camera_axis_len_mm: float = 100.0,
                                window_name: str = "Live Detection Timeline",
                                tracking_model_path: str = TRACKING_MODEL_PATH) -> str:
    """Build the full, self-contained HTML document text for the per-frame
    timeline viewer.

    Args:
        mesh: car mesh, already scaled to millimeters (e.g. from
            pipeline.mesh_utils.load_car_mesh(..., target_triangles=0, scale=1.0)).
        stage2_output: the dict produced by project_to_mesh.project_all_frames.
        tracking_model_path: points3d file for the optional "show tracking
            model" reference overlay (toggled off by default in the viewer);
            pass None to omit the overlay entirely.
    """
    mesh_b64 = base64.b64encode(_mesh_to_binary(mesh)).decode("ascii")
    scene_data = _build_timeline_scene_data(stage2_output, sphere_radius_mm, camera_axis_len_mm,
                                             tracking_model_path=tracking_model_path)

    n_frames = len(scene_data["frames"])
    n_skipped = len(stage2_output.get("skipped_frame_ids", []))
    cam = stage2_output.get("source", {}).get("camera_name", "")
    rig = stage2_output.get("source", {}).get("rig_extrinsics_path")
    rig_str = " (rig composed)" if rig else ""
    cam_str = f" [{cam}{rig_str}]" if cam else ""
    summary = f"{n_frames} frames projected{cam_str}" + (f", {n_skipped} skipped (no VO coverage)" if n_skipped else "")

    if window_name == "Live Detection Timeline" and cam:
        window_name = f"Live Detection Timeline — Camera: {cam}"

    doc = _HTML_TEMPLATE
    doc = doc.replace("__TITLE__", html.escape(window_name))
    doc = doc.replace("__SUMMARY__", html.escape(summary))
    doc = doc.replace("__THREE_JS__", _read_vendor_js("three.min.js"))
    doc = doc.replace("__ORBIT_CONTROLS_JS__", _read_vendor_js("OrbitControls.js"))
    doc = doc.replace("__MESH_B64__", mesh_b64)
    doc = doc.replace("__SCENE_JSON__", json.dumps(scene_data))
    return doc


def show_timeline(mesh, stage2_output: dict, out_path: str = None, **kwargs) -> str:
    """Write the timeline viewer HTML (to out_path, or a fresh temp file)
    and open it in the system's default browser. Returns the path written to."""
    doc = build_timeline_viewer_html(mesh, stage2_output, **kwargs)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="live_backprojection_timeline_", suffix=".html")
        os.close(fd)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    import webbrowser
    webbrowser.open(Path(out_path).resolve().as_uri())
    return out_path
