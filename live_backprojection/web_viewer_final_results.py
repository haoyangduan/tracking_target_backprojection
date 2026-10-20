"""Stage 4 web viewer: the final, clustered 3D "detected tracking model"
vs. the known Creaform-surveyed ground-truth tracking model, one static
scene (no per-frame scrubbing -- this is the consolidated result after
all frames have been merged, see cluster_results.py).

Extends pipeline/web_viewer.py's established Three.js / self-contained-HTML
pattern (same dark theme, same binary mesh embedding, same vendored
three.min.js/OrbitControls.js, same studio-environment lighting): reuses
_mesh_to_binary/_read_vendor_js directly, same as web_viewer_timeline.py.

Each matched pair (one clustered detected point + its nearest ground-truth
point) is drawn as: an amber sphere (ground truth), a sphere colored green
(good match) or red (likely false detect) for the detected point, and a
thin line between them. A live "False-detect threshold" slider recolors
points instantly by re-comparing each pair's already-computed error_mm --
no need to re-cluster/re-match to explore different cutoffs. Click a point
for the full pair detail (coordinates, error, observation count).
"""
import base64
import html
import json
import os
import tempfile
from pathlib import Path

from live_backprojection import _pathsetup  # noqa: F401

from single_frame_backprojection.web_viewer import _mesh_to_binary, _read_vendor_js


def _build_final_results_scene_data(final_results: dict, sphere_radius_mm: float = 4.0) -> dict:
    pairs = [
        {
            "point_line": p["point_line"],
            "detected_point": p["detected_point_mm"],
            "gt_point": p["gt_point_mm"],
            "error_mm": p["error_mm"],
            "n_observations": p["n_observations"],
            "spread_mm": p["spread_mm"],
        }
        for p in final_results["pairs"]
    ]
    return {
        "pairs": pairs,
        "error_threshold_mm": float(final_results["error_threshold_mm"]),
        "merge_dist_mm": float(final_results["merge_dist_mm"]),
        "sphere_radius_mm": float(sphere_radius_mm),
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
    #threshold-readout { color: #94a3b8; min-width: 160px; }
    #detail-panel { position: absolute; right: 16px; bottom: 16px; z-index: 10; max-width: 280px;
      background: rgba(22, 26, 34, 0.9); border: 1px solid #2d3342; border-radius: 8px;
      padding: 10px 14px; font-size: 12px; color: #cbd5e1; }
    #detail-panel .dp-title { font-weight: bold; color: #eceff4; margin-bottom: 4px; }
    #detail-panel .dp-row { margin-top: 2px; }
  </style>
</head>
<body>
  <div id="header">
    <div id="title-block">
      <span id="title">__TITLE__</span>
      <span id="summary">__SUMMARY__</span>
    </div>
    <div id="legend">
      <span><span class="swatch" style="background:#00cc00;"></span>Detected (good match)</span>
      <span><span class="swatch" style="background:#ff3b30;"></span>Detected (likely false)</span>
      <span><span class="swatch" style="background:#ff8c00;"></span>Ground truth (Creaform)</span>
    </div>
    <div id="hint">Rotate: Left drag&nbsp;&nbsp;|&nbsp;&nbsp;Pan: Right drag&nbsp;&nbsp;|&nbsp;&nbsp;Zoom: Wheel</div>
  </div>
  <div id="canvas-wrap"><canvas id="webgl-canvas"></canvas></div>
  <div id="controls">
    <label><input type="checkbox" id="hide-mesh-checkbox"> Hide car mesh</label>
    <label>Mesh opacity <input type="range" id="opacity-slider" min="0" max="100" value="100"></label>
    <label>Detected sphere size <input type="range" id="detected-sphere-slider" min="0" max="300" value="100"></label>
    <label>Ground-truth sphere size <input type="range" id="gt-sphere-slider" min="0" max="300" value="100"></label>
    <label>Tag size <input type="range" id="tag-size-slider" min="0" max="300" value="100"></label>
    <label>False-detect threshold (mm)
      <input type="range" id="threshold-slider" min="0" max="200" value="20" step="1">
      <span id="threshold-readout"></span></label>
    <label><input type="checkbox" id="hide-false-checkbox"> Hide flagged false detects</label>
  </div>
  <div id="detail-panel">Click a point for its match detail.</div>

  <script>__THREE_JS__</script>
  <script>__ORBIT_CONTROLS_JS__</script>
  <script>
    const CAR_MESH_B64 = "__MESH_B64__";
    const SCENE_DATA = __SCENE_JSON__;

    (function () {
      let scene, camera, renderer, controls;
      let carMesh, carMaterial;
      let currentThreshold = SCENE_DATA.error_threshold_mm;

      function buildStudioEnvironment(renderer) {
        const pmremGenerator = new THREE.PMREMGenerator(renderer);
        pmremGenerator.compileEquirectangularShader();
        const envScene = new THREE.Scene();
        const roomGeo = new THREE.SphereGeometry(60, 32, 16);
        const roomMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8, side: THREE.BackSide, roughness: 1.0, metalness: 0.0,
        });
        envScene.add(new THREE.Mesh(roomGeo, roomMat));
        const key = new THREE.PointLight(0xffffff, 1.4, 140, 2); key.position.set(25, 45, 30); envScene.add(key);
        const fill = new THREE.PointLight(0x93c5fd, 0.7, 140, 2); fill.position.set(-35, 15, 20); envScene.add(fill);
        const rim = new THREE.PointLight(0x38bdf8, 0.5, 140, 2); rim.position.set(-15, -25, -35); envScene.add(rim);
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
        hideCheckbox.addEventListener('change', () => { carMesh.visible = !hideCheckbox.checked; });
      }

      function addSphere(center, radiusMm, colorHex) {
        const geo = new THREE.SphereGeometry(radiusMm, 20, 20);
        const mat = new THREE.MeshStandardMaterial({
          color: colorHex, emissive: colorHex, emissiveIntensity: 0.35, roughness: 0.3, metalness: 0.2,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(center[0], center[1], center[2]);
        return mesh;
      }

      function addLine(a, b, colorHex) {
        const geo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(a[0], a[1], a[2]), new THREE.Vector3(b[0], b[1], b[2]),
        ]);
        return new THREE.Line(geo, new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.5 }));
      }

      const LABEL_BASE_W = 45, LABEL_BASE_H = 22.5, LABEL_OFFSET_MM = 30;

      function createIndexLabel(text, colorHex) {
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const colorCss = '#' + colorHex.toString(16).padStart(6, '0');
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'; ctx.strokeStyle = colorCss; ctx.lineWidth = 5;
        const r = 16, x = 4, y = 4, w = 120, h = 56;
        ctx.beginPath();
        ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#ffffff';
        const fontPx = text.length > 4 ? Math.max(16, Math.floor(34 * 4 / text.length)) : 34;
        ctx.font = `bold ${fontPx}px "Segoe UI", Arial, sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 33);
        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        const mat = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false, transparent: true });
        const sprite = new THREE.Sprite(mat);
        sprite.scale.set(LABEL_BASE_W, LABEL_BASE_H, 1);
        return sprite;
      }

      function fmtCoord(p) { return `(${p[0].toFixed(1)}, ${p[1].toFixed(1)}, ${p[2].toFixed(1)}) mm`; }

      const GOOD_COLOR = 0x00cc00, BAD_COLOR = 0xff3b30, GT_COLOR = 0xff8c00;
      const pairEntries = []; // { detectedSphere, gtSphere, line, label, pair }
      const clickableMeshes = [];
      let detectedSphereFactor = 1.0, gtSphereFactor = 1.0, tagFactor = 1.0;

      function applyLabelTransform(label, center, factor) {
        label.scale.set(LABEL_BASE_W * factor, LABEL_BASE_H * factor, 1);
        label.position.set(center[0], center[1] + LABEL_OFFSET_MM * factor, center[2]);
      }

      function addPairs(bbox) {
        const r = SCENE_DATA.sphere_radius_mm;
        SCENE_DATA.pairs.forEach((pair) => {
          const isFalse = pair.error_mm > currentThreshold;
          const detectedSphere = addSphere(pair.detected_point, r, isFalse ? BAD_COLOR : GOOD_COLOR);
          detectedSphere.scale.setScalar(detectedSphereFactor);
          detectedSphere.userData = { pair };
          scene.add(detectedSphere);
          clickableMeshes.push(detectedSphere);
          bbox.expandByPoint(new THREE.Vector3(...pair.detected_point));

          const gtSphere = addSphere(pair.gt_point, r * 0.8, GT_COLOR);
          gtSphere.scale.setScalar(gtSphereFactor);
          gtSphere.userData = { pair, isGt: true };
          scene.add(gtSphere);
          clickableMeshes.push(gtSphere);
          bbox.expandByPoint(new THREE.Vector3(...pair.gt_point));

          const line = addLine(pair.detected_point, pair.gt_point, 0x999999);
          scene.add(line);

          const label = createIndexLabel('#' + pair.point_line, isFalse ? BAD_COLOR : GOOD_COLOR);
          applyLabelTransform(label, pair.detected_point, tagFactor);
          scene.add(label);

          pairEntries.push({ detectedSphere, gtSphere, line, label, pair });
        });
      }

      function updateThresholdColoring() {
        let nFalse = 0;
        pairEntries.forEach(({ detectedSphere, label, pair }) => {
          const isFalse = pair.error_mm > currentThreshold;
          if (isFalse) nFalse++;
          const color = isFalse ? BAD_COLOR : GOOD_COLOR;
          detectedSphere.material.color.setHex(color);
          detectedSphere.material.emissive.setHex(color);
          label.material.map.dispose();
          const newLabel = createIndexLabel('#' + pair.point_line, color);
          label.material.map = newLabel.material.map;
          label.material.needsUpdate = true;
        });
        document.getElementById('threshold-readout').textContent =
          `${currentThreshold}mm -- ${nFalse}/${pairEntries.length} flagged false`;
        applyHideFalse();
      }

      function applyHideFalse() {
        const hide = document.getElementById('hide-false-checkbox').checked;
        pairEntries.forEach(({ detectedSphere, gtSphere, line, label, pair }) => {
          const isFalse = pair.error_mm > currentThreshold;
          const visible = !(hide && isFalse);
          detectedSphere.visible = visible;
          line.visible = visible;
          label.visible = visible;
        });
      }

      function setupThresholdControls() {
        const slider = document.getElementById('threshold-slider');
        slider.value = currentThreshold;
        slider.addEventListener('input', () => {
          currentThreshold = parseFloat(slider.value);
          updateThresholdColoring();
        });
        document.getElementById('hide-false-checkbox').addEventListener('change', applyHideFalse);
        updateThresholdColoring();
      }

      function setupSizeControls() {
        const detSlider = document.getElementById('detected-sphere-slider');
        const gtSlider = document.getElementById('gt-sphere-slider');
        const tagSlider = document.getElementById('tag-size-slider');
        detSlider.addEventListener('input', () => {
          detectedSphereFactor = parseInt(detSlider.value, 10) / 100;
          pairEntries.forEach(({ detectedSphere }) => detectedSphere.scale.setScalar(detectedSphereFactor));
        });
        gtSlider.addEventListener('input', () => {
          gtSphereFactor = parseInt(gtSlider.value, 10) / 100;
          pairEntries.forEach(({ gtSphere }) => gtSphere.scale.setScalar(gtSphereFactor));
        });
        tagSlider.addEventListener('input', () => {
          tagFactor = parseInt(tagSlider.value, 10) / 100;
          pairEntries.forEach(({ label, pair }) => applyLabelTransform(label, pair.detected_point, tagFactor));
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
          const { pair, isGt } = hit.object.userData;
          const verdict = pair.error_mm > currentThreshold
            ? '<span style="color:#ff3b30;">LIKELY FALSE DETECT</span>'
            : '<span style="color:#00cc00;">OK</span>';
          panel.innerHTML = `
            <div class="dp-title">#${pair.point_line} ${isGt ? '(clicked: ground truth)' : '(clicked: detected)'}</div>
            <div class="dp-row">Detected: ${fmtCoord(pair.detected_point)}</div>
            <div class="dp-row">Ground truth: ${fmtCoord(pair.gt_point)}</div>
            <div class="dp-row">Error: ${pair.error_mm.toFixed(2)}mm -- ${verdict}</div>
            <div class="dp-row">Observations merged: ${pair.n_observations} (spread ${pair.spread_mm.toFixed(2)}mm)</div>
          `;
        });
      }

      function frameCamera(bbox) {
        if (bbox.isEmpty()) bbox.set(new THREE.Vector3(-500, -500, -500), new THREE.Vector3(500, 500, 500));
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        const radius = Math.max(size.x, size.y, size.z, 100) * 0.5;
        const dist = (radius / Math.tan((camera.fov * Math.PI / 180) / 2)) * 1.6;
        camera.position.set(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.6);
        camera.near = 1; camera.far = dist * 20 + radius * 4;
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
        controls.enableDamping = true; controls.dampingFactor = 0.08;
        scene.environment = buildStudioEnvironment(renderer);

        scene.add(new THREE.AmbientLight(0xffffff, 0.12));
        const dir1 = new THREE.DirectionalLight(0xffffff, 0.55); dir1.position.set(2500, 4500, 3500); scene.add(dir1);
        const dir2 = new THREE.DirectionalLight(0x93c5fd, 0.25); dir2.position.set(-2500, 2000, 2500); scene.add(dir2);
        const dir3 = new THREE.DirectionalLight(0x38bdf8, 0.2); dir3.position.set(-1500, 1000, -3500); scene.add(dir3);
        const dirBounce = new THREE.DirectionalLight(0x334155, 0.2); dirBounce.position.set(0, -3000, 0); scene.add(dirBounce);

        const bbox = addCarMesh();
        addPairs(bbox);
        frameCamera(bbox);
        setupStaticControls();
        setupSizeControls();
        setupThresholdControls();
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


def build_final_results_viewer_html(mesh, final_results: dict, sphere_radius_mm: float = 4.0,
                                     window_name: str = "Final Detected Tracking Model") -> str:
    """Build the full, self-contained HTML document text for the Stage-4
    final-results viewer.

    Args:
        mesh: car mesh, already scaled to millimeters (e.g. from
            pipeline.mesh_utils.load_car_mesh(..., target_triangles=0, scale=1.0)).
        final_results: the dict produced by cluster_results.build_final_results.
    """
    mesh_b64 = base64.b64encode(_mesh_to_binary(mesh)).decode("ascii")
    scene_data = _build_final_results_scene_data(final_results, sphere_radius_mm)

    n = scene_data["pairs"]
    n_total = len(n)
    n_false = sum(1 for p in n if p["error_mm"] > scene_data["error_threshold_mm"])
    cam = final_results.get("source", {}).get("camera_name", "")
    cam_str = f" [{cam}]" if cam else ""
    if window_name == "Final Detected Tracking Model" and cam:
        window_name = f"Final Detected Tracking Model — Camera: {cam}"
    summary = (f"{n_total} clustered points{cam_str} (merge dist {scene_data['merge_dist_mm']:.0f}mm) -- "
               f"{n_total - n_false} good, {n_false} likely false (threshold {scene_data['error_threshold_mm']:.0f}mm)")

    doc = _HTML_TEMPLATE
    doc = doc.replace("__TITLE__", html.escape(window_name))
    doc = doc.replace("__SUMMARY__", html.escape(summary))
    doc = doc.replace("__THREE_JS__", _read_vendor_js("three.min.js"))
    doc = doc.replace("__ORBIT_CONTROLS_JS__", _read_vendor_js("OrbitControls.js"))
    doc = doc.replace("__MESH_B64__", mesh_b64)
    doc = doc.replace("__SCENE_JSON__", json.dumps(scene_data))
    return doc


def show_final_results(mesh, final_results: dict, out_path: str = None, **kwargs) -> str:
    """Write the final-results viewer HTML (to out_path, or a fresh temp
    file) and open it in the system's default browser. Returns the path
    written to."""
    doc = build_final_results_viewer_html(mesh, final_results, **kwargs)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="live_backprojection_final_results_", suffix=".html")
        os.close(fd)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    import webbrowser
    webbrowser.open(Path(out_path).resolve().as_uri())
    return out_path
