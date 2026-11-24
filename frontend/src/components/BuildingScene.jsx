import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const THEME = {
  design: {
    sky: 0xe8f1f8,
    ground: 0xd7e2df,
    glass: 0x5aa7c8,
    frame: 0x34515e,
    accent: 0xf59e0b,
    light: 0xffffff,
  },
  operation: {
    sky: 0xe9efe7,
    ground: 0xd8e0d6,
    glass: 0x65a392,
    frame: 0x2e4b45,
    accent: 0x2f855a,
    light: 0xfff3d0,
  },
  extreme: {
    sky: 0xf2e2d4,
    ground: 0xe6d5c8,
    glass: 0xc8744f,
    frame: 0x55352c,
    accent: 0xd9480f,
    light: 0xffe0b2,
  },
};

function makeLabel(text, color, options = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = options.width || 256;
  canvas.height = options.height || 96;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(255,255,255,0.86)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  ctx.fillStyle = '#17202a';
  ctx.font = options.font || '600 34px Microsoft YaHei, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(options.scaleX || 14, options.scaleY || 5.2, 1);
  return sprite;
}

function addParameterLabel(scene, text, position, color, scale = 1) {
  const label = makeLabel(text, color, {
    width: 320,
    height: 86,
    font: '700 28px Microsoft YaHei, sans-serif',
    scaleX: 18 * scale,
    scaleY: 4.8 * scale,
  });
  label.position.set(...position);
  scene.add(label);
  return label;
}

function clampNumber(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function visualConfig(spec) {
  const geometry = spec?.geometry || {};
  const envelope = spec?.envelope || {};
  const operation = spec?.operation || {};
  const length = clampNumber(geometry.length_m, 36, 8, 160);
  const depth = clampNumber(geometry.width_m, 36, 8, 160);
  const groundFloorHeight = clampNumber(geometry.ground_floor_height_m, 3.9, 2.6, 8);
  const typicalFloorHeight = clampNumber(geometry.typical_floor_height_m, 3.9, 2.6, 6.5);
  const floors = Math.round(clampNumber(geometry.floor_count, 15, 1, 80));
  const perimeter = clampNumber(geometry.perimeter_depth_m, 5, 1, Math.max(1.1, Math.min(length, depth) / 2 - 0.4));
  const wwr = clampNumber(envelope.window_wall_ratio, 0.4, 0.05, 0.9);
  const orientation = clampNumber(geometry.orientation_deg, 0, -360, 360);
  const wallIns = clampNumber(envelope.wall_insulation_m, 0.02, 0, 0.5);
  const roofIns = clampNumber(envelope.roof_insulation_m, 0.02, 0, 0.5);
  const cooling = clampNumber(operation.cooling_setpoint_c, 26, 10, 40);
  const heating = clampNumber(operation.heating_setpoint_c, 20, 0, 35);
  const coolingCop = clampNumber(operation.cooling_cop, 3, 0.5, 12);
  const heatingCop = clampNumber(operation.heating_cop, 1.8, 0.5, 12);
  const planShape = geometry.plan_shape === 'l_shape' ? 'l_shape' : 'rectangle';
  const notchCorner = ['ne', 'nw', 'se', 'sw'].includes(geometry.notch_corner) ? geometry.notch_corner : 'ne';
  const notchLength = clampNumber(geometry.notch_length_m, length * 0.35, 2, length - 2 * perimeter - 0.5);
  const notchWidth = clampNumber(geometry.notch_width_m, depth * 0.35, 2, depth - 2 * perimeter - 0.5);
  return {
    length,
    depth,
    groundFloorHeight,
    typicalFloorHeight,
    floors,
    perimeter,
    wwr,
    orientation,
    wallIns,
    roofIns,
    cooling,
    heating,
    coolingCop,
    heatingCop,
    planShape,
    notchCorner,
    notchLength,
    notchWidth,
  };
}

// Cumulative slab elevations [0, ..., totalHeight], honoring the ground/typical
// floor-height split -- mirrors office_geometry.py's _floor_z_range.
function floorLevels(cfg) {
  const levels = [0];
  for (let i = 1; i <= cfg.floors; i += 1) {
    const h = i === 1 ? cfg.groundFloorHeight : cfg.typicalFloorHeight;
    levels.push(levels[i - 1] + h);
  }
  return levels;
}

// Decomposes the footprint into 1 (rectangle) or 2 (L-shape) axis-aligned boxes,
// mirroring office_geometry.py's wing-A/wing-B split: wing A keeps the full X
// extent at reduced Z depth, wing B keeps the full-height Z extent at reduced X
// width. Mirroring per notch_corner reuses the same "ne"-canonical derivation
// instead of re-deriving four separate layouts.
function computeWings(cfg) {
  const xh = cfg.length / 2;
  const zh = cfg.depth / 2;
  if (cfg.planShape !== 'l_shape') {
    return [{ xMin: -xh, xMax: xh, zMin: -zh, zMax: zh }];
  }
  const x1 = cfg.length - cfg.notchLength; // wing B's outer X extent (canonical "ne")
  const y1 = cfg.depth - cfg.notchWidth;   // wing A's outer Z extent (canonical "ne")
  const mirrorX = cfg.notchCorner === 'nw' || cfg.notchCorner === 'sw';
  const mirrorZ = cfg.notchCorner === 'se' || cfg.notchCorner === 'sw';
  const mirror = (w) => ({
    xMin: mirrorX ? -w.xMax : w.xMin,
    xMax: mirrorX ? -w.xMin : w.xMax,
    zMin: mirrorZ ? -w.zMax : w.zMin,
    zMax: mirrorZ ? -w.zMin : w.zMax,
  });
  const wingA = mirror({ xMin: -xh, xMax: xh, zMin: -zh, zMax: y1 - zh });
  const wingB = mirror({ xMin: -xh, xMax: x1 - xh, zMin: y1 - zh, zMax: zh });
  return [wingA, wingB];
}

// The Z plane where the two wings meet (interior seam, no window/fin there).
// Rectangle buildings (single wing) have no seam.
function sharedSeamZ(wings) {
  if (wings.length < 2) return null;
  const [a, b] = wings;
  if (Math.abs(a.zMax - b.zMin) < 1e-6) return a.zMax;
  if (Math.abs(a.zMin - b.zMax) < 1e-6) return a.zMin;
  return null;
}

// Outer footprint vertices (4 for rectangle, 6 for L-shape), centered at the
// origin, used only for placing decorative corner accents -- winding order
// doesn't matter here since nothing gets extruded/filled from this list.
function computeBoundary(cfg) {
  const xh = cfg.length / 2;
  const zh = cfg.depth / 2;
  if (cfg.planShape !== 'l_shape') {
    return [[-xh, -zh], [xh, -zh], [xh, zh], [-xh, zh]];
  }
  const x1c = xh - cfg.notchLength;
  const y1c = zh - cfg.notchWidth;
  const mirrorX = cfg.notchCorner === 'nw' || cfg.notchCorner === 'sw';
  const mirrorZ = cfg.notchCorner === 'se' || cfg.notchCorner === 'sw';
  const pts = [[-xh, -zh], [xh, -zh], [xh, y1c], [x1c, y1c], [x1c, zh], [-xh, zh]];
  return pts.map(([x, z]) => [mirrorX ? -x : x, mirrorZ ? -z : z]);
}


export default function BuildingScene({ mode = 'design', intensity = 0.55, spec = null }) {
  const hostRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    host.innerHTML = '';

    const theme = THEME[mode] || THEME.design;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(theme.sky);
    const cfg = visualConfig(spec);
    const levels = floorLevels(cfg);
    const totalH = levels[levels.length - 1];
    const wings = computeWings(cfg);
    const sharedZ = sharedSeamZ(wings);
    const radius = Math.max(cfg.length, cfg.depth, totalH, 36);
    scene.fog = new THREE.Fog(theme.sky, radius * 1.6, radius * 5.2);

    const camera = new THREE.PerspectiveCamera(42, host.clientWidth / host.clientHeight, 0.1, 600);
    camera.position.set(radius * 1.55, radius * 1.05, radius * 1.65);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(0, totalH * 0.48, 0);
    controls.minDistance = Math.max(22, radius * 0.8);
    controls.maxDistance = radius * 5.5;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.autoRotate = true;
    controls.autoRotateSpeed = mode === 'operation' ? 0.45 : 0.28;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x7d8a8d, 2.2);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(theme.light, mode === 'extreme' ? 4.5 : 3.2);
    sun.position.set(-55, 90, 45);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    scene.add(sun);

    const groundMat = new THREE.MeshStandardMaterial({ color: theme.ground, roughness: 0.88 });
    const ground = new THREE.Mesh(new THREE.CircleGeometry(radius * 2.2, 96), groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    const grid = new THREE.GridHelper(radius * 3.2, 30, theme.frame, 0xffffff);
    grid.material.opacity = 0.25;
    grid.material.transparent = true;
    scene.add(grid);

    const group = new THREE.Group();
    group.rotation.y = THREE.MathUtils.degToRad(cfg.orientation);
    scene.add(group);

    const facadeMat = new THREE.MeshStandardMaterial({ color: 0xf6f8fb, roughness: 0.72 });
    const glassMat = new THREE.MeshPhysicalMaterial({
      color: theme.glass,
      metalness: 0.08,
      roughness: 0.18,
      transmission: 0.12,
      transparent: true,
      opacity: 0.78,
    });
    const slabMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.7 });
    const frameMat = new THREE.MeshStandardMaterial({ color: theme.frame, metalness: 0.18, roughness: 0.42 });
    const coreMat = new THREE.MeshStandardMaterial({ color: theme.accent, roughness: 0.48, transparent: true, opacity: 0.82 });
    const finMat = new THREE.MeshStandardMaterial({ color: theme.frame, roughness: 0.5 });

    // Body + slabs + windows + fins are built per wing (1 for rectangle, 2 for
    // L-shape: the full-length wing and the narrower wing sharing one interior
    // seam -- mirrors the two-wing decomposition in office_geometry.py). This
    // keeps the preview's footprint faithful to what EnergyPlus will actually
    // simulate, instead of approximating the L-shape with a single box.
    wings.forEach((wing) => {
      const wingLen = wing.xMax - wing.xMin;
      const wingDepth = wing.zMax - wing.zMin;
      const cx = (wing.xMin + wing.xMax) / 2;
      const cz = (wing.zMin + wing.zMax) / 2;

      const body = new THREE.Mesh(new THREE.BoxGeometry(wingLen, totalH, wingDepth), facadeMat);
      body.position.set(cx, totalH / 2, cz);
      body.castShadow = true;
      body.receiveShadow = true;
      group.add(body);

      levels.forEach((y) => {
        const slab = new THREE.Mesh(new THREE.BoxGeometry(wingLen + 1.2, 0.16, wingDepth + 1.2), slabMat);
        slab.position.set(cx, y, cz);
        slab.castShadow = true;
        group.add(slab);
      });

      const longWin = new THREE.BoxGeometry(wingLen * 0.86, 1, 0.16);
      const sideWin = new THREE.BoxGeometry(0.16, 1, wingDepth * 0.86);
      const frontOpen = !(sharedZ !== null && Math.abs(wing.zMax - sharedZ) < 1e-6);
      const backOpen = !(sharedZ !== null && Math.abs(wing.zMin - sharedZ) < 1e-6);
      for (let f = 1; f <= cfg.floors; f += 1) {
        const floorH = levels[f] - levels[f - 1];
        const windowH = Math.max(0.45, floorH * cfg.wwr);
        const y = levels[f - 1] + floorH * 0.52;
        if (frontOpen) {
          const front = new THREE.Mesh(longWin, glassMat);
          front.scale.y = windowH;
          front.position.set(cx, y, wing.zMax + 0.11);
          group.add(front);
        }
        if (backOpen) {
          const back = new THREE.Mesh(longWin, glassMat);
          back.scale.y = windowH;
          back.position.set(cx, y, wing.zMin - 0.11);
          group.add(back);
        }
        const east = new THREE.Mesh(sideWin, glassMat);
        east.scale.y = windowH;
        east.position.set(wing.xMax + 0.11, y, cz);
        group.add(east);
        const west = new THREE.Mesh(sideWin, glassMat);
        west.scale.y = windowH;
        west.position.set(wing.xMin - 0.11, y, cz);
        group.add(west);
      }

      const xStep = Math.max(4, wingLen / Math.min(10, Math.max(4, Math.round(wingLen / 5))));
      for (let x = wing.xMin + xStep; x < wing.xMax; x += xStep) {
        if (frontOpen) {
          const finFront = new THREE.Mesh(new THREE.BoxGeometry(0.12, totalH, 0.5), finMat);
          finFront.position.set(x, totalH / 2, wing.zMax + 0.38);
          group.add(finFront);
        }
        if (backOpen) {
          const finBack = new THREE.Mesh(new THREE.BoxGeometry(0.12, totalH, 0.5), finMat);
          finBack.position.set(x, totalH / 2, wing.zMin - 0.38);
          group.add(finBack);
        }
      }
      const zStep = Math.max(4, wingDepth / Math.min(10, Math.max(4, Math.round(wingDepth / 5))));
      for (let z = wing.zMin + zStep; z < wing.zMax; z += zStep) {
        const finLeft = new THREE.Mesh(new THREE.BoxGeometry(0.5, totalH, 0.12), finMat);
        finLeft.position.set(wing.xMin - 0.38, totalH / 2, z);
        group.add(finLeft);
        const finRight = new THREE.Mesh(new THREE.BoxGeometry(0.5, totalH, 0.12), finMat);
        finRight.position.set(wing.xMax + 0.38, totalH / 2, z);
        group.add(finRight);
      }
    });

    const boundary = computeBoundary(cfg);
    const edgeGeo = new THREE.BoxGeometry(0.5, totalH + 0.5, 0.5);
    boundary.forEach(([x, z]) => {
      const edge = new THREE.Mesh(edgeGeo, frameMat);
      edge.position.set(x, totalH / 2, z);
      edge.castShadow = true;
      group.add(edge);
    });

    const coreMeshes = wings.map((wing) => {
      const xMin = wing.xMin + cfg.perimeter;
      const xMax = wing.xMax - cfg.perimeter;
      const zMin = wing.zMin + cfg.perimeter;
      const zMax = wing.zMax - cfg.perimeter;
      if (xMax - xMin < 0.6 || zMax - zMin < 0.6) return null;
      const core = new THREE.Mesh(
        new THREE.BoxGeometry(xMax - xMin, totalH + 0.2, zMax - zMin),
        coreMat,
      );
      core.position.set((xMin + xMax) / 2, totalH / 2, (zMin + zMax) / 2);
      group.add(core);
      return core;
    }).filter(Boolean);

    const podium = new THREE.Mesh(new THREE.BoxGeometry(cfg.length + 14, Math.min(4.2, cfg.groundFloorHeight * 1.1), cfg.depth + 10), new THREE.MeshStandardMaterial({
      color: 0xf1f5f2,
      roughness: 0.72,
    }));
    podium.position.y = Math.min(4.2, cfg.groundFloorHeight * 1.1) / 2;
    podium.castShadow = true;
    podium.receiveShadow = true;
    group.add(podium);

    const shapeLabel = cfg.planShape === 'l_shape' ? 'L形' : '矩形';
    const labelText = spec ? `${cfg.floors}层 ${shapeLabel} ${cfg.length.toFixed(0)}×${cfg.depth.toFixed(0)}m` : (mode === 'design' ? '设计方案' : mode === 'operation' ? '运维调控' : '极端天气');
    const label = makeLabel(labelText, `#${theme.accent.toString(16).padStart(6, '0')}`);
    label.position.set(-cfg.length * 0.72, totalH * 0.76, cfg.depth * 0.7);
    scene.add(label);
    const paramLabels = [];
    if (spec) {
      const color = `#${theme.frame.toString(16).padStart(6, '0')}`;
      [
        [`首层/标准层 ${cfg.groundFloorHeight.toFixed(1)}/${cfg.typicalFloorHeight.toFixed(1)}m`, [cfg.length * 0.72, totalH * 0.82, cfg.depth * 0.68], 0.88],
        [`周边区 ${cfg.perimeter.toFixed(1)}m`, [cfg.length * 0.78, totalH * 0.55, -cfg.depth * 0.72], 0.88],
        [`窗墙比 ${cfg.wwr.toFixed(2)}`, [-cfg.length * 0.8, totalH * 0.48, -cfg.depth * 0.72], 0.88],
        [`朝向 ${cfg.orientation.toFixed(0)}°`, [0, totalH + Math.max(8, totalH * 0.12), 0], 0.8],
        [`设定 ${cfg.cooling.toFixed(0)}/${cfg.heating.toFixed(0)}°C`, [-cfg.length * 0.82, totalH * 0.28, cfg.depth * 0.74], 0.82],
        [`COP ${cfg.coolingCop.toFixed(1)}/${cfg.heatingCop.toFixed(1)}`, [cfg.length * 0.82, totalH * 0.3, cfg.depth * 0.74], 0.82],
        [`保温 墙${cfg.wallIns.toFixed(2)} 屋${cfg.roofIns.toFixed(2)}m`, [0, Math.max(8, totalH * 0.16), -cfg.depth * 0.95], 0.78],
      ].forEach(([text, position, scale]) => {
        const sprite = addParameterLabel(scene, text, position, color, scale);
        paramLabels.push({ sprite, baseY: position[1] });
      });
    }

    if (mode === 'operation') {
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x2f855a, transparent: true, opacity: 0.5 });
      for (let i = 0; i < 4; i += 1) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(Math.max(cfg.length, cfg.depth) * 0.7 + i * 7, 0.05, 8, 96), ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = totalH * 0.2 + i * totalH * 0.16;
        group.add(ring);
      }
    }

    if (mode === 'extreme') {
      const heatMat = new THREE.MeshBasicMaterial({ color: 0xd9480f, transparent: true, opacity: 0.14, side: THREE.DoubleSide });
      for (let i = 0; i < 8; i += 1) {
        const wave = new THREE.Mesh(new THREE.PlaneGeometry(radius * 0.12, totalH * 0.62), heatMat);
        wave.position.set(-radius * 0.9 + i * radius * 0.25, totalH * 0.58, -cfg.depth * 0.75);
        wave.rotation.y = Math.PI / 10;
        group.add(wave);
      }
    }

    let raf = 0;
    const clock = new THREE.Clock();

    function animate() {
      const elapsed = clock.getElapsedTime();
      const coreOpacity = 0.64 + Math.sin(elapsed * 2.2) * 0.08 * intensity;
      coreMeshes.forEach((core) => { core.material.opacity = coreOpacity; });
      label.position.y = totalH * 0.76 + Math.sin(elapsed * 1.4) * 0.7;
      paramLabels.forEach((item, index) => {
        item.sprite.position.y = item.baseY + Math.sin(elapsed * 1.2 + index) * 0.28;
      });
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }

    function resize() {
      const w = Math.max(host.clientWidth, 1);
      const h = Math.max(host.clientHeight, 1);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    animate();

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          materials.forEach((material) => {
            if (material.map) material.map.dispose();
            material.dispose();
          });
        }
      });
      renderer.dispose();
      host.innerHTML = '';
    };
  }, [mode, intensity, spec]);

  return <div ref={hostRef} className="scene-host" aria-label="building model" />;
}
