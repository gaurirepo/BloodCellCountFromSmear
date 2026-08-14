"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function CellField() {
  const points = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const count = 2200;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const palette = [
      new THREE.Color("#4a0414"),
      new THREE.Color("#8b1e3f"),
      new THREE.Color("#ef4444"),
      new THREE.Color("#f59e0b"),
      new THREE.Color("#6b1026"),
    ];

    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 10;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 6.2;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 5.5;
      const color = palette[Math.floor(Math.random() * palette.length)];
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geo;
  }, []);

  useFrame(({ clock, pointer }) => {
    const mesh = points.current;
    if (!mesh) return;
    mesh.rotation.y = clock.elapsedTime * 0.035 + pointer.x * 0.35;
    mesh.rotation.x = pointer.y * 0.18;
  });

  return (
    <points ref={points} geometry={geometry}>
      <pointsMaterial
        size={0.028}
        vertexColors
        transparent
        opacity={0.78}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
      />
    </points>
  );
}

export function HeroParticles() {
  return (
    <Canvas
      camera={{ position: [0, 0, 4.2], fov: 55 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: true }}
    >
      <CellField />
    </Canvas>
  );
}
