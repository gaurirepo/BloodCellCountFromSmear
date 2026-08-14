"use client";

import { useCallback, useRef, useState } from "react";
import {
  CLASS_COLORS,
  type AnalyzeBox,
  type CellClass,
} from "@/lib/analyze";

type Props = {
  src: string;
  boxes: AnalyzeBox[];
  loading?: boolean;
  visible: Record<CellClass, boolean>;
  gatesLabel: string;
};

export function FieldViewer({ src, boxes, loading, visible, gatesLabel }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [size, setSize] = useState({ w: 1, h: 1 });
  const [reveal, setReveal] = useState(72);
  const [hover, setHover] = useState<{
    label: string;
    conf: number;
    x: number;
    y: number;
  } | null>(null);

  const setFromClientX = useCallback((clientX: number) => {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setReveal(Math.min(100, Math.max(0, pct)));
  }, []);

  const shown = boxes.filter((box) => visible[box.class as CellClass] !== false);

  return (
    <figure data-split className="border border-white/10">
      <figcaption className="flex items-center justify-between gap-3 px-4 py-3 text-[10px] uppercase tracking-[0.3em] text-white/40">
        <span>Field · drag to compare</span>
        <span className="tracking-[0.22em] text-white/30">
          live gates {gatesLabel}
        </span>
      </figcaption>
      <div
        ref={wrapRef}
        className={`relative cursor-ew-resize select-none ${loading ? "smear-pulse" : ""}`}
        onPointerDown={(event) => {
          if ((event.target as HTMLElement).closest("[data-box]")) return;
          dragging.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          setFromClientX(event.clientX);
        }}
        onPointerMove={(event) => {
          if (dragging.current) setFromClientX(event.clientX);
        }}
        onPointerUp={() => {
          dragging.current = false;
        }}
        onPointerCancel={() => {
          dragging.current = false;
        }}
      >
        <img
          src={src}
          alt="Blood smear field"
          className="w-full"
          draggable={false}
          onLoad={(event) => {
            const img = event.currentTarget;
            setSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${size.w} ${size.h}`}
          preserveAspectRatio="none"
          style={{ clipPath: `inset(0 ${100 - reveal}% 0 0)` }}
        >
          {shown.map((box, index) => {
            const [x1, y1, x2, y2] = box.box;
            const color = CLASS_COLORS[box.class as CellClass] ?? "#94a3b8";
            return (
              <rect
                key={`${box.class}-${index}`}
                data-box
                x={x1}
                y={y1}
                width={Math.max(1, x2 - x1)}
                height={Math.max(1, y2 - y1)}
                fill={`${color}22`}
                stroke={color}
                strokeWidth={Math.max(1.6, size.w / 320)}
                className="cursor-crosshair"
                onPointerEnter={(event) => {
                  const parent = wrapRef.current?.getBoundingClientRect();
                  if (!parent) return;
                  setHover({
                    label: box.class,
                    conf: box.confidence,
                    x: event.clientX - parent.left,
                    y: event.clientY - parent.top,
                  });
                }}
                onPointerMove={(event) => {
                  const parent = wrapRef.current?.getBoundingClientRect();
                  if (!parent) return;
                  setHover({
                    label: box.class,
                    conf: box.confidence,
                    x: event.clientX - parent.left,
                    y: event.clientY - parent.top,
                  });
                }}
                onPointerLeave={() => setHover(null)}
              />
            );
          })}
        </svg>
        <div
          className="pointer-events-none absolute inset-y-0 z-10 w-px bg-white/85"
          style={{ left: `${reveal}%` }}
        />
        <div
          className="pointer-events-none absolute top-1/2 z-10 h-7 w-7 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/50 bg-[#140308]/75"
          style={{ left: `${reveal}%` }}
        />
        {hover ? (
          <div
            className="pointer-events-none absolute z-20 rounded-sm border border-white/15 bg-[#140308]/92 px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-white"
            style={{
              left: Math.min(hover.x + 12, (wrapRef.current?.clientWidth ?? 200) - 140),
              top: hover.y + 12,
            }}
          >
            {hover.label} · {Math.round(hover.conf * 100)}%
          </div>
        ) : null}
        {loading ? <div className="smear-scan" aria-hidden /> : null}
      </div>
      <div className="flex items-center gap-3 px-4 py-3 text-[10px] uppercase tracking-[0.28em] text-white/35">
        <span>Original</span>
        <input
          type="range"
          min={0}
          max={100}
          value={reveal}
          onChange={(event) => setReveal(Number(event.target.value))}
          className="field-reveal"
          aria-label="Reveal detections"
        />
        <span>Detections</span>
      </div>
    </figure>
  );
}
