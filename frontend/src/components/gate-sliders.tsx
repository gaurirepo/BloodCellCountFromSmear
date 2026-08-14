"use client";

import {
  CELL_CLASSES,
  CLASS_COLORS,
  DEFAULT_GATES,
  MAX_GATE,
  MIN_INFER_CONF,
  type CellClass,
} from "@/lib/analyze";

type Props = {
  gates: Record<CellClass, number>;
  onChange: (gates: Record<CellClass, number>) => void;
  onReset: () => void;
  isDefault: boolean;
};

export function GateSliders({ gates, onChange, onReset, isDefault }: Props) {
  return (
    <div className="mt-8 border border-white/10 bg-white/[0.02] px-5 py-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.32em] text-white/40">
            Live operating gates
          </p>
          <p className="mt-2 text-sm tracking-wide text-white/45">
            Infer at {MIN_INFER_CONF.toFixed(2)}. Counts follow the sliders, not
            COCO mAP.
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          disabled={isDefault}
          className="text-[11px] uppercase tracking-[0.28em] text-white/70 hover:text-white disabled:text-white/25"
        >
          Reset 0.60 / 0.40 / 0.40
        </button>
      </div>
      <div className="grid gap-6 sm:grid-cols-3">
        {CELL_CLASSES.map((cls) => (
          <label key={cls} className="block">
            <span className="mb-3 flex items-baseline justify-between text-[10px] uppercase tracking-[0.28em]">
              <span style={{ color: CLASS_COLORS[cls] }}>{cls}</span>
              <span className="font-serif text-xl italic tracking-normal text-white">
                {gates[cls].toFixed(2)}
              </span>
            </span>
            <input
              type="range"
              min={Math.round(MIN_INFER_CONF * 100)}
              max={Math.round(MAX_GATE * 100)}
              step={1}
              value={Math.round(gates[cls] * 100)}
              onChange={(event) =>
                onChange({
                  ...gates,
                  [cls]: Number(event.target.value) / 100,
                })
              }
              className="field-reveal"
              aria-label={`${cls} confidence gate`}
            />
          </label>
        ))}
      </div>
    </div>
  );
}
