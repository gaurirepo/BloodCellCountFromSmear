"use client";

import { CLASS_COLORS, type CellClass } from "@/lib/analyze";
import { HELD_OUT } from "@/lib/metrics";

const CLASSES: CellClass[] = ["WBC", "RBC", "Platelets"];

function Bar({
  value,
  color,
}: {
  value: number;
  color: string;
}) {
  return (
    <div className="h-1.5 overflow-hidden bg-white/10">
      <div
        className="h-full"
        style={{ width: `${Math.min(100, value)}%`, background: color }}
      />
    </div>
  );
}

export function Report() {
  const { current, baseline } = HELD_OUT;
  const delta = (current.map50 - baseline.map50).toFixed(1);

  return (
    <section
      id="report"
      className="relative mx-auto w-[min(96vw,1180px)] px-6 pb-28 pt-10"
    >
      <p className="mb-4 text-[11px] uppercase tracking-[0.38em] text-white/40">
        003 — THE HELD-OUT FIELD
      </p>
      <h2 className="max-w-3xl font-serif text-[clamp(2.4rem,6vw,5.2rem)] italic leading-[0.95] tracking-tight">
        The curve,
        <span className="block not-italic text-white/45">then the gate.</span>
      </h2>
      <p className="mt-6 max-w-2xl text-sm leading-relaxed tracking-wide text-white/45">
        Detector quality is measured as COCO mAP on {HELD_OUT.fields} held-out
        fields ({HELD_OUT.instances} instances). The lab does not count on that
        curve. It infers at 0.40, then keeps RBC ≥ 0.60, WBC ≥ 0.40, platelets ≥
        0.40.
      </p>

      <div className="mt-16 grid gap-4 md:grid-cols-3">
        {[
          ["mAP@50", current.map50, baseline.map50, "%"],
          ["mAP@50-95", current.map50_95, baseline.map50_95, "%"],
          ["Precision", current.precision, baseline.precision, "%"],
        ].map(([label, now, before, suffix]) => (
          <div
            key={String(label)}
            className="border border-white/10 bg-white/[0.02] px-5 py-6"
          >
            <p className="text-[10px] uppercase tracking-[0.32em] text-white/40">
              {label}
            </p>
            <p className="mt-3 font-serif text-5xl italic leading-none">
              {now}
              <span className="text-2xl text-white/40">{suffix}</span>
            </p>
            <p className="mt-4 text-[10px] uppercase tracking-[0.24em] text-white/35">
              baseline {before}
              {suffix}
            </p>
          </div>
        ))}
      </div>

      <p className="mt-4 text-[10px] uppercase tracking-[0.28em] text-white/30">
        +{delta} mAP@50 vs our own baseline after label correction
      </p>

      <div className="mt-16">
        <p className="mb-6 text-[11px] uppercase tracking-[0.32em] text-white/40">
          Per-class mAP@50
        </p>
        <div className="grid gap-8 sm:grid-cols-3">
          {CLASSES.map((cls) => {
            const now = current.perClassMap50[cls];
            const before = baseline.perClassMap50[cls];
            return (
              <div key={cls}>
                <div className="mb-3 flex items-baseline justify-between">
                  <p
                    className="text-[10px] uppercase tracking-[0.32em]"
                    style={{ color: CLASS_COLORS[cls] }}
                  >
                    {cls}
                  </p>
                  <p className="font-serif text-2xl italic">
                    {now}
                    <span className="text-sm text-white/35">%</span>
                  </p>
                </div>
                <Bar value={now} color={CLASS_COLORS[cls]} />
                <p className="mt-3 text-[10px] uppercase tracking-[0.24em] text-white/30">
                  baseline {before}%
                </p>
              </div>
            );
          })}
        </div>
      </div>

      <p className="mt-12 max-w-2xl text-[10px] uppercase leading-relaxed tracking-[0.28em] text-white/30">
        {HELD_OUT.note} Platelets remain the hard class. Research prototype ·
        not a diagnostic device.
      </p>
    </section>
  );
}
