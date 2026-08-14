"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { getGsap } from "@/lib/gsap";
import {
  analyzeImage,
  applyGates,
  CELL_CLASSES,
  CLASS_COLORS,
  countsFromBoxes,
  DEFAULT_GATES,
  downloadAnnotatedPng,
  downloadBoxesCsv,
  fileFromUrl,
  formatGates,
  gatesAreDefault,
  meanConfidence,
  meanConfidenceForClass,
  wbcRbcRatio,
  type AnalyzeResponse,
  type CellClass,
} from "@/lib/analyze";
import { FieldViewer } from "./field-viewer";
import { GateSliders } from "./gate-sliders";
import { LiquidHover } from "./liquid-hover";

const SAMPLES = [
  { src: "/smears/smear-01.jpg", label: "Dense RBC" },
  { src: "/smears/smear-02.jpg", label: "WBC present" },
  { src: "/smears/smear-03.jpg", label: "Crowded field" },
  { src: "/smears/smear-04.jpg", label: "Mixed smear" },
];

const ALL_VISIBLE: Record<CellClass, boolean> = {
  RBC: true,
  WBC: true,
  Platelets: true,
};

export function Analyzer() {
  const root = useRef<HTMLElement>(null);
  const rbcRef = useRef<HTMLSpanElement>(null);
  const wbcRef = useRef<HTMLSpanElement>(null);
  const pltRef = useRef<HTMLSpanElement>(null);
  const countRefs: Record<CellClass, typeof rbcRef> = {
    RBC: rbcRef,
    WBC: wbcRef,
    Platelets: pltRef,
  };

  const [original, setOriginal] = useState<string | null>(null);
  const [fileName, setFileName] = useState("Awaiting specimen");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [visible, setVisible] = useState(ALL_VISIBLE);
  const [gates, setGates] = useState({ ...DEFAULT_GATES });
  const skipGateAnim = useRef(true);

  const gatedBoxes = result ? applyGates(result.boxes, gates) : [];
  const gatedCounts = countsFromBoxes(gatedBoxes);

  useLayoutEffect(() => {
    if (!result) return;
    skipGateAnim.current = true;
    const { gsap } = getGsap();
    const ctx = gsap.context(() => {
      gsap.from("[data-split]", {
        y: 28,
        opacity: 0,
        duration: 0.9,
        stagger: 0.07,
        ease: "power3.out",
      });

      const tick = (el: HTMLSpanElement | null, end: number) => {
        if (!el) return;
        const proxy = { val: 0 };
        el.textContent = "0";
        gsap.to(proxy, {
          val: end,
          duration: 1.55,
          ease: "power2.out",
          onUpdate: () => {
            el.textContent = String(Math.round(proxy.val));
          },
        });
      };

      tick(rbcRef.current, gatedCounts.rbc);
      tick(wbcRef.current, gatedCounts.wbc);
      tick(pltRef.current, gatedCounts.platelets);
    }, root);

    return () => ctx.revert();
  }, [result]);

  useLayoutEffect(() => {
    if (!result) return;
    if (skipGateAnim.current) {
      skipGateAnim.current = false;
      return;
    }
    const { gsap } = getGsap();
    const tick = (el: HTMLSpanElement | null, end: number) => {
      if (!el) return;
      const proxy = { val: Number(el.textContent) || 0 };
      gsap.to(proxy, {
        val: end,
        duration: 0.28,
        ease: "power2.out",
        onUpdate: () => {
          el.textContent = String(Math.round(proxy.val));
        },
      });
    };
    tick(rbcRef.current, gatedCounts.rbc);
    tick(wbcRef.current, gatedCounts.wbc);
    tick(pltRef.current, gatedCounts.platelets);
  }, [gatedCounts.rbc, gatedCounts.wbc, gatedCounts.platelets, result]);

  const reset = () => {
    setOriginal(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setFileName("Awaiting specimen");
    setVisible(ALL_VISIBLE);
    setGates({ ...DEFAULT_GATES });
  };

  const runAnalysis = async (file: File, previewUrl: string, label: string) => {
    setError(null);
    setResult(null);
    setOriginal(previewUrl);
    setFileName(label);
    setLoading(true);
    setVisible(ALL_VISIBLE);
    setGates({ ...DEFAULT_GATES });

    try {
      const payload = await analyzeImage(file);
      setResult(payload);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Analysis failed. Confirm the inference engine is running.",
      );
    } finally {
      setLoading(false);
    }
  };

  const onFile = (file: File) => {
    const url = URL.createObjectURL(file);
    void runAnalysis(file, url, file.name);
  };

  const onSample = async (src: string, label: string) => {
    try {
      const file = await fileFromUrl(
        src,
        `${label.replace(/\s+/g, "-").toLowerCase()}.jpg`,
      );
      await runAnalysis(file, src, label);
    } catch (err) {
      setOriginal(src);
      setFileName(label);
      setLoading(false);
      setError(
        err instanceof Error ? err.message : "Could not send the sample field.",
      );
    }
  };

  const avgConf = result ? meanConfidence(gatedBoxes) : null;
  const ratio = result ? wbcRbcRatio(gatedCounts) : null;
  const gatesLabel = formatGates(gates);

  return (
    <section
      id="lab"
      ref={root}
      className="relative mx-auto w-[min(96vw,1180px)] px-6 pb-32 pt-10"
    >
      <p className="mb-4 text-[11px] uppercase tracking-[0.38em] text-white/40">
        002 — THE SPECIMEN
      </p>
      <h2 className="max-w-3xl font-serif text-[clamp(2.4rem,6vw,5.2rem)] italic leading-[0.95] tracking-tight">
        Place the
        <span className="block not-italic text-white/45">specimen.</span>
      </h2>

      {!original ? (
        <div className="mt-16">
          <label
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              const file = event.dataTransfer.files[0];
              if (file) onFile(file);
            }}
            className="group block cursor-pointer"
          >
            <LiquidHover className="rounded-sm border border-dashed border-white/20 bg-white/[0.02] px-8 py-20 text-center transition group-hover:border-white/45">
              <input
                type="file"
                accept="image/jpeg,image/png,image/bmp,image/webp"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onFile(file);
                }}
              />
              <p className="font-serif text-3xl italic">Drop a microscopic field</p>
              <p className="mt-3 text-sm tracking-wide text-white/45">
                JPG, PNG, BMP · Wright–Giemsa fields work best
              </p>
            </LiquidHover>
          </label>

          <div className="mt-10 grid grid-cols-2 gap-4 md:grid-cols-4">
            {SAMPLES.map((sample) => (
              <button
                key={sample.src}
                type="button"
                onClick={() => {
                  void onSample(sample.src, sample.label);
                }}
                className="text-left"
              >
                <LiquidHover className="aspect-[4/3] rounded-sm border border-white/10">
                  <img
                    src={sample.src}
                    alt={sample.label}
                    className="h-full w-full object-cover"
                  />
                </LiquidHover>
                <span className="mt-2 block text-[10px] uppercase tracking-[0.28em] text-white/40">
                  {sample.label}
                </span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="mt-14">
          <div className="mb-6 flex flex-wrap items-end justify-between gap-6">
            <div className="flex flex-wrap items-center gap-3">
              <p className="text-[11px] uppercase tracking-[0.3em] text-white/45">
                {fileName}
              </p>
              {result && !loading ? (
                <span className="rounded-full border border-white/15 bg-white/[0.04] px-3 py-1 text-[10px] uppercase tracking-[0.28em] text-white/70">
                  {gatedCounts.total} cells in this field
                </span>
              ) : null}
              {result && !loading && avgConf !== null ? (
                <span className="rounded-full border border-white/15 bg-white/[0.04] px-3 py-1 text-[10px] uppercase tracking-[0.28em] text-white/70">
                  mean conf {avgConf}%
                </span>
              ) : null}
              {loading ? (
                <span className="rounded-full border border-[#8b1e3f]/40 bg-[#8b1e3f]/15 px-3 py-1 text-[10px] uppercase tracking-[0.28em] text-[#f4c4ce]">
                  Scanning field
                </span>
              ) : null}
            </div>
            <div className="flex flex-wrap items-center gap-5">
              {result && !loading ? (
                <>
                  <button
                    type="button"
                    onClick={() =>
                      downloadAnnotatedPng(result.annotated_image_base64, fileName)
                    }
                    className="text-[11px] uppercase tracking-[0.28em] text-white/70 hover:text-white"
                  >
                    Overlay PNG
                  </button>
                  <button
                    type="button"
                    onClick={() => downloadBoxesCsv(gatedBoxes, fileName)}
                    className="text-[11px] uppercase tracking-[0.28em] text-white/70 hover:text-white"
                  >
                    Boxes CSV
                  </button>
                </>
              ) : null}
              <button
                type="button"
                onClick={reset}
                className="text-[11px] uppercase tracking-[0.28em] text-white/70 hover:text-white"
              >
                New specimen
              </button>
            </div>
          </div>

          {error ? (
            <div
              role="alert"
              className="mb-6 border border-red-400/30 bg-red-500/[0.08] px-5 py-4 text-sm tracking-wide text-red-200"
            >
              <p className="text-[10px] uppercase tracking-[0.32em] text-red-300">
                Engine fault
              </p>
              <p className="mt-2 text-red-100/90">{error}</p>
            </div>
          ) : null}

          <FieldViewer
            src={original}
            boxes={gatedBoxes}
            loading={loading}
            visible={visible}
            gatesLabel={gatesLabel}
          />

          <GateSliders
            gates={gates}
            onChange={setGates}
            onReset={() => setGates({ ...DEFAULT_GATES })}
            isDefault={gatesAreDefault(gates)}
          />

          <div className="mt-6 flex flex-wrap items-center gap-3">
            {CELL_CLASSES.map((cls) => {
              const on = visible[cls];
              return (
                <button
                  key={cls}
                  type="button"
                  onClick={() =>
                    setVisible((prev) => ({ ...prev, [cls]: !prev[cls] }))
                  }
                  className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[10px] uppercase tracking-[0.28em] transition ${
                    on
                      ? "border-white/20 bg-white/[0.06] text-white/80"
                      : "border-white/10 text-white/30"
                  }`}
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{
                      background: on ? CLASS_COLORS[cls] : "transparent",
                      boxShadow: on ? `0 0 10px ${CLASS_COLORS[cls]}` : "none",
                      border: on ? "none" : `1px solid ${CLASS_COLORS[cls]}`,
                    }}
                  />
                  {cls}
                </button>
              );
            })}
          </div>

          <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {CELL_CLASSES.map((cls) => {
              const color = CLASS_COLORS[cls];
              const conf = result
                ? meanConfidenceForClass(gatedBoxes, cls)
                : null;
              return (
                <button
                  key={cls}
                  type="button"
                  data-split
                  onClick={() =>
                    setVisible((prev) => ({ ...prev, [cls]: !prev[cls] }))
                  }
                  className={`border border-white/10 bg-white/[0.02] px-5 py-6 text-left transition ${
                    visible[cls] ? "opacity-100" : "opacity-40"
                  }`}
                >
                  <p
                    className="text-[10px] uppercase tracking-[0.32em]"
                    style={{ color }}
                  >
                    {cls}
                  </p>
                  <p className="mt-3 font-serif text-5xl italic leading-none">
                    <span ref={countRefs[cls]}>0</span>
                  </p>
                  <p className="mt-3 text-[10px] uppercase tracking-[0.24em] text-white/35">
                    {conf === null ? "no detections" : `mean conf ${conf}%`}
                  </p>
                </button>
              );
            })}
          </div>

          {result && !loading ? (
            <div className="mt-6 space-y-2 text-[10px] uppercase tracking-[0.28em] text-white/30">
              <p>
                WBC:RBC {ratio === null ? "—" : ratio.toFixed(3)} · counts are per
                field of view, not a complete blood count
              </p>
              <p>
                Research prototype · not a diagnostic device · live gates{" "}
                {gatesLabel}
              </p>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
