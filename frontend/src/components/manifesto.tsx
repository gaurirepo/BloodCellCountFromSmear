"use client";

import { useLayoutEffect, useRef } from "react";
import { getGsap } from "@/lib/gsap";

const LINES = [
  "Counting 500 cells by hand",
  "is a great way to lose your sanity.",
  "We taught YOLO to do the staring,",
  "so clinicians don't have to.",
];

export function Manifesto() {
  const root = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    const { gsap } = getGsap();
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ctx = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((line) => {
        gsap.from(line.querySelector("span"), {
          yPercent: reduced ? 0 : 110,
          rotate: reduced ? 0 : 6,
          duration: reduced ? 0.01 : 1.15,
          ease: "power4.out",
          scrollTrigger: {
            trigger: line,
            start: "top 86%",
          },
        });
      });
    }, root);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={root}
      className="relative mx-auto grid min-h-0 w-[min(96vw,1180px)] items-center gap-16 px-6 py-32 lg:grid-cols-[1.1fr_0.9fr]"
    >
      <div>
        <p className="mb-10 text-[11px] uppercase tracking-[0.38em] text-white/40">
          001 — THE EYE SAVER
        </p>
        <div className="space-y-2">
          {LINES.map((line) => (
            <p
              key={line}
              data-reveal
              className="clip-line font-serif text-[clamp(1.6rem,3.4vw,2.7rem)] italic leading-[1.2] text-[#f4efe6]"
            >
              <span className="inline-block">{line}</span>
            </p>
          ))}
        </div>
        <p className="mt-10 text-[11px] uppercase tracking-[0.28em] text-white/35">
          85.4% mAP@50 · 36 held-out fields · research prototype
        </p>
      </div>

      <div>
        <div className="overflow-hidden rounded-[2px] border border-white/10 bg-black">
          <img
            src="/smears/smear-manifesto.png"
            alt="Wright–Giemsa smear at 100x"
            className="w-full object-contain"
          />
        </div>
        <p className="mt-3 text-[10px] uppercase tracking-[0.28em] text-white/35">
          Wright–Giemsa · 100x
        </p>
      </div>

      <ol className="grid gap-5 lg:col-span-2 sm:grid-cols-3">
        {[
          ["01", "Detect", "Localize every cell in the field."],
          ["02", "Classify", "RBC, WBC, or platelet at live gates."],
          ["03", "Count", "Per-class totals for this field of view."],
        ].map(([n, title, body]) => (
          <li key={n} className="border-t border-white/10 pt-4">
            <p className="text-[10px] uppercase tracking-[0.32em] text-white/30">
              {n} · {title}
            </p>
            <p className="mt-2 text-sm leading-relaxed text-white/55">{body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
