"use client";

import { useLayoutEffect, useRef } from "react";
import { getGsap } from "@/lib/gsap";
import { EngineDot } from "./engine-dot";

export function Hero() {
  const root = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    const { gsap } = getGsap();
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const ctx = gsap.context(() => {
      const words = gsap.utils.toArray<HTMLElement>("[data-hero-word]");

      gsap.set(words, { yPercent: 120, rotate: 4, opacity: 0 });
      gsap.to(words, {
        yPercent: 0,
        rotate: 0,
        opacity: 1,
        duration: reduced ? 0.01 : 1.35,
        stagger: reduced ? 0 : 0.12,
        ease: "power4.out",
        delay: reduced ? 0 : 0.15,
      });

      gsap.from("[data-hero-kicker], [data-hero-sub]", {
        y: reduced ? 0 : 24,
        opacity: 0,
        duration: reduced ? 0.01 : 1,
        delay: reduced ? 0 : 0.1,
        stagger: reduced ? 0 : 0.08,
        ease: "power3.out",
      });
    }, root);

    return () => {
      ctx.revert();
    };
  }, []);

  return (
    <section
      ref={root}
      className="relative flex h-screen w-full items-center justify-center overflow-hidden"
    >
      <div data-hero-smear className="absolute inset-0 scale-110 opacity-45">
        <img
          src="/smears/smear-03.jpg"
          alt=""
          className="h-full w-full object-cover contrast-125 saturate-[0.7] mix-blend-luminosity"
        />
        <div
          className="absolute inset-0 mix-blend-multiply"
          style={{
            background:
              "radial-gradient(1200px 700px at 50% 40%, #4a0414 0%, #2b0610 48%, #1a0208 100%)",
          }}
        />
      </div>

      <div className="absolute inset-0 bg-gradient-to-b from-[#1a0208]/80 via-[#2b0610]/55 to-[#140308]" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[1] h-44 bg-gradient-to-b from-transparent to-[#140308]" />

      <header className="absolute left-8 top-8 z-20 flex items-center gap-3 text-[11px] uppercase tracking-[0.32em] text-white/55">
        <EngineDot />
        <span className="text-white/25">·</span>
        SmearDx
      </header>

      <a
        href="#lab"
        className="absolute right-8 top-8 z-20 text-[11px] uppercase tracking-[0.28em] text-white/70 transition hover:text-white"
      >
        Enter the lab
      </a>

      <div className="relative z-10 mx-auto w-[min(96vw,1180px)] px-6 text-center">
        <p
          data-hero-kicker
          className="mb-8 text-[11px] uppercase tracking-[0.48em] text-white/50"
        >
          YOLO26n · RBC / WBC / platelets
        </p>

        <h1 className="hero-title text-[clamp(2.8rem,11vw,8.6rem)]">
          <span data-hero-line className="clip-line">
            <span data-hero-word className="inline-block pr-[0.12em]">
              DETECT.
            </span>
          </span>
          <span data-hero-line className="clip-line">
            <span data-hero-word className="inline-block pr-[0.18em]">
              CLASSIFY.
            </span>
            <span data-hero-word className="inline-block">
              COUNT.
            </span>
          </span>
        </h1>

        <p
          data-hero-sub
          className="mx-auto mt-8 max-w-xl text-sm tracking-wide text-white/45"
        >
          Computer-aided hematology from a single smear. Localize, name, and
          quantify cells in one field of view.
        </p>
      </div>
    </section>
  );
}
