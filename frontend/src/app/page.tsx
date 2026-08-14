import { Analyzer } from "@/components/analyzer";
import { Hero } from "@/components/hero";
import { Manifesto } from "@/components/manifesto";
import { Report } from "@/components/report";

export default function Home() {
  return (
    <main>
      <Hero />
      <Manifesto />
      <Analyzer />
      <Report />
      <footer className="border-t border-white/10 px-8 py-10 text-center text-[10px] uppercase tracking-[0.32em] text-white/35">
        SmearDx · research prototype · not a diagnostic device
      </footer>
    </main>
  );
}
