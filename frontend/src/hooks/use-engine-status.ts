"use client";

import { useEffect, useState } from "react";
import { pingEngine } from "@/lib/analyze";

export function useEngineStatus() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      const ok = await pingEngine();
      if (!cancelled) setOnline(ok);
    };

    void tick();
    const id = window.setInterval(() => {
      void tick();
    }, 8000);

    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return online;
}
