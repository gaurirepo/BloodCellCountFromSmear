"use client";

import { useEngineStatus } from "@/hooks/use-engine-status";

export function EngineDot() {
  const online = useEngineStatus();
  const color =
    online === null ? "bg-white/35" : online ? "bg-emerald-400" : "bg-red-500";
  const glow =
    online === true
      ? "shadow-[0_0_12px_#34d399]"
      : online === false
        ? "shadow-[0_0_12px_#ef4444]"
        : "";
  const label =
    online === null ? "checking" : online ? "engine live" : "engine offline";

  return (
    <span className="inline-flex items-center gap-3">
      <span className={`h-1.5 w-1.5 rounded-full ${color} ${glow}`} />
      <span className="text-white/40">{label}</span>
    </span>
  );
}
