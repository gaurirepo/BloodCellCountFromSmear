"use client";

import { useRef } from "react";

export function LiquidHover({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const onMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const node = ref.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    node.style.setProperty("--mx", `${x}%`);
    node.style.setProperty("--my", `${y}%`);
    const rx = ((y - 50) / 50) * -6;
    const ry = ((x - 50) / 50) * 8;
    node.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg)`;
  };

  return (
    <div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={() => {
        if (ref.current) ref.current.style.transform = "";
      }}
      className={`liquid-frame transition-transform duration-500 ease-out ${className}`}
    >
      {children}
    </div>
  );
}
