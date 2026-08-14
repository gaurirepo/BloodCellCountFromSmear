export type AnalyzeCounts = {
  rbc: number;
  wbc: number;
  platelets: number;
  total: number;
};

export type AnalyzeBox = {
  class: string;
  confidence: number;
  box: [number, number, number, number];
};

export type AnalyzeResponse = {
  counts: AnalyzeCounts;
  boxes: AnalyzeBox[];
  annotated_image_base64: string;
};

export const CLASS_COLORS = {
  RBC: "#ef4444",
  WBC: "#3b82f6",
  Platelets: "#f59e0b",
} as const;

export type CellClass = keyof typeof CLASS_COLORS;

export const CELL_CLASSES = ["RBC", "WBC", "Platelets"] as const;

const ANALYZE_URL = "/api/analyze";

export function annotatedSrc(base64: string) {
  return `data:image/png;base64,${base64}`;
}

export async function pingEngine(): Promise<boolean> {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const body = new FormData();
  body.append("file", file);

  let response: Response;
  try {
    response = await fetch(ANALYZE_URL, {
      method: "POST",
      body,
    });
  } catch {
    throw new Error("Cannot reach the inference engine.");
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The engine returned a response that could not be parsed.");
  }

  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof (payload as { detail: unknown }).detail === "string"
        ? (payload as { detail: string }).detail
        : `Engine error (${response.status}).`;
    throw new Error(detail);
  }

  const data = payload as Partial<AnalyzeResponse>;
  if (
    !data.counts ||
    typeof data.annotated_image_base64 !== "string" ||
    !Array.isArray(data.boxes)
  ) {
    throw new Error("Unexpected analyze payload from the engine.");
  }

  return data as AnalyzeResponse;
}

export async function fileFromUrl(url: string, name: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Could not load the selected sample field.");
  }
  const blob = await response.blob();
  const type = blob.type || "image/jpeg";
  return new File([blob], name, { type });
}

export function meanConfidence(boxes: { confidence: number }[]) {
  if (!boxes.length) return null;
  const avg = boxes.reduce((sum, box) => sum + box.confidence, 0) / boxes.length;
  return Math.round(avg * 100);
}

export function meanConfidenceForClass(boxes: AnalyzeBox[], cls: string) {
  return meanConfidence(boxes.filter((box) => box.class === cls));
}

export const DEFAULT_GATES: Record<CellClass, number> = {
  RBC: 0.6,
  WBC: 0.4,
  Platelets: 0.4,
};

export const MIN_INFER_CONF = 0.4;
export const MAX_GATE = 0.95;

export function applyGates(
  boxes: AnalyzeBox[],
  gates: Record<CellClass, number>,
) {
  return boxes.filter((box) => {
    const gate = gates[box.class as CellClass];
    return gate == null ? false : box.confidence >= gate;
  });
}

export function countsFromBoxes(boxes: AnalyzeBox[]): AnalyzeCounts {
  const counts: AnalyzeCounts = { rbc: 0, wbc: 0, platelets: 0, total: 0 };
  for (const box of boxes) {
    if (box.class === "RBC") counts.rbc += 1;
    else if (box.class === "WBC") counts.wbc += 1;
    else if (box.class === "Platelets") counts.platelets += 1;
  }
  counts.total = counts.rbc + counts.wbc + counts.platelets;
  return counts;
}

export function gatesAreDefault(gates: Record<CellClass, number>) {
  return CELL_CLASSES.every(
    (cls) => Math.abs(gates[cls] - DEFAULT_GATES[cls]) < 0.001,
  );
}

export function formatGates(gates: Record<CellClass, number>) {
  return CELL_CLASSES.map((cls) => gates[cls].toFixed(2)).join(" / ");
}

export function wbcRbcRatio(counts: AnalyzeCounts) {
  if (!counts.rbc) return null;
  return counts.wbc / counts.rbc;
}

function slugName(name: string) {
  return name.replace(/\.[^.]+$/, "").replace(/\s+/g, "-").toLowerCase() || "smear";
}

export function downloadAnnotatedPng(base64: string, name: string) {
  const link = document.createElement("a");
  link.href = annotatedSrc(base64);
  link.download = `${slugName(name)}-overlay.png`;
  link.click();
}

export function downloadBoxesCsv(boxes: AnalyzeBox[], name: string) {
  const rows = [
    "class,confidence,x1,y1,x2,y2",
    ...boxes.map(
      (box) =>
        `${box.class},${box.confidence},${box.box.map((value) => value.toFixed(1)).join(",")}`,
    ),
  ];
  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${slugName(name)}-boxes.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}
