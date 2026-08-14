export const HELD_OUT = {
  fields: 36,
  instances: 471,
  note: "COCO mAP at conf=0.001. Counting in the lab uses live class gates, not this curve.",
  current: {
    map50: 85.4,
    map50_95: 60.07,
    precision: 82.46,
    perClassMap50: {
      WBC: 96.9,
      RBC: 85.4,
      Platelets: 73.9,
    },
  },
  baseline: {
    map50: 81.81,
    map50_95: 56.47,
    precision: 71.74,
    perClassMap50: {
      WBC: 96.9,
      RBC: 83.4,
      Platelets: 65.1,
    },
  },
} as const;
