export interface ICPCalibrationData {
  model_name: string;
  alpha: number;
  threshold_duplicate: number;
  threshold_distinct: number;
  description?: string;
}

export class ICPPredictor {
  constructor(public cal: ICPCalibrationData) {}

  predictSet(probability: number): Set<number> {
    const nonconformityDup = 1.0 - probability;
    const nonconformityDist = probability;

    const includeDup = nonconformityDup <= this.cal.threshold_duplicate;
    const includeDist = nonconformityDist <= this.cal.threshold_distinct;

    const set = new Set<number>();
    if (includeDist) set.add(0);
    if (includeDup) set.add(1);

    if (set.size === 0) {
      // Empty set safeguard -> ambiguous
      set.add(0);
      set.add(1);
    }

    return set;
  }
}
