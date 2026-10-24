export interface HingeDefinition {
  feature: string;
  coefficient: number;
  knot: number;
  reflected?: boolean;
}

export interface MARSModelData {
  model_name: string;
  description?: string;
  required_features: string[];
  intercept: number;
  hinges: HingeDefinition[];
}

export interface HingeContribution {
  feature: string;
  inputValue: number | null;
  knot: number;
  reflected: boolean;
  coefficient: number;
  hingeOutput: number;
}

export interface MARSTraceResult {
  probability: number;
  logit: number;
  intercept: number;
  hingeContributions: HingeContribution[];
}

export class HingeFunction {
  constructor(
    public feature: string,
    public coefficient: number,
    public knot: number,
    public reflected: boolean = false
  ) {}

  evaluate(x: number): number {
    if (this.reflected) {
      return this.coefficient * Math.max(0, this.knot - x);
    }
    return this.coefficient * Math.max(0, x - this.knot);
  }
}

export class MARSScorer {
  public modelName: string;
  public description: string;
  public intercept: number;
  public hinges: HingeFunction[];
  public requiredFeatures: string[];

  constructor(data: MARSModelData) {
    this.modelName = data.model_name;
    this.description = data.description ?? "";
    this.intercept = data.intercept;
    this.requiredFeatures = data.required_features;
    this.hinges = data.hinges.map(
      (h) => new HingeFunction(h.feature, h.coefficient, h.knot, Boolean(h.reflected))
    );
  }

  score(features: Record<string, number | null | undefined>): number {
    let logit = this.intercept;
    for (const hinge of this.hinges) {
      const val = features[hinge.feature];
      if (val !== null && val !== undefined) {
        logit += hinge.evaluate(val);
      }
    }
    if (logit >= 50.0) return 1.0;
    if (logit <= -50.0) return 0.0;
    return 1.0 / (1.0 + Math.exp(-logit));
  }

  scoreWithTrace(features: Record<string, number | null | undefined>): MARSTraceResult {
    let logit = this.intercept;
    const contributions: HingeContribution[] = [];

    for (const hinge of this.hinges) {
      const val = features[hinge.feature] ?? null;
      const contribution = val !== null ? hinge.evaluate(val) : 0.0;
      logit += contribution;
      contributions.push({
        feature: hinge.feature,
        inputValue: val,
        knot: hinge.knot,
        reflected: hinge.reflected,
        coefficient: hinge.coefficient,
        hingeOutput: contribution,
      });
    }

    let prob = 0.0;
    if (logit >= 50.0) prob = 1.0;
    else if (logit <= -50.0) prob = 0.0;
    else prob = 1.0 / (1.0 + Math.exp(-logit));

    return {
      probability: Number(prob.toFixed(6)),
      logit: Number(logit.toFixed(6)),
      intercept: this.intercept,
      hingeContributions: contributions,
    };
  }
}
