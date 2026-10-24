import { MARSScorer, MARSTraceResult } from "./mars_scorer.js";
import { ICPPredictor } from "./conformal.js";

export enum Tier {
  AUTO_MERGE = "auto_merge",
  HUMAN_REVIEW = "human_review",
  AUTO_DISTINCT = "auto_distinct",
}

export interface RoutingDecision {
  tier: Tier;
  probability: number;
  predictionSet: number[];
  reason: string;
  features: Record<string, number | null>;
  marsTrace?: MARSTraceResult;
}

export class DecisionRouter {
  constructor(
    public scorer: MARSScorer,
    public icp: ICPPredictor | null = null,
    public dupThreshold: number = 0.898,
    public distThreshold: number = 0.491
  ) {}

  route(
    features: Record<string, number | null>,
    hardSignals: { isExactIdMatch?: boolean; isExactIdMismatch?: boolean } = {}
  ): RoutingDecision {
    // 1. Hard signals override
    if (hardSignals.isExactIdMatch) {
      return {
        tier: Tier.AUTO_MERGE,
        probability: 1.0,
        predictionSet: [1],
        reason: "hard_signal:exact_id_match",
        features,
      };
    }
    if (hardSignals.isExactIdMismatch) {
      return {
        tier: Tier.AUTO_DISTINCT,
        probability: 0.0,
        predictionSet: [0],
        reason: "hard_signal:exact_id_mismatch",
        features,
      };
    }

    // 2. Probabilistic evaluation via MARS
    const trace = this.scorer.scoreWithTrace(features);
    const prob = trace.probability;

    // 3. Conformal Prediction Set Check
    let pset = [0, 1];
    if (this.icp) {
      const setObj = this.icp.predictSet(prob);
      pset = Array.from(setObj);
    }

    if (pset.length === 1 && pset[0] === 1) {
      return {
        tier: Tier.AUTO_MERGE,
        probability: prob,
        predictionSet: pset,
        reason: `mars:high_confidence_duplicate(p=${prob.toFixed(3)})`,
        features,
        marsTrace: trace,
      };
    }

    if (pset.length === 1 && pset[0] === 0) {
      return {
        tier: Tier.AUTO_DISTINCT,
        probability: prob,
        predictionSet: pset,
        reason: `mars:high_confidence_distinct(p=${prob.toFixed(3)})`,
        features,
        marsTrace: trace,
      };
    }

    // Circuit Breaker triggered
    return {
      tier: Tier.HUMAN_REVIEW,
      probability: prob,
      predictionSet: pset,
      reason: `circuit_breaker:ambiguous_region(p=${prob.toFixed(3)},set={0,1})`,
      features,
      marsTrace: trace,
    };
  }
}
