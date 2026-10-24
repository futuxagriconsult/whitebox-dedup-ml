import { MARSScorer, MARSModelData } from "./mars_scorer.js";
import { ICPPredictor, ICPCalibrationData } from "./conformal.js";
import { DecisionRouter, RoutingDecision, Tier } from "./router.js";
import { extractPairFeatures } from "./features.js";
import Papa from "papaparse";

export interface DeduplicationSummary {
  totalRecords: number;
  candidatePairsEvaluated: number;
  tier1AutoMerge: number;
  tier2HumanReview: number;
  tier3AutoDistinct: number;
}

export class DeduplicationPipeline {
  constructor(
    public records: Record<string, string>[],
    public router: DecisionRouter
  ) {}

  static fromCSV(
    csvContent: string,
    modelData: MARSModelData,
    calibrationData?: ICPCalibrationData
  ): DeduplicationPipeline {
    const parsed = Papa.parse<Record<string, string>>(csvContent, {
      header: true,
      skipEmptyLines: true,
    });
    const scorer = new MARSScorer(modelData);
    const icp = calibrationData ? new ICPPredictor(calibrationData) : null;
    const router = new DecisionRouter(scorer, icp);
    return new DeduplicationPipeline(parsed.data, router);
  }

  scorePair(
    recordA: Record<string, string | number | null | undefined>,
    recordB: Record<string, string | number | null | undefined>
  ): RoutingDecision {
    const feats = extractPairFeatures(recordA, recordB);
    return this.router.route(feats);
  }

  run(): {
    summary: DeduplicationSummary;
    results: RoutingDecision[];
    autoMerge: RoutingDecision[];
    humanReview: RoutingDecision[];
    autoDistinct: RoutingDecision[];
  } {
    const results: RoutingDecision[] = [];
    const autoMerge: RoutingDecision[] = [];
    const humanReview: RoutingDecision[] = [];
    const autoDistinct: RoutingDecision[] = [];

    const n = this.records.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dec = this.scorePair(this.records[i]!, this.records[j]!);
        results.push(dec);
        if (dec.tier === Tier.AUTO_MERGE) autoMerge.push(dec);
        else if (dec.tier === Tier.HUMAN_REVIEW) humanReview.push(dec);
        else autoDistinct.push(dec);
      }
    }

    return {
      summary: {
        totalRecords: n,
        candidatePairsEvaluated: results.length,
        tier1AutoMerge: autoMerge.length,
        tier2HumanReview: humanReview.length,
        tier3AutoDistinct: autoDistinct.length,
      },
      results,
      autoMerge,
      humanReview,
      autoDistinct,
    };
  }
}
