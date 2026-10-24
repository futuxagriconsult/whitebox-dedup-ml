# @futux/whitebox-dedup

[![npm version](https://img.shields.io/npm/v/@futux/whitebox-dedup.svg?color=0B3541)](https://www.npmjs.com/package/@futux/whitebox-dedup)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Playground](https://img.shields.io/badge/Live_Playground-GitHub_Pages-brightgreen.svg)](https://futuxagriconsult.github.io/whitebox-dedup-ml/)

Auditable, interpretable entity deduplication with Multivariate Adaptive Regression Splines (MARS) and Inductive Conformal Prediction (ICP). Built for high-stakes Digital Public Infrastructure (DPI) registries where black-box ML decisions cannot be legally or ethically audited.

Developed by **FutuX Agri-consult Limited**.

---

## Installation

```bash
npm install @futux/whitebox-dedup
```

Or with Yarn / pnpm:

```bash
yarn add @futux/whitebox-dedup
# or
pnpm add @futux/whitebox-dedup
```

---

## Key Features

- **Auditable MARS Scoring**: Non-linear hinge-function equations that output transparent, human-readable math explaining exactly why two records match.
- **Three Conformal Prediction Tiers**:
  - **Tier 1 (Auto-Merge, $\ge 0.85$)**: Conformal p-value exceeds significance threshold ($\alpha = 0.05$); safe for automated merging.
  - **Tier 2 (Circuit Breaker / Review, $0.60 - 0.84$)**: Ambiguous pairs routed to human enumerators or adjudicators with exact contributing features.
  - **Tier 3 (Auto-Reject, $< 0.60$)**: Distinct entities; kept separate.
- **Dynamic Feature Adaptability**: Evaluates string similarity (Jaro-Winkler, Levenshtein, Token Sort), exact phone matches, geo-distance, and custom attributes.

---

## Quickstart

```typescript
import { DeduplicationPipeline } from "@futux/whitebox-dedup";

const records = [
  { id: "1", name: "Ibrahim Musa", phone: "+254712345678" },
  { id: "2", name: "Ibraheem Mussa", phone: "+254712345678" },
  { id: "3", name: "Sarah Cherono", phone: "+254787654321" }
];

const pipeline = new DeduplicationPipeline({
  model: "text_phone_v1",
  confidence: 0.95
});

const results = pipeline.run(records);

results.matches.forEach(match => {
  console.log(`Pair: ${match.recordA.name} <-> ${match.recordB.name}`);
  console.log(`Score: ${(match.score * 100).toFixed(1)}% | Tier: ${match.tier}`);
  console.log(`Auditable Basis: ${JSON.stringify(match.explanation)}`);
});
```

---

## Interactive Playground

Test with synthetic or custom tabular datasets directly in the browser:
[https://futuxagriconsult.github.io/whitebox-dedup-ml/](https://futuxagriconsult.github.io/whitebox-dedup-ml/)

---

## License

MIT. Maintained by **FutuX Agri-consult Limited** ([info@futuxconsult.com](mailto:info@futuxconsult.com)).
