# @futux/whitebox-dedup (Interpretable Entity Deduplication in TypeScript)

[![npm version](https://img.shields.io/npm/v/@futux/whitebox-dedup.svg?color=0B3541&logo=npm&logoColor=white)](https://www.npmjs.com/package/@futux/whitebox-dedup)
[![npm downloads](https://img.shields.io/npm/dm/@futux/whitebox-dedup.svg)](https://www.npmjs.com/package/@futux/whitebox-dedup)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Web Demo](https://img.shields.io/badge/Interactive_Playground-Live_Online-0FBFF1?style=flat&logo=google-chrome&logoColor=white)](https://futuxagriconsult.github.io/whitebox-dedup-ml/)

> **Auditable, transparent entity deduplication and record linkage with Multivariate Adaptive Regression Splines (MARS) and Inductive Conformal Prediction (ICP).** High-performance, zero-native-dependency TypeScript library for Node.js, Next.js, and browser Web Workers.

---

### 🌐 Try the Interactive Web Playground

Test `@futux/whitebox-dedup` live directly in your browser:  
👉 **[Launch Live Web Demo (https://futuxagriconsult.github.io/whitebox-dedup-ml/)](https://futuxagriconsult.github.io/whitebox-dedup-ml/)**

---

## 🚀 Key Features

- **Auditable MARS Scoring:** No black-box neural networks. Outputs clear hinge-function math explaining precisely why two entities matched.
- **Three Conformal Statistical Tiers:**
  - **Tier 1 (Auto-Merge, $\ge 0.85$):** Conformal $p$-value $\ge 0.95$; statistically safe for automated merging.
  - **Tier 2 (Circuit Breaker / Review, $0.60 - 0.84$):** Ambiguous boundary cases routed to human adjudicators with exact contributing features.
  - **Tier 3 (Auto-Reject, $< 0.60$):** Distinct records; kept separate.
- **Client & Server Compatible:** Runs in Node.js, Bun, Deno, Next.js API routes, and client-side Web Workers.
- **Zero Heavy Native Dependencies:** Fast cold-start, pure TypeScript, instant browser compilation.

---

## 📦 Installation

```bash
# npm
npm install @futux/whitebox-dedup

# yarn
yarn add @futux/whitebox-dedup

# pnpm
pnpm add @futux/whitebox-dedup
```

---

## 💡 Quick Start

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
  console.log(`Auditable Basis:`, match.explanation);
});
```

---

## 🔬 How the Math Works

The score is calculated via transparent piecewise linear splines:

```text
Score = 1.0 / (1.0 + exp(-(
    -3.8421
    + 5.1204 * max(0, jaro_winkler - 0.7200)
    - 2.8910 * max(0, 0.7200 - jaro_winkler)
    + 1.8432 * max(0, token_sort_ratio - 0.6500)
    + 0.9420 * exact_phone_match
)))
```

Every prediction can be audited, reviewed, and justified in regulatory or administrative hearings.

---

## 🐍 Also Available in Python

For Python data pipelines and pandas integration, install the companion package:
```bash
pip install whitebox-dedup
```
See [GitHub Repository](https://github.com/futuxagriconsult/whitebox-dedup-ml) for CLI and calibration tools.

---

## 📄 License

MIT. Developed and maintained by **FutuX Agri-consult Limited** ([info@futuxconsult.com](mailto:info@futuxconsult.com)).
