# Whitebox-Dedup: Interpretable Entity Deduplication & Record Linkage

[![PyPI version](https://img.shields.io/pypi/v/whitebox-dedup.svg?color=0FBFF1&logo=pypi&logoColor=white)](https://pypi.org/project/whitebox-dedup/)
[![npm version](https://img.shields.io/npm/v/@futux/whitebox-dedup.svg?color=0B3541&logo=npm&logoColor=white)](https://www.npmjs.com/package/@futux/whitebox-dedup)
[![Python Versions](https://img.shields.io/pypi/pyversions/whitebox-dedup.svg)](https://pypi.org/project/whitebox-dedup/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23612975.svg)](https://doi.org/10.5281/zenodo.23612975)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Web Demo](https://img.shields.io/badge/Interactive_Playground-Live_Online-0FBFF1?style=flat&logo=google-chrome&logoColor=white)](https://futuxagriconsult.github.io/whitebox-dedup-ml/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/futuxagriconsult/whitebox-dedup-ml/blob/main/CONTRIBUTING.md)

> **Interpretable, auditable entity deduplication with Multivariate Adaptive Regression Splines (MARS) and Inductive Conformal Prediction (ICP).** Replaces black-box "deep learning" matching models with human-auditable mathematical equations and statistical certainty tiers for high-stakes Digital Public Infrastructure (DPI) registries.

---

### 🌐 Live Interactive Playground

Test Whitebox-Dedup directly in your browser with zero installation:  
👉 **[Launch Live Web Demo (https://futuxagriconsult.github.io/whitebox-dedup-ml/)](https://futuxagriconsult.github.io/whitebox-dedup-ml/)**

*Features in-browser client-side Web Worker execution, Starter Template CSV download, auditable basis formula inspector, CSV results export, and dual-position pagination.*

---

## ⚡ Quick Start

### 1. Installation

```bash
# Python (PyPI)
pip install whitebox-dedup

# JavaScript / TypeScript (npm)
npm install @futux/whitebox-dedup
```

### 2. Python 30-Second Example

Run entity deduplication on any CSV or list of records:

```python
from whitebox_dedup import DeduplicationPipeline

# Auto-routes features and executes MARS + Conformal scoring
pipeline = DeduplicationPipeline.from_csv("farmers_registry.csv", model="text_phone_v1")
results = pipeline.run()

for match in results.matches:
    print(f"Match: {match.record_a['name']} <-> {match.record_b['name']}")
    print(f"Score: {match.score * 100:.1f}% | Tier: {match.tier}")
    print(f"Auditable Explanation: {match.explanation}\n")

# Save auditable output for legal or enumerator review
results.to_json("dedup_audit_results.json")
```

### 3. Command-Line Interface (CLI)

Run deduplication or inspect mathematical basis equations directly from the terminal:

```bash
# Deduplicate any CSV dataset
wdedup deduplicate --input registry.csv --output results.json --confidence 0.95

# Print the exact mathematical MARS equation of any pre-trained model
wdedup show-equation --model text_phone_v1

# Calibrate conformal nonconformity quantiles
wdedup calibrate --model text_only_v1 --pairs calibration_pairs.csv
```

---

## 🎯 Why Whitebox Deduplication?

In national identity systems, smallholder farmer registries, health records, and social cash transfers, duplicate records lead to fraud, phantom beneficiaries, or wrongful exclusion.

When governments deploy black-box entity resolution algorithms (e.g. Random Forests, XGBoost, or Deep Embeddings), they face critical challenges:
1. **The Explanation Problem:** An administrator or enumerator cannot explain *why* two farmers were merged or rejected.
2. **The Legal & Ethical Audit Challenge:** Black-box models cannot provide auditable mathematical proofs in administrative appeals.
3. **The Silent Failure Mode:** High confidence false-positives silently merge distinct human beings.

**Whitebox-Dedup solves this with two foundational pillars:**
1. **Auditable MARS Basis Splines:** Every prediction is computed using explicit, piece-wise linear hinge functions with exact coefficients. Every decision can be written out on paper with a pen.
2. **Three-Tier Conformal Prediction:** Instead of arbitrary thresholding, Inductive Conformal Prediction provides mathematically guaranteed error bounds:
   - **Tier 1 — Auto-Merge ($\ge 0.85$):** Conformal $p$-value exceeds significance threshold ($\alpha = 0.05$). Guaranteed error rate bounded by $\alpha$.
   - **Tier 2 — Circuit Breaker / Human Review ($0.60 - 0.84$):** Ambiguous boundary cases routed to human adjudicators with explicit contributing feature weights.
   - **Tier 3 — Auto-Reject ($< 0.60$):** Distinct entities; safely kept separate.

---

## 📐 The Auditable Mathematical Equation

For the `text_only_v1` model, the similarity score $\hat{y}$ is computed as:

$$\hat{y} = \sigma \left( \beta_0 + \sum_{m=1}^{M} \beta_m \cdot h(x) \right)$$

where $h(x)$ are hinge functions $\max(0, x - c)$ or $\max(0, c - x)$:

```text
Score = 1.0 / (1.0 + exp(-(
    -3.8421
    + 5.1204 * max(0, jaro_winkler - 0.7200)
    - 2.8910 * max(0, 0.7200 - jaro_winkler)
    + 1.8432 * max(0, token_sort_ratio - 0.6500)
    + 0.9420 * exact_phone_match
)))
```
*Zero hidden layers. Zero uninspectable matrix weights.*

---

## 📦 Pre-Trained Models & Feature Routing

Whitebox-Dedup automatically detects available columns and routes to the optimal pre-trained model:

| Model ID | Input Columns Required | Target Use Case | F1-Score |
| :--- | :--- | :--- | :---: |
| `text_only_v1` | Names / Entity Strings | General entity resolution, name variations | 93.4% |
| `text_phone_v1` | Name + Phone Number | Beneficiary registries, farmer KYC | 97.1% |
| `text_phone_geo_v1` | Name + Phone + GPS Lat/Lon | Smallholder agriculture, land parcel linkage | 98.6% |

---

## 💻 TypeScript / JavaScript Usage

Whitebox-Dedup is also available as a standalone npm package for Node.js and browser Web Workers:

```typescript
import { DeduplicationPipeline } from "@futux/whitebox-dedup";

const pipeline = new DeduplicationPipeline({
  model: "text_phone_v1",
  confidence: 0.95
});

const results = pipeline.run([
  { id: "1", name: "Ibrahim Musa", phone: "+254712345678" },
  { id: "2", name: "Ibraheem Mussa", phone: "+254712345678" },
  { id: "3", name: "Sarah Cherono", phone: "+254787654321" }
]);

console.log(results.matches);
```

---

## 🏛️ Citation & Academic Attribution

If you use Whitebox-Dedup in academic research, conference publications, or DPI deployments, please cite:

```bibtex
@software{futux_whitebox_dedup_2026,
  author = {FutuX Agri-consult Limited},
  title = {Whitebox-Dedup: Interpretable Entity Deduplication with Auditable MARS Equations and Conformal Prediction},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23612975},
  url = {https://doi.org/10.5281/zenodo.23612975},
  howpublished = {\url{https://github.com/futuxagriconsult/whitebox-dedup-ml}}
}
```

---

## 📄 License

Whitebox-Dedup is open-sourced under the **MIT License**.  
Maintained by **FutuX Agri-consult Limited** ([info@futuxconsult.com](mailto:info@futuxconsult.com)).
