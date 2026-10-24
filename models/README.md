# Pre-trained Models

This directory contains the serialised MARS model weight files and ICP calibration files
for `whitebox-dedup`. All model files are plain JSON: auditors and regulators can read
and verify the coefficients in any text editor or spreadsheet.

---

## What is in Each File

### `text_only_v1.json`

The primary model for text-only deduplication (no GPS, no phone). It uses a single input:
`jw_name_full`, the Jaro-Winkler full-name similarity between two records.

The equation is:

```
logit[P(duplicate)] = -102.231
  + 562.965 * max(0, jw_name_full - 0.559)
  - 679.490 * max(0, jw_name_full - 0.806)
  +  96.830 * max(0, 0.898 - jw_name_full)
  + 122.623 * max(0, jw_name_full - 0.898)

P(duplicate) = 1 / (1 + exp(-logit))
```

Any analyst can reproduce this calculation in Excel or Google Sheets without any software.

### `calibration/text_only_v1_icp.json`

Inductive Conformal Prediction thresholds for `text_only_v1`. The calibration partition
contained 720 held-out pairs. The thresholds encode the 95th percentile nonconformity
scores for duplicate and distinct classes separately.

These thresholds determine when the circuit breaker fires (prediction set = {0, 1}).

---

## Training Data

The training data for `text_only_v1` consists of 3,598 labelled candidate pairs from
12 agricultural programme beneficiary registries collected in Nigeria between 2021 and 2026
by FutuX Agri-consult Limited. The raw records are confidential and are not published.
Only the fitted model coefficients are distributed.

The naming pool reflects:
- Northern Nigerian patronymic naming conventions
- Hausa, Fulani, and Kanuri name structures
- Shannon entropy H = 10.355 bits (highly repetitive name pool)
- Elevated same-name collision rates compared to European datasets

---

## How to Interpret the MARS Coefficients

MARS (Multivariate Adaptive Regression Splines) builds a piecewise linear model using
hinge functions of the form `max(0, x - knot)` or `max(0, knot - x)`.

Each hinge activates only when `x` crosses the knot value. The coefficient controls
the slope of that segment.

For `text_only_v1`:

| Knot  | Coefficient | Direction | Interpretation                                           |
|-------|-------------|-----------|----------------------------------------------------------|
| 0.559 | +562.965    | Forward   | Above JW=0.559, probability rises steeply               |
| 0.806 | -679.490    | Forward   | Above JW=0.806, the slope reverses (transition plateau) |
| 0.898 | +96.830     | Reflected | Below JW=0.898, small upward push                       |
| 0.898 | +122.623    | Forward   | Above JW=0.898, probability jumps to near-certainty     |

The intercept of -102.231 ensures that at JW=0 (no name similarity), the duplicate
probability is effectively 0.

---

## How to Train Your Own Model

Training requires the optional `train` extras:

```bash
yarn add -D scikit-learn numpy
```

Then use `trainer.py` at the package root:

```bash
python trainer.py \
  --labelled-pairs my_labelled_pairs.csv \
  --label-col is_duplicate \
  --feature-cols jw_name_full \
  --output models/my_custom_model.json
```

The trainer uses the `earth` algorithm (MARS) from scikit-learn's pipeline and exports
the fitted coefficients to the JSON format understood by `MARSScorer`.

---

## Which Model to Use

| Use case                                    | Recommended model    |
|---------------------------------------------|----------------------|
| Name-only registry (NIN, voter roll)        | `text_only_v1`       |
| Name + phone (credit applicants)            | `text_phone_v1`      |
| Name + phone + GPS (agricultural roster)    | `text_phone_geo_v1`  |
| Custom training data                        | Train your own       |

`DeduplicationPipeline` automatically selects the best model for the active features.
If a more capable model is unavailable, it falls back to `text_only_v1`.

---

## Licence

Model weights are released under the MIT licence, the same as the package itself.
Training data is proprietary to FutuX Agri-consult Limited and is not redistributed.
