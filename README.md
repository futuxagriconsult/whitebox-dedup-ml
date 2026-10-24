# whitebox-dedup

Interpretable entity deduplication using auditable MARS equations and Inductive Conformal Prediction, with a flexible feature registry that adapts to whatever columns the data has.

---

## Quick Install

```bash
pip install whitebox-dedup
```

First result in three lines:

```python
from whitebox_dedup import DeduplicationPipeline
pipeline = DeduplicationPipeline.from_csv("farmers.csv")
print(pipeline.run())
```

CLI equivalent:

```bash
wdedup deduplicate --input farmers.csv --output results.json
wdedup show-equation --model text_only_v1
```

---

## Why It Exists

Deduplication in African programme registries (farmer benefit lists, NIN databases, voter rolls) has historically been delegated to black-box ML models that produce a score nobody can audit. When a beneficiary challenges a duplicate flag, the programme officer has no arithmetic to show them. Courts and auditors cannot verify what the system did.

FutuX Agri-consult Limited built this library after observing the same pattern across 12 agricultural programmes in Nigeria between 2021 and 2026: opaque deduplication created downstream fraud disputes that consumed more resources than the duplicates themselves. The solution is not a more powerful model. The solution is a model whose decision can be written on one page and checked by a field officer with a calculator.

The MARS equation is that page. Every hinge term, its knot, and its coefficient are stored in a plain JSON file that any regulator can read. The conformal prediction layer adds a formal coverage guarantee: when the model cannot decide, it says so explicitly and routes the pair to a human. Automated decisions are made only when the statistical evidence is strong enough to sustain audit.

---

## How It Works

### Feature Registry

Every feature the engine can compute is registered with the columns it requires. Features whose columns are absent are silently skipped. The engine adapts to whatever schema the user has.

| Feature          | Required columns              | Description                                      |
|------------------|-------------------------------|--------------------------------------------------|
| `jw_name_full`   | any NAME column               | Jaro-Winkler on full concatenated name           |
| `jw_name_first`  | any NAME column               | Jaro-Winkler on first token only                 |
| `jw_name_last`   | any NAME column               | Jaro-Winkler on last token only                  |
| `exact_id`       | any ID column                 | Exact string match after normalisation           |
| `exact_phone`    | any PHONE column              | Exact match after stripping formatting           |
| `haversine_m`    | LATITUDE + LONGITUDE          | Great-circle distance in metres                  |
| `same_admin_1`   | any ADMIN_1 column            | Same state/region/province                       |
| `same_admin_2`   | any ADMIN_2 column            | Same LGA/district/ward                           |
| `dob_match`      | any DATE_OF_BIRTH column      | Exact DOB string match                           |
| `same_gender`    | any GENDER column             | Same gender (normalised to first letter)         |
| `same_enumerator`| any ENUMERATOR column         | Same field agent collected both records          |

### Three-Tier Decision Routing

Every candidate pair is assigned to exactly one tier:

| Tier              | Condition                                           | Action                          |
|-------------------|-----------------------------------------------------|---------------------------------|
| 1: AUTO_MERGE     | High-confidence duplicate (or hard ID match)        | System merges automatically     |
| 2: HUMAN_REVIEW   | Ambiguous: conformal set is {0, 1}                  | Diverted to human administrator |
| 3: AUTO_DISTINCT  | High-confidence distinct (or hard ID mismatch)      | No action taken                 |

Hard signals short-circuit the MARS model entirely:
- Exact unique ID match: Tier 1 immediately
- Exact unique ID mismatch (both non-null): Tier 3 immediately
- Exact phone match + JW >= 0.70: Tier 1
- DOB exact match + JW >= 0.85: Tier 1

### The MARS Equation

```
logit[P(duplicate)] = -102.231
  + 562.965 * max(0, jw_name_full - 0.559)
  - 679.490 * max(0, jw_name_full - 0.806)
  +  96.830 * max(0, 0.898 - jw_name_full)
  + 122.623 * max(0, jw_name_full - 0.898)

P(duplicate) = 1 / (1 + exp(-logit))
```

This is the complete model. No hidden layers. No embeddings. A spreadsheet is sufficient to verify any decision.

---

## Pre-trained Models

| Model name          | Required features                     | Use case                            |
|---------------------|---------------------------------------|-------------------------------------|
| `text_only_v1`      | `jw_name_full`                        | Name-only registries (NIN, voter)   |
| `text_phone_v1`     | `jw_name_full`, `exact_phone`         | Name + phone (credit applicants)    |
| `text_phone_geo_v1` | `jw_name_full`, `exact_phone`, GPS    | Agricultural rosters with GPS       |

`DeduplicationPipeline` selects the best available model automatically.

---

## Flexible Column Guide

### NIN Registry (name + NIN + DOB + gender, no GPS)

```python
from whitebox_dedup.schema import EntitySchema, EntityField, FieldRole
schema = EntitySchema(fields={
    "nin":           EntityField("nin",           FieldRole.ID, unique_key=True),
    "full_name":     EntityField("full_name",     FieldRole.NAME),
    "date_of_birth": EntityField("date_of_birth", FieldRole.DATE_OF_BIRTH),
    "gender":        EntityField("gender",        FieldRole.GENDER),
    "state_origin":  EntityField("state_origin",  FieldRole.ADMIN_1),
})
from whitebox_dedup import DeduplicationPipeline
pipeline = DeduplicationPipeline.from_schema(schema, "nin_records.csv")
results = pipeline.run()
```

Active features: `exact_id`, `jw_name_full`, `jw_name_first`, `jw_name_last`, `dob_match`, `same_gender`, `same_admin_1`

### Agricultural Roster (name + phone + GPS)

```python
schema = EntitySchema.infer("farmers.csv")   # auto-detect all columns
pipeline = DeduplicationPipeline.from_csv("farmers.csv", schema=schema)
results = pipeline.run()
print(results.summary())
```

### Patient Registry (hospital_id + name + DOB)

```python
schema = EntitySchema(fields={
    "hospital_id": EntityField("hospital_id", FieldRole.ID, unique_key=True),
    "full_name":   EntityField("full_name",   FieldRole.NAME),
    "dob":         EntityField("dob",         FieldRole.DATE_OF_BIRTH),
    "gender":      EntityField("gender",      FieldRole.GENDER),
})
pipeline = DeduplicationPipeline.from_schema(schema, "patients.csv")
results = pipeline.run()
```

---

## Python API Reference

### `DeduplicationPipeline`

```python
DeduplicationPipeline.from_csv(path, schema=None)
    # Load CSV, infer or apply schema, select best model.

DeduplicationPipeline.from_schema(schema, source)
    # Explicit schema with CSV path or DataFrame.

pipeline.run() -> DeduplicationResult
    # Run blocking + feature extraction + routing.

pipeline.score_pair(row_a: dict, row_b: dict) -> RoutingResult
    # Score one pair without full blocking.
```

### `EntitySchema`

```python
EntitySchema.infer(source)           # Auto-detect from CSV or DataFrame
EntitySchema.from_yaml(path)         # Load from YAML config file
EntitySchema.from_dict(data)         # Load from Python dict
schema.active_features()             # Set of computable feature names
schema.fields_by_role(FieldRole.X)   # Column names for a given role
```

### `MARSScorer`

```python
MARSScorer.load_pretrained("text_only_v1")
MARSScorer.load(path)
scorer.score(features: dict) -> float          # Probability in [0, 1]
scorer.score_with_trace(features) -> dict      # Full arithmetic trace
scorer.equation_text() -> str                  # Human-readable equation
scorer.save(path)                              # Export to JSON
```

### `DeduplicationResult`

```python
result.summary() -> dict
result.tier1_auto_merge    # list of pair dicts
result.tier2_human_review  # list of pair dicts
result.tier3_auto_distinct # list of pair dicts
```

---

## CLI Reference

```bash
# Deduplicate a CSV file
wdedup deduplicate --input farmers.csv [--schema schema.yaml] [--output results.json]

# Score a single pair interactively
wdedup score-pair [--schema schema.yaml]

# Print the MARS equation
wdedup show-equation [--model text_only_v1]
```

---

## Training Your Own Model

Training requires the optional `train` extras:

```bash
pip install "whitebox-dedup[train]"
```

Prepare a labelled pairs CSV with a binary `is_duplicate` column and feature columns. Then run `trainer.py`:

```bash
python trainer.py \
  --labelled-pairs my_labelled_pairs.csv \
  --label-col is_duplicate \
  --feature-cols jw_name_full \
  --output models/my_custom_model.json
```

The trainer fits a MARS model (scikit-learn pipeline), extracts the piecewise linear terms, and serialises them to the JSON format that `MARSScorer` loads. The output file is standalone: no scikit-learn dependency at inference time.

See `models/README.md` for a full explanation of the coefficient format.

---

## JavaScript Package

A companion JavaScript/TypeScript package for browser and Node.js environments is planned. It will load the same JSON model files and run identical arithmetic, enabling client-side auditing without a Python runtime.

---

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` before opening a pull request, especially the Data Privacy Rule.

---

## Data Privacy Rule

Real beneficiary records are never committed to this repository. Only synthetic fixtures (generated by `tests/fixtures/generate_fixtures.py`) and model weight JSON files are committed. See `CONTRIBUTING.md` for the full policy.

---

## License

MIT. Copyright FutuX Agri-consult Limited.

---

## Citation

If this library contributes to published research, please cite:

```bibtex
@software{whitebox_dedup_2024,
  author    = {{FutuX Agri-consult Limited}},
  title     = {whitebox-dedup: Interpretable entity deduplication with MARS equations},
  year      = {2024},
  url       = {https://github.com/futux/whitebox-dedup-ml},
  version   = {0.1.0}
}
```
