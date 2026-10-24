"""
CLI for whitebox-dedup.

Commands:
  wdedup deduplicate --input FILE [--schema YAML] [--output JSON]
  wdedup score-pair  [--schema YAML]
  wdedup show-equation [--model MODEL_NAME]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def _print_err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def _deduplicate_cmd(args) -> int:
    """Run full deduplication on a CSV file."""
    from .pipeline import DeduplicationPipeline
    from .schema import EntitySchema

    input_path = Path(args.input)
    if not input_path.exists():
        _print_err(f"Input file not found: {input_path}")
        return 1

    schema = None
    if args.schema:
        schema_path = Path(args.schema)
        if not schema_path.exists():
            _print_err(f"Schema file not found: {schema_path}")
            return 1
        schema = EntitySchema.from_yaml(schema_path)

    print(f"Loading data from: {input_path}")
    pipeline = DeduplicationPipeline.from_csv(input_path, schema=schema)
    print(f"Schema active features: {sorted(pipeline.schema.active_features())}")
    print(f"Model: {pipeline.scorer.model_name}")
    print("Running deduplication...")

    result = pipeline.run()
    summary = result.summary()

    print("\nResults:")
    print(f"  Records processed:       {summary['total_records']}")
    print(f"  Candidate pairs:         {summary['candidate_pairs_evaluated']}")
    print(f"  Tier 1 (auto-merge):     {summary['tier1_auto_merge']}")
    print(f"  Tier 2 (human review):   {summary['tier2_human_review']}")
    print(f"  Tier 3 (auto-distinct):  {summary['tier3_auto_distinct']}")

    if args.output:
        output_path = Path(args.output)
        output_data = {
            "summary": summary,
            "tier1_auto_merge": result.tier1_auto_merge,
            "tier2_human_review": result.tier2_human_review,
            "tier3_auto_distinct": result.tier3_auto_distinct,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nFull results written to: {output_path}")

    return 0


def _score_pair_cmd(args) -> int:
    """Interactively score a single pair of records."""
    from .pipeline import DeduplicationPipeline
    from .schema import EntitySchema

    schema = None
    if args.schema:
        schema_path = Path(args.schema)
        if not schema_path.exists():
            _print_err(f"Schema file not found: {schema_path}")
            return 1
        schema = EntitySchema.from_yaml(schema_path)
    else:
        print("No schema provided. Using default agricultural schema.")
        from .schema import EntityField, FieldRole
        schema = EntitySchema(fields={
            "full_name": EntityField("full_name", FieldRole.NAME),
            "phone": EntityField("phone", FieldRole.PHONE),
            "state": EntityField("state", FieldRole.ADMIN_1),
        })

    print("Enter record A as JSON (e.g. {\"full_name\": \"Aminu Bello\", \"phone\": \"08012345678\"}):")
    try:
        line_a = input("> ").strip()
        row_a = json.loads(line_a)
    except (json.JSONDecodeError, EOFError) as exc:
        _print_err(f"Invalid JSON for record A: {exc}")
        return 1

    print("Enter record B as JSON:")
    try:
        line_b = input("> ").strip()
        row_b = json.loads(line_b)
    except (json.JSONDecodeError, EOFError) as exc:
        _print_err(f"Invalid JSON for record B: {exc}")
        return 1

    from .mars_scorer import MARSScorer
    from .router import DecisionRouter

    scorer = DeduplicationPipeline._load_best_model(schema)
    try:
        from .conformal import ICPPredictor
        icp = ICPPredictor.load_pretrained(scorer.model_name)
    except FileNotFoundError:
        icp = None

    router = DecisionRouter(scorer, icp)
    from .features import extract_all
    feats = extract_all(row_a, row_b, schema)
    routing = router.route(feats, schema)

    print(f"\nTier:        {routing.tier.value}")
    print(f"Probability: {routing.probability:.4f}")
    print(f"Reason:      {routing.reason}")
    print(f"Features:    {json.dumps({k: round(v, 4) if v is not None else None for k, v in feats.items()}, indent=2)}")

    if routing.mars_trace:
        print("\nMARS trace:")
        for contrib in routing.mars_trace["hinge_contributions"]:
            print(f"  {contrib['feature']} hinge at {contrib['knot']}: {contrib['hinge_output']:.4f}")

    return 0


def _show_equation_cmd(args) -> int:
    """Print the MARS equation for a given model."""
    from .mars_scorer import MARSScorer

    model_name = args.model or "text_only_v1"
    try:
        scorer = MARSScorer.load_pretrained(model_name)
    except FileNotFoundError as exc:
        _print_err(str(exc))
        return 1

    print(f"Model: {scorer.model_name}")
    if scorer.description:
        print(f"Description: {scorer.description}")
    print()
    print(scorer.equation_text())
    return 0


def main() -> None:
    """Entry point for the wdedup CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="wdedup",
        description="whitebox-dedup: interpretable entity deduplication",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deduplicate command
    dedup_parser = subparsers.add_parser(
        "deduplicate",
        help="Deduplicate a CSV file",
    )
    dedup_parser.add_argument("--input", required=True, help="Path to input CSV file")
    dedup_parser.add_argument("--schema", default=None, help="Path to schema YAML file (optional)")
    dedup_parser.add_argument("--output", default=None, help="Path to output JSON file (optional)")

    # score-pair command
    score_parser = subparsers.add_parser(
        "score-pair",
        help="Interactively score a single pair of records",
    )
    score_parser.add_argument("--schema", default=None, help="Path to schema YAML file (optional)")

    # show-equation command
    eq_parser = subparsers.add_parser(
        "show-equation",
        help="Print the MARS equation for a model",
    )
    eq_parser.add_argument("--model", default="text_only_v1", help="Model name (default: text_only_v1)")

    args = parser.parse_args()

    if args.command == "deduplicate":
        sys.exit(_deduplicate_cmd(args))
    elif args.command == "score-pair":
        sys.exit(_score_pair_cmd(args))
    elif args.command == "show-equation":
        sys.exit(_show_equation_cmd(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
