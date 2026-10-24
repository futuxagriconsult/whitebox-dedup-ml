"""
Trainer module for whitebox-dedup.

Fits auditable MARS hinge functions on labelled pair datasets and produces
plain JSON weight files and ICP calibration quantiles.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .mars_scorer import HingeFunction, MARSScorer
from .conformal import ICPCalibration


class MARSTrainer:
    """
    Fits piecewise linear hinge functions on pair features to produce an auditable
    JSON MARS model file.
    """

    def __init__(
        self,
        feature_cols: list[str],
        knots_per_feature: int = 3,
        model_name: str = "custom_mars_v1",
        description: str = "",
    ):
        self.feature_cols = feature_cols
        self.knots_per_feature = knots_per_feature
        self.model_name = model_name
        self.description = description or f"Trained MARS model on {', '.join(feature_cols)}"

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = "is_duplicate",
        alpha: float = 0.05,
    ) -> tuple[MARSScorer, ICPCalibration]:
        """
        Fit hinge terms on candidate features using logistic regression.
        Returns a fitted MARSScorer and an ICPCalibration object.
        """
        from sklearn.linear_model import LogisticRegression

        X_raw = df[self.feature_cols].fillna(0.0).values
        y = df[label_col].astype(int).values

        # Generate prospective candidate knots per feature based on quantiles
        hinge_defs: list[tuple[int, str, float, bool]] = []
        basis_columns = []

        for f_idx, f_name in enumerate(self.feature_cols):
            vals = X_raw[:, f_idx]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 2:
                # Binary feature (e.g. exact_phone, same_admin)
                knot = 0.5
                hinge_defs.append((f_idx, f_name, knot, False))
                basis_columns.append(np.maximum(0.0, vals - knot))
            else:
                # Continuous feature (e.g. jw_name_full, haversine_m)
                quantiles = np.quantile(vals, np.linspace(0.2, 0.8, self.knots_per_feature))
                for knot in np.unique(quantiles):
                    # Forward hinge
                    hinge_defs.append((f_idx, f_name, float(knot), False))
                    basis_columns.append(np.maximum(0.0, vals - knot))
                    # Reflected hinge
                    hinge_defs.append((f_idx, f_name, float(knot), True))
                    basis_columns.append(np.maximum(0.0, knot - vals))

        if len(basis_columns) == 0:
            raise ValueError("No basis functions could be constructed from feature columns.")

        X_basis = np.column_stack(basis_columns)

        # Fit regularized logistic regression to select & weight active hinges
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X_basis, y)

        intercept = float(clf.intercept_[0])
        coefs = clf.coef_[0]

        # Filter out negligible hinges (abs(coef) < 1e-3)
        hinges: list[HingeFunction] = []
        for (f_idx, f_name, knot, reflected), coef in zip(hinge_defs, coefs):
            if abs(coef) >= 1e-3:
                hinges.append(
                    HingeFunction(
                        feature=f_name,
                        coefficient=round(float(coef), 4),
                        knot=round(float(knot), 4),
                        reflected=reflected,
                    )
                )

        scorer = MARSScorer(
            intercept=round(intercept, 4),
            hinges=hinges,
            model_name=self.model_name,
            description=self.description,
            required_features=self.feature_cols,
        )

        # Compute Inductive Conformal Prediction nonconformity quantiles
        cal = self._calibrate(scorer, df, label_col, alpha=alpha)

        return scorer, cal

    def _calibrate(
        self,
        scorer: MARSScorer,
        df: pd.DataFrame,
        label_col: str,
        alpha: float = 0.05,
    ) -> ICPCalibration:
        """Compute ICP nonconformity thresholds on training/validation partition."""
        y = df[label_col].astype(int).values
        probs = np.array([scorer.score(row) for _, row in df[self.feature_cols].iterrows()])

        # Nonconformity: alpha(x, y) = 1 - P(Y=y | x)
        nc_dup = [1.0 - p for p, label in zip(probs, y) if label == 1]
        nc_dist = [p for p, label in zip(probs, y) if label == 0]

        th_dup = float(np.quantile(nc_dup, 1.0 - alpha)) if len(nc_dup) > 0 else 0.20
        th_dist = float(np.quantile(nc_dist, 1.0 - alpha)) if len(nc_dist) > 0 else 0.15

        return ICPCalibration(
            alpha=alpha,
            threshold_duplicate=round(th_dup, 4),
            threshold_distinct=round(th_dist, 4),
            model_name=self.model_name,
        )


def export_model_and_calibration(
    scorer: MARSScorer,
    calibration: ICPCalibration,
    model_dir: str | Path,
) -> tuple[Path, Path]:
    """Export model JSON and calibration JSON to specified directory."""
    out_dir = Path(model_dir)
    cal_dir = out_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{scorer.model_name}.json"
    cal_path = cal_dir / f"{scorer.model_name}_icp.json"

    model_data = {
        "model_name": scorer.model_name,
        "description": scorer.description,
        "required_features": scorer.required_features,
        "intercept": scorer.intercept,
        "hinges": [h.to_dict() for h in scorer.hinges],
    }

    cal_data = {
        "model_name": calibration.model_name,
        "alpha": calibration.alpha,
        "description": f"Inductive Conformal Prediction calibration quantiles for {calibration.model_name}.",
        "threshold_duplicate": calibration.threshold_duplicate,
        "threshold_distinct": calibration.threshold_distinct,
    }

    model_path.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
    cal_path.write_text(json.dumps(cal_data, indent=2), encoding="utf-8")

    return model_path, cal_path


def main() -> None:
    """CLI runner for model training."""
    parser = argparse.ArgumentParser(description="Train a MARS model on labelled candidate pairs")
    parser.add_argument("--pairs", required=True, help="Path to labelled candidate pairs CSV")
    parser.add_argument("--features", required=True, nargs="+", help="Feature column names to include")
    parser.add_argument("--label", default="is_duplicate", help="Ground truth column name")
    parser.add_argument("--model-name", default="custom_model_v1", help="Model name")
    parser.add_argument("--out-dir", default="models", help="Output directory")

    args = parser.parse_args()

    df = pd.read_csv(args.pairs)
    trainer = MARSTrainer(feature_cols=args.features, model_name=args.model_name)
    scorer, cal = trainer.fit(df, label_col=args.label)
    m_path, c_path = export_model_and_calibration(scorer, cal, args.out_dir)
    print(f"Model saved to: {m_path}")
    print(f"Calibration saved to: {c_path}")


if __name__ == "__main__":
    main()
