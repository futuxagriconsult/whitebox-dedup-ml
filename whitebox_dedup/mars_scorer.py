"""
MARSScorer: zero-dependency implementation of the explicit MARS deduplication equation.

The model is stored as a JSON file containing an intercept and a list of hinge functions.
Any auditor can verify or run the model using basic spreadsheet arithmetic.

The pre-trained west_africa_text_v1 model uses Jaro-Winkler full name similarity (JW)
as its primary signal:

  logit[P(duplicate)] = -102.231
    + 562.965 * max(0, JW - 0.559)
    - 679.490 * max(0, JW - 0.806)
    +  96.830 * max(0, 0.898 - JW)
    + 122.623 * max(0, JW - 0.898)

  P(duplicate) = 1 / (1 + exp(-logit))
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HingeFunction:
    """
    A single MARS hinge: coefficient * max(0, x - knot)
    If reflected=True: coefficient * max(0, knot - x)
    """
    feature: str          # name of the input feature
    coefficient: float
    knot: float
    reflected: bool = False

    def evaluate(self, x: float) -> float:
        if self.reflected:
            return self.coefficient * max(0.0, self.knot - x)
        return self.coefficient * max(0.0, x - self.knot)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "coefficient": self.coefficient,
            "knot": self.knot,
            "reflected": self.reflected,
        }


class MARSScorer:
    """
    Evaluates a MARS duplicate probability equation with full arithmetic traceability.

    The score is a probability in [0, 1]. The model logic is completely inspectable:
    every hinge term, its coefficient, and its contribution are available.

    Examples
    --------
    >>> scorer = MARSScorer.load_pretrained("west_africa_text_v1")
    >>> prob = scorer.score({"jw_name_full": 0.75})
    >>> print(f"Duplicate probability: {prob:.3f}")

    Show the equation in human-readable form:
    >>> print(scorer.equation_text())
    """

    def __init__(
        self,
        intercept: float,
        hinges: list[HingeFunction],
        model_name: str = "custom",
        description: str = "",
        required_features: list[str] | None = None,
    ):
        self.intercept = intercept
        self.hinges = hinges
        self.model_name = model_name
        self.description = description
        self.required_features = required_features or list({h.feature for h in hinges})

    @classmethod
    def load(cls, path: str | Path) -> "MARSScorer":
        """Load a MARS model from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        hinges = [HingeFunction(**h) for h in data["hinges"]]
        return cls(
            intercept=data["intercept"],
            hinges=hinges,
            model_name=data.get("model_name", "custom"),
            description=data.get("description", ""),
            required_features=data.get("required_features"),
        )

    @classmethod
    def load_pretrained(cls, model_name: str) -> "MARSScorer":
        """Load a named pre-trained model from the models/ directory."""
        models_dir = Path(__file__).parent.parent / "models"
        path = models_dir / f"{model_name}.json"
        if not path.exists():
            available = [p.stem for p in models_dir.glob("*.json")]
            raise FileNotFoundError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        return cls.load(path)

    def score(self, features: dict[str, Optional[float]]) -> float:
        """
        Compute duplicate probability for a feature dict.
        Missing features (None values) contribute 0 to their hinge terms.
        """
        logit = self.intercept
        for hinge in self.hinges:
            x = features.get(hinge.feature)
            if x is not None:
                logit += hinge.evaluate(x)
        if logit >= 50.0:
            return 1.0
        if logit <= -50.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-logit))

    def score_with_trace(self, features: dict[str, Optional[float]]) -> dict:
        """
        Compute probability with full arithmetic trace of each hinge contribution.
        Returns: {probability, logit, intercept, hinge_contributions: list}
        """
        logit = self.intercept
        contributions = []
        for hinge in self.hinges:
            x = features.get(hinge.feature)
            contribution = hinge.evaluate(x) if x is not None else 0.0
            logit += contribution
            contributions.append({
                "feature": hinge.feature,
                "input_value": x,
                "knot": hinge.knot,
                "reflected": hinge.reflected,
                "coefficient": hinge.coefficient,
                "hinge_output": contribution,
            })
        if logit >= 50.0:
            prob = 1.0
        elif logit <= -50.0:
            prob = 0.0
        else:
            prob = 1.0 / (1.0 + math.exp(-logit))
        return {
            "probability": round(prob, 6),
            "logit": round(logit, 6),
            "intercept": self.intercept,
            "hinge_contributions": contributions,
        }

    def equation_text(self) -> str:
        """Return the MARS equation as a human-readable string."""
        lines = [f"logit[P(duplicate)] = {self.intercept}"]
        for h in self.hinges:
            sign = "+" if h.coefficient >= 0 else "-"
            abs_coef = abs(h.coefficient)
            if h.reflected:
                term = f"max(0, {h.knot} - {h.feature})"
            else:
                term = f"max(0, {h.feature} - {h.knot})"
            lines.append(f"  {sign} {abs_coef} * {term}")
        lines.append("")
        lines.append("P(duplicate) = 1 / (1 + exp(-logit))")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        data = {
            "model_name": self.model_name,
            "description": self.description,
            "required_features": self.required_features,
            "intercept": self.intercept,
            "hinges": [h.to_dict() for h in self.hinges],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
