"""
whitebox-dedup: Interpretable entity deduplication with auditable MARS equations
and Inductive Conformal Prediction.

Developed by FutuX Agri-consult Limited.

Key exports:
- DeduplicationPipeline: end-to-end deduplication runner
- EntitySchema: declares the semantic role of each column
- MARSScorer: zero-dependency MARS equation evaluator
- DeduplicationResult: structured output from pipeline.run()
"""
from .pipeline import DeduplicationPipeline, DeduplicationResult
from .schema import EntitySchema
from .mars_scorer import MARSScorer
from .trainer import MARSTrainer

__all__ = [
    "DeduplicationPipeline",
    "EntitySchema",
    "MARSScorer",
    "MARSTrainer",
    "DeduplicationResult",
]

__version__ = "0.1.0"
__author__ = "Daniel Olagunju, Temitope Olarewaju"
