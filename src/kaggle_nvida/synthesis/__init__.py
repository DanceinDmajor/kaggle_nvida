"""Synthetic data generation helpers."""

from kaggle_nvida.synthesis.math_generator import generate_math_dataset
from kaggle_nvida.synthesis.reasoning_variants import (
    build_candidate_selection_variant,
    build_self_correction_variant,
)

__all__ = [
    "build_candidate_selection_variant",
    "build_self_correction_variant",
    "generate_math_dataset",
]

