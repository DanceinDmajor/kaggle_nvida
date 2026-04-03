"""Synthetic data generation helpers."""

from kaggle_nvida.synthesis.document_generator import generate_document_reasoning_dataset
from kaggle_nvida.synthesis.instruction_generator import generate_structured_instruction_dataset
from kaggle_nvida.synthesis.math_generator import generate_math_dataset
from kaggle_nvida.synthesis.reasoning_variants import (
    build_candidate_selection_variant,
    build_self_correction_variant,
)

__all__ = [
    "build_candidate_selection_variant",
    "build_self_correction_variant",
    "generate_document_reasoning_dataset",
    "generate_structured_instruction_dataset",
    "generate_math_dataset",
]
