"""Evaluation slice and scoring utilities."""

from kaggle_nvida.evaluation.scoring import score_prediction_file
from kaggle_nvida.evaluation.slices import build_eval_slices

__all__ = ["build_eval_slices", "score_prediction_file"]

