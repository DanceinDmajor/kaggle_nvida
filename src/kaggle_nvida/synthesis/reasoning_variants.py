"""Utilities that turn a solved example into selection and repair variants."""

from __future__ import annotations

import copy

from kaggle_nvida.schemas import Message, Supervision, TrainingExample


def build_candidate_selection_variant(example: TrainingExample, wrong_answer: str) -> TrainingExample:
    """Create a candidate-selection example from a solved base example."""
    clone = copy.deepcopy(example)
    clone.variant_id = "select"
    clone.reasoning_style = "candidate_selection"
    clone.example_id = f"{example.problem_id}_select"
    clone.stage_target = ["stage2", "stage3"]
    clone.messages = [
        clone.messages[0],
        Message(
            role="user",
            content=(
                f"{example.messages[1].content}\n\n"
                f"Candidate A: Final answer: {wrong_answer}\n"
                f"Candidate B: {example.messages[2].content}\n\n"
                "Choose the better candidate and explain briefly."
            ),
        ),
        Message(
            role="assistant",
            content=(
                "Candidate B is better because it matches the correct computation.\n"
                f"Final answer: {example.target.final_answer}"
            ),
        ),
    ]
    clone.supervision = Supervision(
        loss_mask_style="assistant_only",
        has_reasoning=True,
        has_final_answer_tag=True,
        final_answer_span=f"Final answer: {example.target.final_answer}",
    )
    clone.provenance.transform_chain.append("build_candidate_selection_variant_v1")
    return clone


def build_self_correction_variant(example: TrainingExample, wrong_answer: str) -> TrainingExample:
    """Create a self-correction example from a solved base example."""
    clone = copy.deepcopy(example)
    clone.variant_id = "repair"
    clone.reasoning_style = "self_correction"
    clone.example_id = f"{example.problem_id}_repair"
    clone.stage_target = ["stage2", "stage3"]
    clone.messages = [
        clone.messages[0],
        Message(
            role="user",
            content=(
                f"{example.messages[1].content}\n\n"
                f"A previous attempt said: Final answer: {wrong_answer}\n"
                "Find the mistake and provide the corrected answer."
            ),
        ),
        Message(
            role="assistant",
            content=(
                "The previous attempt used the wrong computation. "
                f"The corrected result is {example.target.final_answer}.\n"
                f"Final answer: {example.target.final_answer}"
            ),
        ),
    ]
    clone.supervision = Supervision(
        loss_mask_style="assistant_only",
        has_reasoning=True,
        has_final_answer_tag=True,
        final_answer_span=f"Final answer: {example.target.final_answer}",
    )
    clone.provenance.transform_chain.append("build_self_correction_variant_v1")
    return clone

