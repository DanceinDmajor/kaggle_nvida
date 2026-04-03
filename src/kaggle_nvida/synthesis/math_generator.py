"""Simple synthetic math data generation with multiple reasoning variants."""

from __future__ import annotations

import random
import uuid
from dataclasses import replace

from kaggle_nvida.schemas import (
    Difficulty,
    Message,
    Provenance,
    Quality,
    Safety,
    Supervision,
    Target,
    TrainingExample,
)


def _normalize_answer(value: int | float) -> str:
    if int(value) == value:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _make_problem(rng: random.Random) -> tuple[str, str, list[str], str]:
    operator = rng.choice(["+", "-", "*"])
    a = rng.randint(2, 50)
    b = rng.randint(2, 30)
    if operator == "+":
        answer = a + b
        explanation = [f"Add {a} and {b}.", f"The sum is {answer}."]
        question = f"What is {a} + {b}?"
        subtype = "arithmetic_addition"
    elif operator == "-":
        if b > a:
            a, b = b, a
        answer = a - b
        explanation = [f"Subtract {b} from {a}.", f"The difference is {answer}."]
        question = f"What is {a} - {b}?"
        subtype = "arithmetic_subtraction"
    else:
        answer = a * b
        explanation = [f"Multiply {a} by {b}.", f"The product is {answer}."]
        question = f"What is {a} * {b}?"
        subtype = "arithmetic_multiplication"
    return question, _normalize_answer(answer), explanation, subtype


def _assistant_response(reasoning_style: str, explanation: list[str], answer: str) -> tuple[str, Supervision]:
    if reasoning_style == "full_reasoning":
        content = "Reasoning:\n- " + "\n- ".join(explanation) + f"\nFinal answer: {answer}"
        supervision = Supervision(
            loss_mask_style="assistant_only",
            has_reasoning=True,
            has_final_answer_tag=True,
            final_answer_span=f"Final answer: {answer}",
        )
    elif reasoning_style == "brief_reasoning":
        content = f"Reasoning: {' '.join(explanation[:1])}\nFinal answer: {answer}"
        supervision = Supervision(
            loss_mask_style="assistant_only",
            has_reasoning=True,
            has_final_answer_tag=True,
            final_answer_span=f"Final answer: {answer}",
        )
    else:
        content = f"Final answer: {answer}"
        supervision = Supervision(
            loss_mask_style="assistant_only",
            has_reasoning=False,
            has_final_answer_tag=True,
            final_answer_span=f"Final answer: {answer}",
        )
    return content, supervision


def generate_math_dataset(
    count: int,
    seed: int = 7,
    split: str = "train",
    source_name: str = "synthetic_math_v1",
    generator_model: str = "programmatic_generator",
) -> list[TrainingExample]:
    """Generate arithmetic examples with full, brief, and answer-only variants."""
    rng = random.Random(seed)
    examples: list[TrainingExample] = []

    for index in range(count):
        question, answer, explanation, subtype = _make_problem(rng)
        problem_id = f"math_{index:06d}_{uuid.uuid4().hex[:8]}"
        difficulty = Difficulty(
            source_level="easy",
            profile_bucket="unprofiled",
            estimated_steps=max(1, len(explanation)),
        )
        provenance = Provenance(
            source_name=source_name,
            source_sample_id=problem_id,
            generator_model=generator_model,
            generator_prompt_id="math_programmatic_v1",
            transform_chain=["generate_math_problem_v1"],
        )
        quality = Quality(verified=True, judge_score=1.0, quality_band="high")

        for variant_id, reasoning_style in (
            ("full", "full_reasoning"),
            ("brief", "brief_reasoning"),
            ("ansonly", "answer_only"),
        ):
            assistant_content, supervision = _assistant_response(reasoning_style, explanation, answer)
            example = TrainingExample(
                example_id=f"{problem_id}_{variant_id}",
                problem_id=problem_id,
                variant_id=variant_id,
                task_family="math",
                task_subtype=subtype,
                split=split,
                stage_target=["stage1", "stage3"] if variant_id != "full" else ["stage1", "stage2"],
                format_type="chat",
                reasoning_style=reasoning_style,
                messages=[
                    Message(role="system", content="You are a careful reasoning assistant."),
                    Message(role="user", content=question),
                    Message(role="assistant", content=assistant_content),
                ],
                target=Target(
                    final_answer=answer,
                    normalized_final_answer=answer,
                    answer_format="short_text",
                ),
                supervision=supervision,
                quality=replace(quality),
                difficulty=replace(difficulty),
                provenance=replace(provenance),
                safety=Safety(),
            )
            examples.append(example)
    return examples

