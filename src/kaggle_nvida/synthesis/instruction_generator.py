"""Synthetic instruction-following and structured output data."""

from __future__ import annotations

import json
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


def _make_record(rng: random.Random) -> tuple[str, str]:
    people = ["Mina", "Jordan", "Priya", "Leo", "Hana", "Mateo"]
    statuses = ["active", "blocked", "review", "complete"]
    priorities = ["low", "medium", "high"]
    person = rng.choice(people)
    status = rng.choice(statuses)
    priority = rng.choice(priorities)
    due_day = rng.randint(1, 28)
    prompt = (
        f"Create a JSON object for this task.\n"
        f"Owner: {person}\n"
        f"Status: {status}\n"
        f"Priority: {priority}\n"
        f"Due day: {due_day}\n\n"
        "Return exactly these keys in this order: owner, status, priority, due_day."
    )
    answer = json.dumps(
        {
            "owner": person,
            "status": status,
            "priority": priority,
            "due_day": due_day,
        },
        ensure_ascii=True,
    )
    return prompt, answer


def generate_structured_instruction_dataset(
    count: int,
    seed: int = 23,
    split: str = "train",
    source_name: str = "synthetic_structured_v1",
    generator_model: str = "programmatic_generator",
) -> list[TrainingExample]:
    """Generate structured-output instruction data."""
    rng = random.Random(seed)
    examples: list[TrainingExample] = []
    for index in range(count):
        question, answer = _make_record(rng)
        problem_id = f"structured_{index:06d}_{uuid.uuid4().hex[:8]}"
        difficulty = Difficulty(source_level="medium", profile_bucket="unprofiled", estimated_steps=2)
        provenance = Provenance(
            source_name=source_name,
            source_sample_id=problem_id,
            generator_model=generator_model,
            generator_prompt_id="structured_instruction_v1",
            transform_chain=["generate_structured_instruction_v1"],
        )
        quality = Quality(verified=True, judge_score=1.0, quality_band="high")

        for variant_id, reasoning_style in (
            ("brief", "brief_reasoning"),
            ("ansonly", "answer_only"),
        ):
            if reasoning_style == "brief_reasoning":
                assistant = (
                    "Reasoning: Extract the fields and preserve the requested key order.\n"
                    f"Final answer: {answer}"
                )
                supervision = Supervision(final_answer_span=f"Final answer: {answer}")
            else:
                assistant = f"Final answer: {answer}"
                supervision = Supervision(
                    has_reasoning=False,
                    final_answer_span=f"Final answer: {answer}",
                )

            examples.append(
                TrainingExample(
                    example_id=f"{problem_id}_{variant_id}",
                    problem_id=problem_id,
                    variant_id=variant_id,
                    task_family="structured",
                    task_subtype="json_object_construction",
                    split=split,
                    stage_target=["stage1", "stage3"],
                    format_type="chat",
                    reasoning_style=reasoning_style,
                    messages=[
                        Message(role="system", content="You are a precise structured-output assistant."),
                        Message(role="user", content=question),
                        Message(role="assistant", content=assistant),
                    ],
                    target=Target(
                        final_answer=answer,
                        normalized_final_answer=answer,
                        answer_format="json",
                    ),
                    supervision=supervision,
                    quality=replace(quality),
                    difficulty=replace(difficulty),
                    provenance=replace(provenance),
                    safety=Safety(),
                )
            )
    return examples

