"""Synthetic document-grounded reasoning data."""

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


def _build_document_bundle(rng: random.Random) -> tuple[str, str, list[str], str]:
    cities = ["Seoul", "Toronto", "Berlin", "Lisbon", "Osaka", "Nairobi"]
    projects = ["Atlas", "Helios", "Orion", "Delta", "Nimbus", "Aster"]
    city_a, city_b = rng.sample(cities, 2)
    project_a, project_b = rng.sample(projects, 2)
    year_a = rng.randint(2018, 2024)
    year_b = rng.randint(2018, 2024)
    budget_a = rng.randint(5, 25)
    budget_b = rng.randint(5, 25)

    prompt = (
        "Read the notes and answer the question.\n\n"
        f"Document 1: Project {project_a} is based in {city_a}. It started in {year_a} and has a budget of {budget_a} million dollars.\n"
        f"Document 2: Project {project_b} is based in {city_b}. It started in {year_b} and has a budget of {budget_b} million dollars.\n"
        "Document 3: The project with the larger budget should be prioritized in the report.\n\n"
        "Question: Which project should be prioritized, and what city is it based in?"
    )
    if budget_a >= budget_b:
        answer = f"Project {project_a}, {city_a}"
        explanation = [
            f"Project {project_a} has a budget of {budget_a} million dollars.",
            f"Project {project_b} has a budget of {budget_b} million dollars.",
            f"The larger budget belongs to project {project_a}, which is based in {city_a}.",
        ]
    else:
        answer = f"Project {project_b}, {city_b}"
        explanation = [
            f"Project {project_a} has a budget of {budget_a} million dollars.",
            f"Project {project_b} has a budget of {budget_b} million dollars.",
            f"The larger budget belongs to project {project_b}, which is based in {city_b}.",
        ]
    return prompt, answer, explanation, "document_comparison"


def generate_document_reasoning_dataset(
    count: int,
    seed: int = 13,
    split: str = "train",
    source_name: str = "synthetic_retrieval_v1",
    generator_model: str = "programmatic_generator",
) -> list[TrainingExample]:
    """Generate synthetic document QA with evidence-backed reasoning."""
    rng = random.Random(seed)
    examples: list[TrainingExample] = []
    for index in range(count):
        question, answer, explanation, subtype = _build_document_bundle(rng)
        problem_id = f"retrieval_{index:06d}_{uuid.uuid4().hex[:8]}"
        difficulty = Difficulty(source_level="medium", profile_bucket="unprofiled", estimated_steps=3)
        provenance = Provenance(
            source_name=source_name,
            source_sample_id=problem_id,
            generator_model=generator_model,
            generator_prompt_id="document_reasoning_v1",
            transform_chain=["generate_document_reasoning_v1"],
        )
        quality = Quality(verified=True, judge_score=1.0, quality_band="high")

        for variant_id, reasoning_style in (
            ("full", "full_reasoning"),
            ("brief", "brief_reasoning"),
            ("ansonly", "answer_only"),
        ):
            if reasoning_style == "full_reasoning":
                assistant = "Reasoning:\n- " + "\n- ".join(explanation) + f"\nFinal answer: {answer}"
                supervision = Supervision(final_answer_span=f"Final answer: {answer}")
            elif reasoning_style == "brief_reasoning":
                assistant = f"Reasoning: {explanation[-1]}\nFinal answer: {answer}"
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
                    task_family="retrieval",
                    task_subtype=subtype,
                    split=split,
                    stage_target=["stage1", "stage2"] if variant_id == "full" else ["stage1", "stage3"],
                    format_type="chat",
                    reasoning_style=reasoning_style,
                    messages=[
                        Message(role="system", content="You are a careful reasoning assistant."),
                        Message(role="user", content=question),
                        Message(role="assistant", content=assistant),
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
            )
    return examples

