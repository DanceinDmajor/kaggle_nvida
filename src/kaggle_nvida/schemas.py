"""Shared schema definitions for training examples and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Target:
    final_answer: str
    normalized_final_answer: str
    answer_format: str = "short_text"


@dataclass
class Supervision:
    loss_mask_style: str = "assistant_only"
    has_reasoning: bool = True
    has_final_answer_tag: bool = True
    final_answer_span: str = ""


@dataclass
class Quality:
    verified: bool = False
    judge_score: float = 0.0
    quality_band: str = "unknown"


@dataclass
class Difficulty:
    source_level: str = "unknown"
    profile_bucket: str = "unprofiled"
    estimated_steps: int = 0


@dataclass
class Provenance:
    source_name: str
    source_sample_id: str
    generator_model: str
    generator_prompt_id: str
    transform_chain: list[str] = field(default_factory=list)


@dataclass
class Safety:
    pii: bool = False
    policy_ok: bool = True


@dataclass
class TrainingExample:
    example_id: str
    problem_id: str
    variant_id: str
    task_family: str
    task_subtype: str
    split: str
    stage_target: list[str]
    format_type: str
    reasoning_style: str
    messages: list[Message]
    target: Target
    supervision: Supervision
    quality: Quality
    difficulty: Difficulty
    provenance: Provenance
    safety: Safety

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "problem_id": self.problem_id,
            "variant_id": self.variant_id,
            "task_family": self.task_family,
            "task_subtype": self.task_subtype,
            "split": self.split,
            "stage_target": self.stage_target,
            "format_type": self.format_type,
            "reasoning_style": self.reasoning_style,
            "messages": [asdict(message) for message in self.messages],
            "target": asdict(self.target),
            "supervision": asdict(self.supervision),
            "quality": asdict(self.quality),
            "difficulty": asdict(self.difficulty),
            "provenance": asdict(self.provenance),
            "safety": asdict(self.safety),
        }


@dataclass
class ManifestRecord:
    example_id: str
    problem_id: str
    variant_id: str
    path: str
    line_number: int
    task_family: str
    task_subtype: str
    reasoning_style: str
    token_count_prompt: int
    token_count_completion: int
    token_count_total: int
    source_name: str
    source_sample_id: str
    generator_model: str
    verifier_type: str
    verifier_pass: bool
    judge_score: float
    exact_dup_cluster: str
    semantic_dup_cluster: str
    difficulty_bucket: str
    profile_bucket: str
    mixture_weight: float
    selected_for_stage1: bool
    selected_for_stage2: bool
    selected_for_stage3: bool
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
