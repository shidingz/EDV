"""Core data structures used by the EDV pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Task:
    """A task handled by agents."""

    task_id: str
    domain: str
    instruction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation proposed by an agent."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return f"{self.name}({json.dumps(dict(self.arguments), ensure_ascii=True)})"


@dataclass(frozen=True)
class Observation:
    """Environment feedback for a tool call."""

    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        if self.success:
            return f"success: {self.result}"
        return f"error: {self.error}"


@dataclass(frozen=True)
class Step:
    """One reasoning/action/observation step."""

    thought: str
    tool_call: Optional[ToolCall] = None
    observation: Optional[Observation] = None


@dataclass
class Trajectory:
    """Execution trace produced by one agent on one task."""

    trajectory_id: str
    agent_name: str
    task_id: str
    steps: Sequence[Step]
    success: bool
    final_answer: str
    score: float = 0.0
    notes: Mapping[str, Any] = field(default_factory=dict)

    def compact_summary(self) -> str:
        parts = [
            f"trajectory={self.trajectory_id}",
            f"agent={self.agent_name}",
            f"success={self.success}",
        ]
        for step in self.steps:
            parts.append(f"thought={step.thought}")
            if step.tool_call is not None:
                parts.append(f"tool={step.tool_call.render()}")
            if step.observation is not None:
                parts.append(f"observation={step.observation.render()}")
        parts.append(f"final={self.final_answer}")
        return " | ".join(parts)


@dataclass(frozen=True)
class Experience:
    """A reusable memory item distilled from trajectories."""

    experience_id: str
    title: str
    description: str
    content: str
    tags: Tuple[str, ...]
    source_task_id: str
    supporting_trajectory_ids: Tuple[str, ...] = ()
    rejected_trajectory_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.title,
                self.description,
                self.content,
                " ".join(self.tags),
                self.source_task_id,
            ]
        )


@dataclass(frozen=True)
class Vote:
    """One verifier's decision about a candidate memory."""

    agent_name: str
    accept: bool
    rationale: str


@dataclass(frozen=True)
class VerificationDecision:
    """Routing decision after consensus verification."""

    experience: Experience
    votes: Sequence[Vote]
    destination: str

    @property
    def accepted_agents(self) -> Tuple[str, ...]:
        return tuple(vote.agent_name for vote in self.votes if vote.accept)


@dataclass
class ConstructionReport:
    """Full report for the offline EDV memory-construction stage."""

    task: Task
    trajectories: Sequence[Trajectory]
    candidates: Sequence[Experience]
    decisions: Sequence[VerificationDecision]


@dataclass(frozen=True)
class MemoryHit:
    """A retrieved memory item and its similarity score."""

    experience: Experience
    score: float
    bank: str


@dataclass
class InferenceReport:
    """Full report for inference-time model selection and retrieval."""

    task: Task
    selected_agent: str
    retrieved_memories: Sequence[MemoryHit]
    trajectory: Trajectory
    ability_snapshot: Dict[str, Dict[str, Dict[str, float]]]
