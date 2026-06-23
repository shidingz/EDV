"""Agents, distillation, and verification components."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Sequence

from edv.types import Experience, Step, Task, ToolCall, Trajectory, VerificationDecision, Vote


LANGUAGE_NAME = {
    "en": "English",
    "ru": "Russian",
    "fr": "French",
    "de": "German",
}

LANGUAGE_CODE = {name.lower(): code for code, name in LANGUAGE_NAME.items()}


class BaseAgent(ABC):
    """Base class for all executor/verifier agents."""

    def __init__(self, name: str, profile: str = "") -> None:
        self.name = name
        self.profile = profile or name

    def execute(
        self,
        task: Task,
        environment,
        memories: Sequence[Experience] = (),
    ) -> Trajectory:
        """Run one task and return a trajectory."""

        tool_call = self.plan_tool_call(task, memories)
        observation = environment.run_tool(tool_call)
        thought = self._build_thought(task, memories)
        final_answer = observation.result if observation.success else observation.error or "failed"
        steps = [Step(thought=thought, tool_call=tool_call, observation=observation)]
        return Trajectory(
            trajectory_id=_stable_id(self.name, task.task_id, tool_call.render()),
            agent_name=self.name,
            task_id=task.task_id,
            steps=steps,
            success=observation.success,
            final_answer=final_answer,
            score=1.0 if observation.success else 0.0,
            notes={"profile": self.profile},
        )

    @abstractmethod
    def plan_tool_call(self, task: Task, memories: Sequence[Experience]) -> ToolCall:
        """Produce the next tool call."""

    def verify_experience(
        self,
        candidate: Experience,
        task: Task,
        trajectories: Sequence[Trajectory],
    ) -> Vote:
        """Strict default-reject verifier used by toy agents.

        A real LLM-backed verifier can replace this method. The toy verifier
        accepts only memories that are specific, grounded in the trajectories,
        and linked to a concrete tool-schema failure.
        """

        text = candidate.searchable_text().lower()
        summaries = "\n".join(t.compact_summary() for t in trajectories).lower()
        has_schema_failure = "invalid language argument" in summaries
        has_successful_iso_call = '"fromlanguage": "en"' in summaries and '"tolanguage": "ru"' in summaries
        is_specific = "iso 639-1" in text and "translateword" in text
        is_relevant = task.domain in candidate.tags or task.domain in text

        accept = bool(has_schema_failure and has_successful_iso_call and is_specific and is_relevant)
        if accept:
            rationale = "Grounded schema correction observed across failed and successful trajectories."
        else:
            rationale = "Rejected by default because the candidate is not sufficiently grounded or specific."
        return Vote(agent_name=self.name, accept=accept, rationale=rationale)

    def _build_thought(self, task: Task, memories: Sequence[Experience]) -> str:
        if memories:
            titles = ", ".join(memory.title for memory in memories)
            return f"Use retrieved memories: {titles}"
        return f"Solve task in profile: {self.profile}"


class NaturalLanguageAgent(BaseAgent):
    """An intentionally imperfect agent that uses full language names by default."""

    def __init__(self, name: str = "natural-language-agent") -> None:
        super().__init__(name=name, profile="prefers human-readable tool arguments")

    def plan_tool_call(self, task: Task, memories: Sequence[Experience]) -> ToolCall:
        word = _task_word(task)
        from_code = _language_code(task.metadata.get("from_language", "en"))
        to_code = _language_code(task.metadata.get("to_language", "ru"))

        if _memory_mentions_iso_codes(memories):
            from_language = from_code
            to_language = to_code
        else:
            from_language = LANGUAGE_NAME.get(from_code, from_code)
            to_language = LANGUAGE_NAME.get(to_code, to_code)

        return ToolCall(
            name="translateWord",
            arguments={
                "fromLanguage": from_language,
                "toLanguage": to_language,
                "word": word,
            },
        )


class IsoCodeAgent(BaseAgent):
    """A schema-first agent that uses ISO 639-1 codes."""

    def __init__(self, name: str = "iso-code-agent") -> None:
        super().__init__(name=name, profile="checks tool schemas before acting")

    def plan_tool_call(self, task: Task, memories: Sequence[Experience]) -> ToolCall:
        return ToolCall(
            name="translateWord",
            arguments={
                "fromLanguage": _language_code(task.metadata.get("from_language", "en")),
                "toLanguage": _language_code(task.metadata.get("to_language", "ru")),
                "word": _task_word(task),
            },
        )


class ContrastiveDistiller:
    """Third-party distiller that compares trajectories before writing memory."""

    def __init__(self, name: str = "contrastive-distiller") -> None:
        self.name = name

    def distill(self, task: Task, trajectories: Sequence[Trajectory]) -> List[Experience]:
        successes = [trajectory for trajectory in trajectories if trajectory.success]
        failures = [trajectory for trajectory in trajectories if not trajectory.success]
        if not successes or not failures:
            return []

        summaries = "\n".join(t.compact_summary() for t in trajectories).lower()
        if task.domain == "translation" and "invalid language argument" in summaries:
            content = (
                "When calling translateWord, pass ISO 639-1 two-letter codes "
                "for fromLanguage and toLanguage, for example fromLanguage='en' "
                "and toLanguage='ru'. Do not pass full language names such as "
                "'English' or 'Russian', because the tool schema rejects them."
            )
            title = "Use ISO 639-1 Codes for Translation Tools"
            return [
                Experience(
                    experience_id=_stable_id(task.task_id, title, content),
                    title=title,
                    description=(
                        "Contrastive trajectories show that natural-language "
                        "language names fail while ISO-code arguments succeed."
                    ),
                    content=content,
                    tags=("translation", "tool-schema", "iso-639-1"),
                    source_task_id=task.task_id,
                    supporting_trajectory_ids=tuple(t.trajectory_id for t in successes),
                    rejected_trajectory_ids=tuple(t.trajectory_id for t in failures),
                    metadata={"distiller": self.name},
                )
            ]
        return []


class ConsensusVerifier:
    """Verify candidates with the execution group.

    EDV uses a strict default-reject policy:
    - unanimous approval -> shared memory
    - partial approval -> private memory for approving agents
    - no approval -> discard
    """

    def verify(
        self,
        candidates: Sequence[Experience],
        verifiers: Sequence[BaseAgent],
        task: Task,
        trajectories: Sequence[Trajectory],
    ) -> List[VerificationDecision]:
        decisions: List[VerificationDecision] = []
        for candidate in candidates:
            votes = [agent.verify_experience(candidate, task, trajectories) for agent in verifiers]
            accepts = [vote for vote in votes if vote.accept]
            if len(accepts) == len(votes) and votes:
                destination = "shared"
            elif accepts:
                destination = "private"
            else:
                destination = "discard"
            decisions.append(
                VerificationDecision(
                    experience=candidate,
                    votes=votes,
                    destination=destination,
                )
            )
        return decisions


def _task_word(task: Task) -> str:
    word = task.metadata.get("word")
    if word:
        return str(word).lower()
    if "'" in task.instruction:
        return task.instruction.split("'")[1].lower()
    return task.instruction.strip().split()[-1].strip(".").lower()


def _language_code(value: object) -> str:
    raw = str(value).strip()
    if len(raw) == 2:
        return raw.lower()
    return LANGUAGE_CODE.get(raw.lower(), raw.lower())


def _memory_mentions_iso_codes(memories: Iterable[Experience]) -> bool:
    return any("iso 639-1" in memory.searchable_text().lower() for memory in memories)


def _stable_id(*parts: object) -> str:
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:12]
