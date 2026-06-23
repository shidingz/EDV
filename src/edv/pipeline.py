"""End-to-end Execute-Distill-Verify pipeline."""

from __future__ import annotations

from typing import Dict, List, Sequence

from edv.ability import AbilityMatrix
from edv.agents import (
    BaseAgent,
    ConsensusVerifier,
    ContrastiveDistiller,
    IsoCodeAgent,
    NaturalLanguageAgent,
)
from edv.envs import ToyTranslationEnvironment
from edv.memory import MemoryBank
from edv.types import ConstructionReport, Experience, InferenceReport, MemoryHit, Task


class EDVPipeline:
    """Coordinate offline experience construction and inference-time reuse."""

    def __init__(
        self,
        executors: Sequence[BaseAgent],
        distiller: ContrastiveDistiller,
        verifier: ConsensusVerifier,
        memory: MemoryBank,
        ability_matrix: AbilityMatrix,
        environment,
    ) -> None:
        if not executors:
            raise ValueError("EDV requires at least one execution agent.")
        self.executors = list(executors)
        self.distiller = distiller
        self.verifier = verifier
        self.memory = memory
        self.ability_matrix = ability_matrix
        self.environment = environment

    @classmethod
    def with_components(
        cls,
        executors: Sequence[BaseAgent] = (),
        distiller: ContrastiveDistiller = None,
        verifier: ConsensusVerifier = None,
        memory: MemoryBank = None,
        ability_matrix: AbilityMatrix = None,
        environment=None,
    ) -> "EDVPipeline":
        return cls(
            executors=executors
            or [
                NaturalLanguageAgent("legacy-name-agent"),
                IsoCodeAgent("schema-first-agent"),
            ],
            distiller=distiller or ContrastiveDistiller(),
            verifier=verifier or ConsensusVerifier(),
            memory=memory or MemoryBank(),
            ability_matrix=ability_matrix or AbilityMatrix(),
            environment=environment or ToyTranslationEnvironment(),
        )

    def construct_experience(self, task: Task) -> ConstructionReport:
        """Run EDV offline memory construction for one task."""

        trajectories = [agent.execute(task, self.environment, memories=()) for agent in self.executors]
        for trajectory in trajectories:
            self.ability_matrix.update(
                agent_name=trajectory.agent_name,
                domain=task.domain,
                success=trajectory.success,
            )

        candidates = self.distiller.distill(task, trajectories)
        decisions = self.verifier.verify(
            candidates=candidates,
            verifiers=self.executors,
            task=task,
            trajectories=trajectories,
        )
        for decision in decisions:
            self.memory.apply_decision(decision)

        return ConstructionReport(
            task=task,
            trajectories=trajectories,
            candidates=candidates,
            decisions=decisions,
        )

    def infer(self, task: Task, k: int = 3) -> InferenceReport:
        """Select an agent, retrieve memories, and solve a new task."""

        agent_names = [agent.name for agent in self.executors]
        selected_name = self.ability_matrix.best_agent(task.domain, agent_names)
        selected_agent = self._agent_by_name(selected_name)
        query = f"{task.domain} {task.instruction}"

        retrieved = self.memory.retrieve_shared(query, k=k)
        if len(retrieved) < k:
            retrieved.extend(self.memory.retrieve_private(selected_name, query, k=k - len(retrieved)))
        retrieved = _dedupe_hits(retrieved)

        trajectory = selected_agent.execute(
            task=task,
            environment=self.environment,
            memories=[hit.experience for hit in retrieved],
        )
        return InferenceReport(
            task=task,
            selected_agent=selected_name,
            retrieved_memories=retrieved,
            trajectory=trajectory,
            ability_snapshot=self.ability_matrix.snapshot(),
        )

    def _agent_by_name(self, name: str) -> BaseAgent:
        for agent in self.executors:
            if agent.name == name:
                return agent
        raise KeyError(f"Unknown agent: {name}")


def _dedupe_hits(hits: Sequence[MemoryHit]) -> List[MemoryHit]:
    seen = set()
    output: List[MemoryHit] = []
    for hit in hits:
        if hit.experience.experience_id in seen:
            continue
        seen.add(hit.experience.experience_id)
        output.append(hit)
    return output
