"""Shared and private memory banks."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence

from edv.types import Experience, MemoryHit, VerificationDecision


class MemoryBank:
    """Hierarchical EDV memory: shared bank plus per-agent private banks."""

    def __init__(self) -> None:
        self.shared: List[Experience] = []
        self.private: DefaultDict[str, List[Experience]] = defaultdict(list)

    def apply_decision(self, decision: VerificationDecision) -> None:
        if decision.destination == "shared":
            self.add_shared(decision.experience)
        elif decision.destination == "private":
            for agent_name in decision.accepted_agents:
                self.add_private(agent_name, decision.experience)

    def add_shared(self, experience: Experience) -> None:
        if not self._contains(self.shared, experience.experience_id):
            self.shared.append(experience)

    def add_private(self, agent_name: str, experience: Experience) -> None:
        if not self._contains(self.private[agent_name], experience.experience_id):
            self.private[agent_name].append(experience)

    def retrieve_shared(self, query: str, k: int = 3, min_score: float = 0.01) -> List[MemoryHit]:
        return self._retrieve(self.shared, query, "shared", k=k, min_score=min_score)

    def retrieve_private(
        self,
        agent_name: str,
        query: str,
        k: int = 3,
        min_score: float = 0.01,
    ) -> List[MemoryHit]:
        return self._retrieve(self.private[agent_name], query, f"private:{agent_name}", k=k, min_score=min_score)

    def save_json(self, path: Path) -> None:
        data = {
            "shared": [asdict(exp) for exp in self.shared],
            "private": {
                agent_name: [asdict(exp) for exp in experiences]
                for agent_name, experiences in self.private.items()
            },
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "MemoryBank":
        data = json.loads(path.read_text(encoding="utf-8"))
        bank = cls()
        bank.shared = [_experience_from_dict(item) for item in data.get("shared", [])]
        for agent_name, items in data.get("private", {}).items():
            bank.private[agent_name] = [_experience_from_dict(item) for item in items]
        return bank

    @staticmethod
    def _contains(experiences: Sequence[Experience], experience_id: str) -> bool:
        return any(exp.experience_id == experience_id for exp in experiences)

    @staticmethod
    def _retrieve(
        experiences: Iterable[Experience],
        query: str,
        bank: str,
        k: int,
        min_score: float,
    ) -> List[MemoryHit]:
        query_tokens = _tokens(query)
        hits: List[MemoryHit] = []
        for experience in experiences:
            memory_tokens = _tokens(experience.searchable_text())
            score = _jaccard(query_tokens, memory_tokens)
            if score >= min_score:
                hits.append(MemoryHit(experience=experience, score=score, bank=bank))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]


def _experience_from_dict(data: Dict[str, object]) -> Experience:
    return Experience(
        experience_id=str(data["experience_id"]),
        title=str(data["title"]),
        description=str(data["description"]),
        content=str(data["content"]),
        tags=tuple(data.get("tags", ())),
        source_task_id=str(data["source_task_id"]),
        supporting_trajectory_ids=tuple(data.get("supporting_trajectory_ids", ())),
        rejected_trajectory_ids=tuple(data.get("rejected_trajectory_ids", ())),
        metadata=dict(data.get("metadata", {})),
    )


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
