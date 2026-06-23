"""Ability matrix for inference-time agent selection."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


class AbilityMatrix:
    """Tracks which agents work well for each task domain."""

    def __init__(self, prior: float = 0.5) -> None:
        self.prior = prior
        self._stats: Dict[str, Dict[str, List[int]]] = {}

    def update(self, agent_name: str, domain: str, success: bool) -> None:
        domain_stats = self._stats.setdefault(domain, {})
        cell = domain_stats.setdefault(agent_name, [0, 0])
        cell[0] += int(success)
        cell[1] += 1

    def score(self, agent_name: str, domain: str) -> float:
        successes, attempts = self._cell(agent_name, domain)
        if attempts == 0:
            return self.prior
        return successes / attempts

    def attempts(self, agent_name: str, domain: str) -> int:
        return self._cell(agent_name, domain)[1]

    def best_agent(self, domain: str, agent_names: Iterable[str]) -> str:
        ranked = self.rank_agents(domain, agent_names)
        if not ranked:
            raise ValueError("No candidate agents were provided.")
        return ranked[0][0]

    def rank_agents(self, domain: str, agent_names: Iterable[str]) -> List[Tuple[str, float]]:
        scored = [(name, self.score(name, domain), self.attempts(name, domain)) for name in agent_names]
        scored.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
        return [(name, score) for name, score, _ in scored]

    def snapshot(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        output: Dict[str, Dict[str, Dict[str, float]]] = {}
        for domain, agent_stats in self._stats.items():
            output[domain] = {}
            for agent_name, (successes, attempts) in agent_stats.items():
                output[domain][agent_name] = {
                    "successes": float(successes),
                    "attempts": float(attempts),
                    "score": successes / attempts if attempts else self.prior,
                }
        return output

    def _cell(self, agent_name: str, domain: str) -> Tuple[int, int]:
        cell = self._stats.get(domain, {}).get(agent_name, [0, 0])
        return int(cell[0]), int(cell[1])
