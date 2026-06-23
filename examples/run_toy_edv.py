"""Run a small EDV case end to end."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edv import EDVPipeline, IsoCodeAgent, NaturalLanguageAgent, Task


def main() -> None:
    pipeline = EDVPipeline.with_components(
        executors=[
            NaturalLanguageAgent("legacy-name-agent"),
            IsoCodeAgent("schema-first-agent"),
        ]
    )

    train_task = Task(
        task_id="train-translation-array",
        domain="translation",
        instruction="Translate the programming term 'array' from English to Russian.",
        metadata={"word": "array", "from_language": "en", "to_language": "ru"},
    )
    construction = pipeline.construct_experience(train_task)

    print("== EXPERIENCE CONSTRUCTION ==")
    for trajectory in construction.trajectories:
        status = "PASS" if trajectory.success else "FAIL"
        print(f"[{status}] {trajectory.agent_name}: {trajectory.final_answer}")

    print("\nCandidate memories:")
    for candidate in construction.candidates:
        print(f"- {candidate.title}")

    print("\nVerification decisions:")
    for decision in construction.decisions:
        votes = ", ".join(f"{vote.agent_name}={vote.accept}" for vote in decision.votes)
        print(f"- {decision.experience.title} -> {decision.destination} ({votes})")

    test_task = Task(
        task_id="test-translation-stack",
        domain="translation",
        instruction="Translate the programming term 'stack' from English to Russian.",
        metadata={"word": "stack", "from_language": "en", "to_language": "ru"},
    )
    inference = pipeline.infer(test_task)

    print("\n== INFERENCE ==")
    print(f"Selected agent: {inference.selected_agent}")
    if inference.retrieved_memories:
        print("Retrieved memories:")
        for hit in inference.retrieved_memories:
            print(f"- [{hit.bank} score={hit.score:.3f}] {hit.experience.title}")
    else:
        print("Retrieved memories: none")
    print(f"Final answer: {inference.trajectory.final_answer}")


if __name__ == "__main__":
    main()
