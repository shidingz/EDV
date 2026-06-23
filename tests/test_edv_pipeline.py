"""Regression tests for the toy EDV pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edv import EDVPipeline, IsoCodeAgent, NaturalLanguageAgent, Task


class EDVPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = EDVPipeline.with_components(
            executors=[
                NaturalLanguageAgent("legacy-name-agent"),
                IsoCodeAgent("schema-first-agent"),
            ]
        )

    def test_construct_experience_writes_shared_memory(self) -> None:
        task = Task(
            task_id="train-translation-array",
            domain="translation",
            instruction="Translate the programming term 'array' from English to Russian.",
            metadata={"word": "array", "from_language": "en", "to_language": "ru"},
        )
        report = self.pipeline.construct_experience(task)

        self.assertEqual(len(report.trajectories), 2)
        self.assertEqual([trajectory.success for trajectory in report.trajectories], [False, True])
        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.decisions[0].destination, "shared")
        self.assertEqual(len(self.pipeline.memory.shared), 1)

    def test_inference_retrieves_verified_memory(self) -> None:
        train = Task(
            task_id="train-translation-array",
            domain="translation",
            instruction="Translate the programming term 'array' from English to Russian.",
            metadata={"word": "array", "from_language": "en", "to_language": "ru"},
        )
        self.pipeline.construct_experience(train)

        test = Task(
            task_id="test-translation-stack",
            domain="translation",
            instruction="Translate the programming term 'stack' from English to Russian.",
            metadata={"word": "stack", "from_language": "en", "to_language": "ru"},
        )
        report = self.pipeline.infer(test)

        self.assertEqual(report.selected_agent, "schema-first-agent")
        self.assertTrue(report.retrieved_memories)
        self.assertTrue(report.trajectory.success)
        self.assertEqual(report.trajectory.final_answer, "stek")


if __name__ == "__main__":
    unittest.main()
