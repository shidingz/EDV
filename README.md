# EDV: Execute-Distill-Verify for Agentic Experience Learning

This repository is a compact, runnable reference implementation of the EDV
pipeline described in the paper:

> Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for
> Agentic Experience Learning

The code focuses on the algorithmic structure rather than benchmark
reproduction. It includes a deterministic toy environment so that the whole
pipeline can be executed without API keys, GPUs, or external services.

## What is implemented

| Paper concept | Code location |
| --- | --- |
| Heterogeneous parallel execution | `src/edv/agents.py`, `src/edv/pipeline.py` |
| Third-party contrastive distillation | `ContrastiveDistiller` in `src/edv/agents.py` |
| Consensus-based verification | `ConsensusVerifier` in `src/edv/agents.py` |
| Shared/private memory banks | `src/edv/memory.py` |
| Ability matrix model selection | `src/edv/ability.py` |
| Inference-time retrieval and solving | `EDVPipeline.infer` in `src/edv/pipeline.py` |
| Runnable demonstration | `examples/run_toy_edv.py` |

## Quickstart

```bash
python -m pip install -e .
python examples/run_toy_edv.py
python -m unittest discover -s tests
```

You can also run the demo without installation:

```bash
PYTHONPATH=src python examples/run_toy_edv.py
```

Expected demo behavior:

1. Two heterogeneous executors solve the same tool-use task.
2. One executor fails by passing natural-language language names.
3. One executor succeeds by using ISO 639-1 language codes.
4. The distiller extracts a transferable memory item.
5. The verifier accepts it by unanimous consensus.
6. The memory is written to the shared memory bank.
7. A later task retrieves that memory during inference.

## Repository layout

```text
.
├── data/
│   └── toy_tasks.json
├── docs/
│   └── algorithm.md
├── examples/
│   └── run_toy_edv.py
├── src/
│   └── edv/
│       ├── __init__.py
│       ├── ability.py
│       ├── agents.py
│       ├── envs.py
│       ├── llm.py
│       ├── memory.py
│       ├── pipeline.py
│       └── types.py
├── tests/
│   └── test_edv_pipeline.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## Minimal example

```python
from edv import EDVPipeline, IsoCodeAgent, NaturalLanguageAgent, Task

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
pipeline.construct_experience(train_task)

test_task = Task(
    task_id="test-translation-stack",
    domain="translation",
    instruction="Translate the programming term 'stack' from English to Russian.",
    metadata={"word": "stack", "from_language": "en", "to_language": "ru"},
)
report = pipeline.infer(test_task)
print(report.trajectory.final_answer)
```

## Extending to real LLM agents

The toy agents expose the same high-level methods a real agent needs:

- `execute(task, environment, memories)`
- `plan_tool_call(task, memories)`
- `verify_experience(candidate, task, trajectories)`

For real experiments, replace `NaturalLanguageAgent` and `IsoCodeAgent` with
LLM-backed executors. `src/edv/llm.py` contains a prompt-only adapter skeleton
that can be wired to OpenAI, vLLM, or any local model service.

## Notes

This repository intentionally does not reproduce tau2-bench, Mind2Web, or MMTB
benchmark numbers. Its purpose is to provide a clean, inspectable codebase that
captures the EDV control flow and can run a small case end to end.
