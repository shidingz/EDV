# EDV: Execute-Distill-Verify Paradigm for Agentic Experience Learning

> **Paper**: Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning
>
> Authors: Shiding Zhu\*, Yudi Qi\*, Yajie Wang\*, Jiaze Li, Chao Song, Yaorui Shi, Yibo Miao, Hanqi Gao, Kai Zhang†

## 📖 Overview
EDV (Execute-Distill-Verify) is a reliable experience learning framework for LLM agents, targeting the core **Self-Confirmation Trap** in single-agent self-evolution.

By decoupling execution, distillation, and validation roles through heterogeneous multi-agent collaboration, EDV transforms experience learning from an isolated self-reflection loop into a collaborative experience construction and filtering pipeline. It suppresses erroneous and noisy experience before memory insertion, enabling more robust continual self-improvement for agents in open-world environments.

## ❗ The Self-Confirmation Trap
Most existing experience learning methods rely on single-agent closed loops: the same agent executes tasks, interprets outcomes, distills lessons, and decides what to write into memory. This design inherently causes the **Self-Confirmation Trap**:
- Wrong-but-self-consistent trajectories are mistakenly treated as valid successful experience
- Errors are progressively amplified through repeated memory retrieval and reuse
- The issue is especially severe in long-horizon tasks with no explicit ground-truth feedback

EDV breaks this closed loop through role decoupling and multi-agent consensus mechanisms.

## 🧠 EDV Framework
EDV consists of two core stages: **Experience Construction (offline)** and **Inference-Time Usage (online)**.

### Stage 1: Experience Construction
1. **Execute**
   - Multiple heterogeneous agents explore the same task space in parallel
   - Generate diverse candidate trajectories to expand solution-space coverage
   - Mitigate the exploration bias of any single agent

2. **Distill**
   - A designated third-party distillation agent performs cross-trajectory comparative analysis
   - Extracts reusable candidate experience rules instead of restating a single best trajectory
   - Eliminates executor-centric self-summarization bias

3. **Verify**
   - The execution group jointly validates candidate experiences via a consensus-based mechanism
   - Unanimously approved experience → shared memory bank (general reusable knowledge)
   - Partially approved experience → corresponding private memory bank (agent-specific knowledge)
   - Unqualified experience is directly discarded

### Stage 2: Inference-Time Usage
- **Ability Matrix**: Routes new tasks to the most suitable agent based on historical performance statistics
- **Hierarchical Retrieval**: Queries the shared memory bank first, then falls back to the agent-specific private memory bank
- Retrieved memories are injected into the task context to guide subsequent reasoning

## ✨ Key Features
- Heterogeneous parallel trajectory generation for diverse solution space exploration
- Third-party contrastive distillation to eliminate executor cognitive bias
- Consensus-based validation with default-reject policy for high memory quality
- Shared + private hierarchical memory architecture for both generalization and personalization
- Lightweight ability matrix routing with no extra inference overhead
- Consistently outperforms strong baselines across multiple long-horizon benchmarks

## 🔧 Code Overview
This repository currently provides a lightweight reference implementation of the EDV pipeline. The goal is to make the core idea easy to inspect and run, rather than to reproduce the full benchmark results.

The main workflow is implemented in `src/edv/pipeline.py`. Given a task, `EDVPipeline.construct_experience()` first asks multiple executor agents to solve the same task, then sends their trajectories to a distiller, and finally uses a verifier to decide whether the distilled experience should enter shared memory, private memory, or be discarded. At inference time, `EDVPipeline.infer()` selects a suitable agent with the ability matrix, retrieves relevant memories, and runs the selected agent with those memories.

The code is organized as follows:
- `src/edv/agents.py`: toy executor agents, the contrastive distiller, and the consensus verifier.
- `src/edv/memory.py`: shared and private memory banks with simple retrieval.
- `src/edv/ability.py`: the ability matrix used to route tasks to agents.
- `src/edv/envs.py`: a small translation-tool environment for demonstration.
- `examples/run_toy_edv.py`: a minimal runnable case showing Execute → Distill → Verify → Memory Retrieval.
- `docs/algorithm.md`: pseudocode-style explanation of the full EDV flow.

To run the toy case:
```bash
python examples/run_toy_edv.py
```

This example intentionally includes one agent that fails by using natural-language tool arguments and another agent that succeeds by following the tool schema. EDV compares the trajectories, distills the useful rule, verifies it, stores it as memory, and retrieves it for a later task.

## 📊 Experimental Results
We evaluate EDV on three challenging long-horizon agent benchmarks:

### 1. τ²-bench (Real-world Issue Resolution)
EDV achieves **86.6% average Pass@1**, outperforming:
- No Memory baseline: +7.0 ~ +10.2 points
- ReasoningBank baseline: +4.7 ~ +7.6 points
- Router ensemble baseline: +3.1 points

### 2. Mind2Web (Web Interaction)
EDV maintains strong generalization performance across cross-task, cross-website, and cross-domain settings, with consistent improvements over single-agent memory methods and ensemble baselines.

### 3. MMTB (Multi-tool Task Execution)
EDV achieves an overall score of **58.10**, surpassing the Router baseline (55.96) and all single-model baselines.

> Human audit results confirm that EDV significantly improves memory quality in terms of correctness, actionability and specificity, while reducing hallucination rate and potential harm from memory reuse.

## 🚀 Code Release
This repository contains a compact runnable implementation of the EDV algorithmic flow. Full benchmark reproduction scripts and dataset-specific adapters will be released separately after cleanup.


## 📝 Citation
If you find our work useful, please cite our paper:
```bibtex
@article{zhu2025escaping,
  title={Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning},
  author={Zhu, Shiding and Qi, Yudi and Wang, Yajie and Li, Jiaze and Song, Chao and Shi, Yaorui and Miao, Yibo and Gao, Hanqi and Zhang, Kai},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License
This project will be released under the MIT License.

## 🙏 Acknowledgments
We thank the authors of τ²-bench, Mind2Web, and MMTB for their open benchmarks, and the open-source community for LLM inference and tool-use related works.
