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
**Full code and reproduction scripts are coming soon!** We are organizing the codebase and will release the complete implementation at this repository shortly.


## 📝 Citation
If you find our work useful, please cite our paper:
```bibtex
@article{zhu2026escaping,
  title={Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning},
  author={Zhu, Shiding and Qi, Yudi and Wang, Yajie and Li, Jiaze and Song, Chao and Shi, Yaorui and Miao, Yibo and Gao, Hanqi and Zhang, Kai},
  journal={arXiv preprint arXiv:2606.24428},
  year={2026}
}
```

## 📄 License
This project will be released under the MIT License.

## 🙏 Acknowledgments
We thank the authors of τ²-bench, Mind2Web, and MMTB for their open benchmarks, and the open-source community for LLM inference and tool-use related works.