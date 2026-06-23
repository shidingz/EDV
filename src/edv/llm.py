"""Optional adapter skeleton for real LLM-backed EDV agents."""

from __future__ import annotations

from typing import Sequence

from edv.agents import BaseAgent
from edv.types import Experience, Task, ToolCall


class PromptOnlyLLMAgent(BaseAgent):
    """Prompt builder for integrating a real model client.

    This class intentionally does not call a provider. Use it as the boundary
    between EDV orchestration and an OpenAI, vLLM, LiteLLM, or local service
    implementation.
    """

    def __init__(self, name: str, model_name: str) -> None:
        super().__init__(name=name, profile=f"llm:{model_name}")
        self.model_name = model_name

    def build_prompt(self, task: Task, memories: Sequence[Experience]) -> str:
        memory_block = "\n".join(
            f"- {memory.title}: {memory.content}" for memory in memories
        ) or "(none)"
        return (
            "You are an EDV execution agent.\n"
            f"Model profile: {self.model_name}\n"
            f"Task domain: {task.domain}\n"
            f"Task: {task.instruction}\n"
            f"Retrieved memory:\n{memory_block}\n"
            "Return the next tool call as strict JSON with keys name and arguments."
        )

    def plan_tool_call(self, task: Task, memories: Sequence[Experience]) -> ToolCall:
        raise NotImplementedError(
            "Connect this adapter to a model client and parse the returned JSON tool call."
        )
