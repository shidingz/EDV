"""Toy environments used by examples and tests."""

from __future__ import annotations

from typing import Dict, Tuple

from edv.types import Observation, ToolCall


class ToyTranslationEnvironment:
    """Small tool-use environment for demonstrating EDV.

    The environment intentionally rejects natural-language language names. This
    creates a self-contained failure mode similar to tool-schema mistakes in
    long-horizon agent benchmarks.
    """

    def __init__(self) -> None:
        self.translations: Dict[Tuple[str, str, str], str] = {
            ("en", "ru", "array"): "massiv",
            ("en", "ru", "stack"): "stek",
            ("en", "ru", "queue"): "ochered",
        }

    def run_tool(self, call: ToolCall) -> Observation:
        if call.name != "translateWord":
            return Observation(
                success=False,
                error=f"Unknown tool: {call.name}",
                raw={"tool": call.name},
            )

        from_language = str(call.arguments.get("fromLanguage", ""))
        to_language = str(call.arguments.get("toLanguage", ""))
        word = str(call.arguments.get("word", "")).lower()

        if not self._is_iso_639_1(from_language):
            return Observation(
                success=False,
                error=(
                    "Invalid language argument for fromLanguage. "
                    "Expected ISO 639-1 code such as 'en'."
                ),
                raw={"bad_argument": "fromLanguage", "value": from_language},
            )
        if not self._is_iso_639_1(to_language):
            return Observation(
                success=False,
                error=(
                    "Invalid language argument for toLanguage. "
                    "Expected ISO 639-1 code such as 'ru'."
                ),
                raw={"bad_argument": "toLanguage", "value": to_language},
            )

        result = self.translations.get((from_language, to_language, word))
        if result is None:
            result = f"{word}-{to_language}"
        return Observation(
            success=True,
            result=result,
            raw={"from": from_language, "to": to_language, "word": word},
        )

    @staticmethod
    def _is_iso_639_1(value: str) -> bool:
        return len(value) == 2 and value.isalpha() and value.islower()
