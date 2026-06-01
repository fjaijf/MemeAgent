from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item.strip())
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in text_parts if part)
    return str(content).strip()


class MemeAgent:
    """A minimal agent that directly calls an LLM with a prompt."""

    def __init__(self, llm: Any, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, topic: str, context: str = "") -> str:
        user_prompt = f"""
Topic: {topic}

Extra context:
{context or "None"}

Please produce:
1. A one-paragraph summary of the meme narrative
2. The current sentiment signal
3. The main upside catalysts
4. The main downside risks
5. A short final stance: bullish, neutral, or bearish
""".strip()

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))
