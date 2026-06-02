from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
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

    def _image_path_to_data_url(self, image_path: str) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "image/png"

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def run(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        image_paths = image_paths or []
        image_urls = image_urls or []
        image_count = len(image_paths) + len(image_urls)

        image_instruction = ""
        if image_count:
            image_instruction = (
                f"\nAttached images: {image_count}\n"
                "Please incorporate the visual evidence into the analysis. "
                "If the images contain memes, logos, screenshots, charts, or "
                "social posts, explain what is visible and how it affects the "
                "meme narrative and sentiment.\n"
            )

        user_prompt = f"""
Topic: {topic}

Extra context:
{context or "None"}
{image_instruction}

Please produce:
1. A one-paragraph summary of the meme narrative
2. The current sentiment signal
3. The main upside catalysts
4. The main downside risks
5. A short final stance: bullish, neutral, or bearish
""".strip()
        

        user_content: str | list[dict[str, Any]]
        if image_count:
            content_blocks: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for image_path in image_paths:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_path_to_data_url(image_path)},
                    }
                )
            for image_url in image_urls:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )
            user_content = content_blocks
        else:
            user_content = user_prompt

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))
