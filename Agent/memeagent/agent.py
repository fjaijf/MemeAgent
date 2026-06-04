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

    def _build_user_content_with_images(
        self,
        text: str,
        image_paths: list[str],
        image_urls: list[str],
    ) -> str | list[dict[str, Any]]:
        if not image_paths and not image_urls:
            return text

        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
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
        return content_blocks

    def describe_images_for_search(
        self,
        topic: str = "",
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        image_paths = image_paths or []
        image_urls = image_urls or []
        if not image_paths and not image_urls:
            return ""

        user_prompt = f"""
Topic hint: {topic or "None"}

User-provided context:
{context or "None"}

Look at the attached meme image(s) and produce a search-oriented visual description in Chinese.
Focus on details that can help retrieve related background information before the final analysis.

Return these sections:
1. OCR/text visible in the image: transcribe all visible text, preserving language and notable spelling.
2. Visual description: describe people, characters, objects, layout, symbols, expressions, gestures, colors, and editing style.
3. Meme template or cultural references: identify possible meme templates, named figures, events, slogans, fandoms, communities, or platform conventions.
4. Harm/sentiment cues: list visible cues related to hostility, stereotyping, misinformation, persuasion, mobilization, irony, satire, fear, anger, ridicule, or ambiguity.
5. Search keywords: provide 8-15 concise keywords or phrases in both Chinese and English when useful. Include exact OCR phrases, likely meme names, and visual entities.

Be descriptive rather than interpretive. If something is uncertain, mark it as uncertain.
""".strip()

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self._build_user_content_with_images(
                    user_prompt,
                    image_paths,
                    image_urls,
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))

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
                "If the images contain memes, symbols, screenshots, comments, "
                "captions, edits, or social posts, explain what is visible and "
                "how it affects harmfulness, sentiment, audience reception, "
                "intent, and meme evolution.\n"
            )

        user_prompt = f"""
Topic: {topic}

Extra context:
{context or "None"}
{image_instruction}

Please produce a researcher-oriented meme analysis in Chinese unless the user asks otherwise.
Do not frame the topic as crypto, finance, or market speculation unless the topic or evidence explicitly requires it.

Required sections:
1. Meme object and context: identify the meme, its symbols, references, and likely cultural setting.
2. Sentiment analysis: describe dominant emotions, polarity, intensity, ambiguity, and possible audience split.
3. Harmfulness analysis: assess risks such as harassment, hate, stereotyping, misinformation, manipulation, panic, self-harm encouragement, radicalization, or reputational damage. Give a low/medium/high risk level with reasons.
4. Audience and reception prediction: infer likely target audiences, vulnerable groups, in-groups/out-groups, and likely interpretations.
5. Intent recognition: infer whether the meme appears humorous, satirical, persuasive, mobilizing, deceptive, provocative, identity-signaling, commercial, or coordinated. Separate evidence from speculation.
6. Evolution tracking: explain possible origin clues, variants, mutation paths, cross-platform spread, and how meaning may change over time.
7. Evidence gaps and confidence: state what the search/image/context supports, what remains uncertain, and a confidence level.
""".strip()


        user_content: str | list[dict[str, Any]]
        user_content = self._build_user_content_with_images(
            user_prompt,
            image_paths,
            image_urls,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        return _normalize_content(getattr(response, "content", response))
