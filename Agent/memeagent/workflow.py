from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from .agent import MemeAgent
from .search_agent import WebSearchAgent


@dataclass(frozen=True)
class WorkflowResult:
    search_report: str
    analysis: str
    combined_context: str
    visual_report: str = ""
    retrieval_plan: str = ""
    input_mode: str = "text_only"


class MemeResearchWorkflow:
    """Coordinates image description, retrieval, then meme analysis."""

    def __init__(self, meme_agent: MemeAgent, search_agent: WebSearchAgent) -> None:
        self.meme_agent = meme_agent
        self.search_agent = search_agent

    def _emit_progress(
        self,
        progress: Callable[[str, str], None] | None,
        stage: str,
        message: str,
    ) -> None:
        if progress:
            progress(stage, message)

    def _detect_input_mode(
        self,
        topic: str,
        context: str,
        image_paths: list[str],
        image_urls: list[str],
    ) -> str:
        has_text = bool(topic.strip() or context.strip())
        has_images = bool(image_paths or image_urls)

        if has_text and has_images:
            return "text_and_image"
        if has_images:
            return "image_only"
        if has_text:
            return "text_only"
        raise ValueError("Please provide at least --topic, --context, --image, or --image-url.")

    def _extract_search_terms(self, visual_report: str) -> str:
        quoted_terms = re.findall(r'"([^"]{2,80})"', visual_report)
        terms: list[str] = []

        for term in quoted_terms:
            cleaned = " ".join(term.split()).strip()
            if cleaned and cleaned.lower() not in {item.lower() for item in terms}:
                terms.append(cleaned)
            if len(terms) >= 8:
                break

        if terms:
            return " ".join(f'"{term}"' for term in terms)

        fallback_terms = re.findall(r"[A-Za-z][A-Za-z0-9' -]{2,40}", visual_report)
        for term in fallback_terms:
            cleaned = " ".join(term.split()).strip()
            if cleaned and cleaned.lower() not in {item.lower() for item in terms}:
                terms.append(cleaned)
            if len(terms) >= 10:
                break

        return " ".join(terms)

    def _build_search_context(
        self,
        context: str,
        visual_report: str,
        retrieval_plan: str,
    ) -> str:
        parts: list[str] = []
        if context.strip():
            parts.append(context.strip())

        if visual_report.strip():
            search_terms = self._extract_search_terms(visual_report)
            if search_terms:
                parts.append(search_terms)

        if retrieval_plan.strip():
            parts.append("Supplemental retrieval plan:\n" + retrieval_plan.strip())

        return "\n\n".join(parts)

    def run(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
        use_search: bool = True,
        progress: Callable[[str, str], None] | None = None,
        stream_analysis: bool = False,
        analysis_delta: Callable[[str], None] | None = None,
        search_ready: Callable[[str, str, str, str], None] | None = None,
    ) -> WorkflowResult:
        image_paths = image_paths or []
        image_urls = image_urls or []
        search_report = ""
        visual_report = ""
        retrieval_plan = ""
        combined_parts: list[str] = []
        input_mode = self._detect_input_mode(topic, context, image_paths, image_urls)
        has_images = input_mode in {"image_only", "text_and_image"}
        self._emit_progress(progress, "input", f"Input mode detected: {input_mode}.")

        if context.strip():
            combined_parts.append(context.strip())

        if has_images:
            self._emit_progress(
                progress,
                "vision",
                "Describing image content for retrieval and analysis.",
            )
            try:
                visual_report = self.meme_agent.describe_images_for_search(
                    topic=topic,
                    context=context,
                    image_paths=image_paths,
                    image_urls=image_urls,
                )
            except Exception as exc:
                raise RuntimeError("Image pre-analysis LLM call failed.") from exc
            self._emit_progress(progress, "vision", "Image-derived search context ready.")
            if visual_report:
                combined_parts.append(
                    "Image-derived meme description for retrieval and analysis:\n"
                    + visual_report
                )

        if use_search:
            self._emit_progress(
                progress,
                "planning",
                "Planning supplemental retrieval queries.",
            )
            try:
                retrieval_plan = self.meme_agent.plan_retrieval(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    input_mode=input_mode,
                )
            except Exception as exc:
                raise RuntimeError("Retrieval planning LLM call failed.") from exc
            if retrieval_plan:
                combined_parts.append("Supplemental retrieval plan:\n" + retrieval_plan)
            self._emit_progress(progress, "planning", "Supplemental retrieval plan ready.")
            self._emit_progress(progress, "search", "Collecting web and news context.")
            search_context = self._build_search_context(
                context,
                visual_report,
                retrieval_plan,
            )
            search_report = self.search_agent.run(topic=topic, context=search_context)
            combined_parts.append(
                "Internet search findings. Cite web sources as [W#] and news "
                "sources as [N#] exactly as labeled below:\n"
                + search_report
            )
            self._emit_progress(progress, "search", "Retrieval finished.")

        combined_context = "\n\n".join(part for part in combined_parts if part).strip()
        if search_ready:
            search_ready(input_mode, search_report, visual_report, retrieval_plan)
        self._emit_progress(progress, "analysis", "Running final meme analysis.")
        try:
            if stream_analysis:
                chunks: list[str] = []
                try:
                    for chunk in self.meme_agent.stream(
                        topic=topic,
                        context=combined_context,
                        image_paths=image_paths,
                        image_urls=image_urls,
                    ):
                        chunks.append(chunk)
                        if analysis_delta:
                            analysis_delta(chunk)
                except Exception as stream_exc:
                    if not chunks:
                        analysis = self.meme_agent.run(
                            topic=topic,
                            context=combined_context,
                            image_paths=image_paths,
                            image_urls=image_urls,
                        )
                    else:
                        warning = (
                            "\n\n[Stream Warning] The streaming connection closed "
                            "before the final chunk arrived. The analysis above is "
                            "the partial output received before interruption."
                        )
                        chunks.append(warning)
                        if analysis_delta:
                            analysis_delta(warning)
                        analysis = "".join(chunks)
                    logger_message = getattr(stream_exc, "args", [""])[0]
                    self._emit_progress(
                        progress,
                        "analysis",
                        f"Streaming ended early: {logger_message}",
                    )
                else:
                    analysis = "".join(chunks)
            else:
                analysis = self.meme_agent.run(
                    topic=topic,
                    context=combined_context,
                    image_paths=image_paths,
                    image_urls=image_urls,
                )
        except Exception as exc:
            raise RuntimeError("Final analysis LLM call failed.") from exc
        self._emit_progress(progress, "analysis", "Final analysis ready.")
        return WorkflowResult(
            search_report=search_report,
            analysis=analysis,
            combined_context=combined_context,
            visual_report=visual_report,
            retrieval_plan=retrieval_plan,
            input_mode=input_mode,
        )
