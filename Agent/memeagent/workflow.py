from __future__ import annotations

from dataclasses import dataclass

from .agent import MemeAgent
from .search_agent import WebSearchAgent


@dataclass(frozen=True)
class WorkflowResult:
    search_report: str
    analysis: str
    combined_context: str


class MemeResearchWorkflow:
    """Coordinates retrieval first, then meme analysis."""

    def __init__(self, meme_agent: MemeAgent, search_agent: WebSearchAgent) -> None:
        self.meme_agent = meme_agent
        self.search_agent = search_agent

    def run(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
        use_search: bool = True,
    ) -> WorkflowResult:
        search_report = ""
        combined_parts: list[str] = []

        if context.strip():
            combined_parts.append(context.strip())

        if use_search:
            search_report = self.search_agent.run(topic=topic, context=context)
            combined_parts.append("Internet search findings:\n" + search_report)

        combined_context = "\n\n".join(part for part in combined_parts if part).strip()
        analysis = self.meme_agent.run(
            topic=topic,
            context=combined_context,
            image_paths=image_paths,
            image_urls=image_urls,
        )
        return WorkflowResult(
            search_report=search_report,
            analysis=analysis,
            combined_context=combined_context,
        )
