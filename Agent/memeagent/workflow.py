from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from .agent import MemeAgent
from .heads import (
    HeadResult,
    MemeAnalysisHeadRunner,
    format_head_results,
    normalize_head_names,
)
from .memory import MemeMemoryStore
from .search_agent import WebSearchAgent


@dataclass(frozen=True)
class WorkflowResult:
    search_report: str
    analysis: str
    combined_context: str
    visual_report: str = ""
    retrieval_plan: str = ""
    memory_report: str = ""
    input_mode: str = "text_only"


@dataclass(frozen=True)
class MultiHeadWorkflowResult:
    head_results: list[HeadResult]
    formatted_output: str
    search_report: str
    combined_context: str
    visual_report: str = ""
    retrieval_plan: str = ""
    memory_report: str = ""
    input_mode: str = "text_only"


class MemeResearchWorkflow:
    """Coordinates image description, retrieval, then meme analysis."""

    def __init__(
        self,
        meme_agent: MemeAgent,
        search_agent: WebSearchAgent,
        controller_agent: MemeAgent | None = None,
        memory_store: MemeMemoryStore | None = None,
        memory_recall_limit: int = 3,
    ) -> None:
        self.meme_agent = meme_agent
        self.controller_agent = controller_agent or meme_agent
        self.search_agent = search_agent
        self.memory_store = memory_store
        self.memory_recall_limit = memory_recall_limit

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

    def _reflection_should_continue(self, reflection: str) -> bool:
        match = re.search(
            r"SHOULD_CONTINUE\s*:\s*(?:[-*]\s*)?(yes|no)\b",
            reflection,
            flags=re.I,
        )
        if match:
            return match.group(1).lower() == "yes"
        return False

    def _reflection_has_queries(self, reflection: str) -> bool:
        return bool(
            re.search(
                r"SUPPLEMENTAL_(?:WEB|NEWS)_QUERIES\s*:\s*\n\s*[-*]\s+(?!None\b|无\b|没有\b)",
                reflection,
                flags=re.I,
            )
        )

    def _retrieval_decision_needs_search(self, retrieval_decision: str) -> bool:
        match = re.search(
            r"RETRIEVAL_NEEDED\s*:\s*(?:[-*]\s*)?(yes|no)\b",
            retrieval_decision,
            flags=re.I,
        )
        if match:
            return match.group(1).lower() == "yes"
        return True

    def _prepare_retrieval_plan(
        self,
        topic: str,
        context: str,
        visual_report: str,
        memory_report: str,
        input_mode: str,
        force_search: bool,
        progress: Callable[[str, str], None] | None,
    ) -> tuple[str, bool, str]:
        if force_search:
            self._emit_progress(
                progress,
                "planning",
                "Force search enabled; planning retrieval queries.",
            )
            try:
                retrieval_plan = self.controller_agent.plan_retrieval(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    input_mode=input_mode,
                )
            except Exception as exc:
                raise RuntimeError("Forced retrieval planning LLM call failed.") from exc
            return retrieval_plan, True, "Forced retrieval plan"

        self._emit_progress(
            progress,
            "planning",
            "Deciding whether external retrieval is needed.",
        )
        try:
            retrieval_plan = self.controller_agent.decide_retrieval(
                topic=topic,
                context=context,
                visual_report=visual_report,
                memory_report=memory_report,
                input_mode=input_mode,
            )
        except Exception as exc:
            raise RuntimeError("Retrieval decision LLM call failed.") from exc
        return (
            retrieval_plan,
            self._retrieval_decision_needs_search(retrieval_plan),
            "Retrieval decision and query plan",
        )

    def _label_search_report(self, search_report: str, round_index: int) -> str:
        if round_index <= 1:
            return search_report

        def replace_label(match: re.Match[str]) -> str:
            return f"[R{round_index}-{match.group(1)}{match.group(2)}]"

        return re.sub(r"\[(W|N)(\d+)\]", replace_label, search_report)

    def _run_search_with_reflection(
        self,
        topic: str,
        context: str,
        visual_report: str,
        retrieval_plan: str,
        input_mode: str,
        iterative_search: bool,
        search_max_rounds: int,
        progress: Callable[[str, str], None] | None,
    ) -> str:
        search_context = self._build_search_context(
            context,
            visual_report,
            retrieval_plan,
        )
        self._emit_progress(progress, "search", "Collecting web and news context.")
        first_report = self._label_search_report(
            self.search_agent.run(topic=topic, context=search_context),
            round_index=1,
        )
        if not iterative_search or search_max_rounds <= 1:
            self._emit_progress(progress, "search", "Retrieval finished.")
            return first_report

        round_blocks = [f"## Retrieval Round 1\n\n{first_report}"]
        cumulative_report = "\n\n".join(round_blocks)
        base_context = search_context
        max_rounds = max(1, search_max_rounds)

        for next_round in range(2, max_rounds + 1):
            reflection_round = next_round - 1
            self._emit_progress(
                progress,
                "reflection",
                f"Reflecting on retrieval round {reflection_round}.",
            )
            try:
                reflection = self.controller_agent.reflect_retrieval(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    retrieval_plan=retrieval_plan,
                    search_report=cumulative_report,
                    input_mode=input_mode,
                    round_index=reflection_round,
                    max_rounds=max_rounds,
                )
            except Exception as exc:
                raise RuntimeError("Retrieval reflection LLM call failed.") from exc

            round_blocks.append(
                f"## Retrieval Reflection after Round {reflection_round}\n\n{reflection}"
            )
            cumulative_report = "\n\n".join(round_blocks)
            if not self._reflection_should_continue(reflection):
                self._emit_progress(
                    progress,
                    "reflection",
                    "Retrieval reflection decided to stop.",
                )
                break
            if not self._reflection_has_queries(reflection):
                self._emit_progress(
                    progress,
                    "reflection",
                    "Retrieval reflection found no concrete follow-up queries.",
                )
                break

            self._emit_progress(
                progress,
                "search",
                f"Collecting retrieval round {next_round}.",
            )
            followup_context = (
                f"{base_context}\n\n"
                f"Retrieval reflection for round {next_round}:\n{reflection}"
            )
            next_report = self._label_search_report(
                self.search_agent.run(topic=topic, context=followup_context),
                round_index=next_round,
            )
            round_blocks.append(f"## Retrieval Round {next_round}\n\n{next_report}")
            cumulative_report = "\n\n".join(round_blocks)

        self._emit_progress(progress, "search", "Iterative retrieval finished.")
        return cumulative_report

    def run(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
        use_search: bool = True,
        force_search: bool = False,
        progress: Callable[[str, str], None] | None = None,
        stream_analysis: bool = False,
        analysis_delta: Callable[[str], None] | None = None,
        search_ready: Callable[[str, str, str, str], None] | None = None,
        iterative_search: bool = False,
        search_max_rounds: int = 3,
    ) -> WorkflowResult:
        image_paths = image_paths or []
        image_urls = image_urls or []
        search_report = ""
        visual_report = ""
        retrieval_plan = ""
        memory_report = ""
        combined_parts: list[str] = []
        input_mode = self._detect_input_mode(topic, context, image_paths, image_urls)
        has_images = input_mode in {"image_only", "text_and_image"}
        self._emit_progress(progress, "input", f"Input mode detected: {input_mode}.")

        if context.strip():
            combined_parts.append(context.strip())

        if self.memory_store:
            self._emit_progress(progress, "memory", "Checking local MemeAgent memory.")
            memory_records = self.memory_store.recall(
                topic=topic,
                image_paths=image_paths,
                image_urls=image_urls,
                limit=self.memory_recall_limit,
            )
            memory_card = self.memory_store.recall_card(topic)
            memory_report = self.memory_store.format_records(
                memory_records,
                card=memory_card,
            )
            if memory_report:
                combined_parts.append(memory_report)
                self._emit_progress(progress, "memory", "Local memory attached.")
            else:
                self._emit_progress(progress, "memory", "No local memory found.")

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

        if use_search or force_search:
            retrieval_plan, should_search, plan_label = self._prepare_retrieval_plan(
                topic=topic,
                context=context,
                visual_report=visual_report,
                memory_report=memory_report,
                input_mode=input_mode,
                force_search=force_search,
                progress=progress,
            )
            if retrieval_plan:
                combined_parts.append(f"{plan_label}:\n" + retrieval_plan)
            if should_search:
                planning_message = (
                    "External retrieval forced."
                    if force_search
                    else "External retrieval approved."
                )
                self._emit_progress(progress, "planning", planning_message)
                search_report = self._run_search_with_reflection(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    retrieval_plan=retrieval_plan,
                    input_mode=input_mode,
                    iterative_search=iterative_search,
                    search_max_rounds=search_max_rounds,
                    progress=progress,
                )
                combined_parts.append(
                    "Internet search findings. Cite source labels exactly as shown below. "
                    "Single-round labels look like [W1] or [N1]; iterative retrieval may "
                    "also include labels such as [R2-W1] or [R2-N1]:\n"
                    + search_report
                )
            else:
                self._emit_progress(
                    progress,
                    "planning",
                    "External retrieval skipped by LLM decision.",
                )

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

        if self.memory_store:
            self._emit_progress(progress, "memory", "Saving analysis to local memory.")
            self.memory_store.remember(
                topic=topic,
                image_paths=image_paths,
                image_urls=image_urls,
                input_mode=input_mode,
                analysis=analysis,
                visual_report=visual_report,
                retrieval_plan=retrieval_plan,
                search_report=search_report,
            )
        self._emit_progress(progress, "analysis", "Final analysis ready.")
        return WorkflowResult(
            search_report=search_report,
            analysis=analysis,
            combined_context=combined_context,
            visual_report=visual_report,
            retrieval_plan=retrieval_plan,
            memory_report=memory_report,
            input_mode=input_mode,
        )

    def run_heads(
        self,
        topic: str,
        context: str = "",
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
        task_heads: list[str] | None = None,
        use_search: bool = True,
        force_search: bool = False,
        progress: Callable[[str, str], None] | None = None,
        search_ready: Callable[[str, str, str, str], None] | None = None,
        iterative_search: bool = False,
        search_max_rounds: int = 3,
    ) -> MultiHeadWorkflowResult:
        image_paths = image_paths or []
        image_urls = image_urls or []
        head_names = normalize_head_names(task_heads)
        search_report = ""
        visual_report = ""
        retrieval_plan = ""
        memory_report = ""
        combined_parts: list[str] = []
        input_mode = self._detect_input_mode(topic, context, image_paths, image_urls)
        has_images = input_mode in {"image_only", "text_and_image"}
        self._emit_progress(progress, "input", f"Input mode detected: {input_mode}.")

        if context.strip():
            combined_parts.append("[User Context]\n" + context.strip())

        if self.memory_store:
            self._emit_progress(progress, "memory", "Checking local MemeAgent memory.")
            memory_records = self.memory_store.recall(
                topic=topic,
                image_paths=image_paths,
                image_urls=image_urls,
                limit=self.memory_recall_limit,
            )
            memory_card = self.memory_store.recall_card(topic)
            memory_report = self.memory_store.format_records(
                memory_records,
                card=memory_card,
            )
            if memory_report:
                combined_parts.append("Local memory:\n" + memory_report)
                self._emit_progress(progress, "memory", "Local memory attached.")
            else:
                self._emit_progress(progress, "memory", "No local memory found.")

        if has_images:
            self._emit_progress(
                progress,
                "vision",
                "Describing image content for retrieval and task heads.",
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
            self._emit_progress(progress, "vision", "Image-derived context ready.")
            if visual_report:
                combined_parts.append(
                    "[Image]\nImage-derived meme description for retrieval and analysis:\n"
                    + visual_report
                )

        if use_search or force_search:
            retrieval_plan, should_search, plan_label = self._prepare_retrieval_plan(
                topic=topic,
                context=context,
                visual_report=visual_report,
                memory_report=memory_report,
                input_mode=input_mode,
                force_search=force_search,
                progress=progress,
            )
            if retrieval_plan:
                combined_parts.append(f"{plan_label}:\n" + retrieval_plan)
            if should_search:
                planning_message = (
                    "External retrieval forced."
                    if force_search
                    else "External retrieval approved."
                )
                self._emit_progress(progress, "planning", planning_message)
                search_report = self._run_search_with_reflection(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    retrieval_plan=retrieval_plan,
                    input_mode=input_mode,
                    iterative_search=iterative_search,
                    search_max_rounds=search_max_rounds,
                    progress=progress,
                )
                combined_parts.append(
                    "Internet search findings. Cite source labels exactly as shown below. "
                    "Single-round labels look like [W1] or [N1]; iterative retrieval may "
                    "also include labels such as [R2-W1] or [R2-N1]:\n"
                    + search_report
                )
            else:
                self._emit_progress(
                    progress,
                    "planning",
                    "External retrieval skipped by LLM decision.",
                )

        combined_context = "\n\n".join(part for part in combined_parts if part).strip()
        if search_ready:
            search_ready(input_mode, search_report, visual_report, retrieval_plan)

        head_list = ", ".join(head_names)
        self._emit_progress(progress, "heads", f"Running task heads: {head_list}.")
        runner = MemeAnalysisHeadRunner(
            llm=self.meme_agent.llm,
            system_prompt=self.meme_agent.system_prompt,
        )
        try:
            head_results = runner.run_heads(
                head_names=head_names,
                topic=topic,
                evidence_context=combined_context,
                input_mode=input_mode,
            )
        except Exception as exc:
            raise RuntimeError("Multi-head analysis LLM call failed.") from exc

        formatted_output = format_head_results(head_results)
        if self.memory_store:
            self._emit_progress(progress, "memory", "Saving multi-head analysis to local memory.")
            self.memory_store.remember(
                topic=topic,
                image_paths=image_paths,
                image_urls=image_urls,
                input_mode=input_mode,
                analysis=formatted_output,
                visual_report=visual_report,
                retrieval_plan=retrieval_plan,
                search_report=search_report,
            )
        self._emit_progress(progress, "heads", "Task heads ready.")
        return MultiHeadWorkflowResult(
            head_results=head_results,
            formatted_output=formatted_output,
            search_report=search_report,
            combined_context=combined_context,
            visual_report=visual_report,
            retrieval_plan=retrieval_plan,
            memory_report=memory_report,
            input_mode=input_mode,
        )
