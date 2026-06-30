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
from .trajectory import MemeTrajectoryCache


@dataclass(frozen=True)
class WorkflowResult:
    search_report: str
    analysis: str
    combined_context: str
    visual_report: str = ""
    retrieval_plan: str = ""
    controller_report: str = ""
    memory_report: str = ""
    input_mode: str = "text_only"
    trajectory_run_id: str = ""


@dataclass(frozen=True)
class MultiHeadWorkflowResult:
    head_results: list[HeadResult]
    formatted_output: str
    search_report: str
    combined_context: str
    visual_report: str = ""
    retrieval_plan: str = ""
    controller_report: str = ""
    memory_report: str = ""
    input_mode: str = "text_only"
    trajectory_run_id: str = ""


class MemeResearchWorkflow:
    """Coordinates image description, retrieval, then meme analysis."""

    def __init__(
        self,
        meme_agent: MemeAgent,
        search_agent: WebSearchAgent,
        controller_agent: MemeAgent | None = None,
        memory_store: MemeMemoryStore | None = None,
        memory_recall_limit: int = 3,
        trajectory_cache: MemeTrajectoryCache | None = None,
    ) -> None:
        self.meme_agent = meme_agent
        self.controller_agent = controller_agent or meme_agent
        self.search_agent = search_agent
        self.memory_store = memory_store
        self.memory_recall_limit = memory_recall_limit
        self.trajectory_cache = trajectory_cache

    def _emit_progress(
        self,
        progress: Callable[[str, str], None] | None,
        stage: str,
        message: str,
    ) -> None:
        if progress:
            progress(stage, message)

    def _start_trajectory(
        self,
        *,
        workflow_kind: str,
        topic: str,
        context: str,
        image_paths: list[str],
        image_urls: list[str],
        input_mode: str,
        options: dict[str, object],
    ) -> str:
        if not self.trajectory_cache:
            return ""
        try:
            return self.trajectory_cache.start_run(
                workflow_kind=workflow_kind,
                topic=topic,
                context=context,
                image_paths=image_paths,
                image_urls=image_urls,
                input_mode=input_mode,
                options=options,
            )
        except Exception:
            return ""

    def _record_trajectory(
        self,
        run_id: str,
        *,
        stage: str,
        name: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        if not run_id or not self.trajectory_cache:
            return
        try:
            self.trajectory_cache.record_event(
                run_id,
                stage=stage,
                name=name,
                payload=payload or {},
            )
        except Exception:
            return

    def _finish_trajectory(
        self,
        run_id: str,
        *,
        output: dict[str, object],
    ) -> None:
        if not run_id or not self.trajectory_cache:
            return
        try:
            self.trajectory_cache.finish_run(run_id, output=output)
        except Exception:
            return

    def _fail_trajectory(
        self,
        run_id: str,
        *,
        error: Exception,
        output: dict[str, object],
    ) -> None:
        if not run_id or not self.trajectory_cache:
            return
        try:
            self.trajectory_cache.fail_run(run_id, error=str(error), output=output)
        except Exception:
            return

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

    def _controller_confidence(self, controller_plan: str) -> float:
        match = re.search(
            r"ITERATION_CONFIDENCE\s*:\s*(?:[-*]\s*)?([01](?:\.\d+)?)",
            controller_plan,
            flags=re.I,
        )
        if not match:
            return 0.0
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return 0.0

    def _controller_should_finalize(
        self,
        controller_plan: str,
        confidence_threshold: float,
    ) -> bool:
        return self._controller_confidence(controller_plan) >= confidence_threshold

    def _extract_controller_section(self, controller_plan: str, section: str) -> str:
        pattern = re.compile(
            rf"(?ims)^\s*{re.escape(section)}\s*:\s*\n"
            rf"(?P<body>.*?)(?=^\s*[A-Z][A-Z0-9_ ]{{2,}}\s*:\s*$|\Z)"
        )
        match = pattern.search(controller_plan)
        if match:
            return match.group("body").strip()
        inline = re.search(rf"(?im)^\s*{re.escape(section)}\s*:\s*(.+)$", controller_plan)
        return inline.group(1).strip() if inline else ""

    def _normalize_controller_text(self, value: str) -> str:
        normalized = value.lower()
        normalized = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", normalized, flags=re.M)
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
        return " ".join(normalized.split())

    def _is_repeated_controller_request(
        self,
        normalized_request: str,
        prior_request_fingerprints: set[str],
    ) -> bool:
        request_terms = set(normalized_request.split())
        if not request_terms:
            return True
        for prior_request in prior_request_fingerprints:
            prior_terms = set(prior_request.split())
            if not prior_terms:
                continue
            overlap = len(request_terms & prior_terms) / max(
                1,
                min(len(request_terms), len(prior_terms)),
            )
            if overlap >= 0.75:
                return True
        return False

    def _has_new_image_answerable_requests(
        self,
        controller_plan: str,
        prior_request_fingerprints: set[str],
    ) -> bool:
        request_text = "\n".join(
            part
            for part in (
                self._extract_controller_section(controller_plan, "FOCUS_QUESTIONS"),
                self._extract_controller_section(
                    controller_plan,
                    "MULTIMODAL_ANALYSIS_REQUESTS",
                ),
            )
            if part
        )
        normalized = self._normalize_controller_text(request_text)
        if not normalized or normalized in {"none", "无", "没有"}:
            return False

        external_only_terms = (
            "source",
            "origin",
            "platform",
            "reddit",
            "twitter",
            "x com",
            "tiktok",
            "template lineage",
            "prior iteration",
            "prior iterations",
            "known meme",
            "stock photo database",
            "web",
            "search",
            "external",
            "documented",
            "document",
            "来源",
            "平台",
            "检索",
            "搜索",
            "外部",
            "出处",
            "源头",
            "模板谱系",
        )
        image_terms = (
            "image",
            "visual",
            "ocr",
            "text",
            "layout",
            "watermark",
            "ui",
            "logo",
            "symbol",
            "visible",
            "pixel",
            "pose",
            "framing",
            "gesture",
            "color",
            "图像",
            "视觉",
            "文字",
            "水印",
            "可见",
            "姿势",
            "构图",
            "符号",
        )
        has_image_terms = any(term in normalized for term in image_terms)
        has_external_terms = any(term in normalized for term in external_only_terms)
        if has_external_terms and not has_image_terms:
            return False

        fingerprint = normalized[:500]
        if self._is_repeated_controller_request(fingerprint, prior_request_fingerprints):
            return False
        prior_request_fingerprints.add(fingerprint)
        return True

    def _prepare_retrieval_plan(
        self,
        topic: str,
        context: str,
        visual_report: str,
        input_mode: str,
        progress: Callable[[str, str], None] | None,
    ) -> tuple[str, str]:
        self._emit_progress(
            progress,
            "planning",
            "Planning forced retrieval queries.",
        )
        try:
            retrieval_plan = self.controller_agent.plan_retrieval(
                topic=topic,
                context=context,
                visual_report=visual_report,
                input_mode=input_mode,
            )
        except Exception as exc:
            raise RuntimeError("Retrieval planning LLM call failed.") from exc
        return retrieval_plan, "Forced retrieval plan"

    def _plan_analysis_iteration(
        self,
        topic: str,
        context: str,
        visual_report: str,
        search_report: str,
        iteration_history: str,
        input_mode: str,
        round_index: int,
        max_rounds: int,
        confidence_threshold: float,
        retrieval_enabled: bool,
        progress: Callable[[str, str], None] | None,
    ) -> str:
        self._emit_progress(
            progress,
            "controller",
            f"Planning analysis round {round_index}.",
        )
        try:
            return self.controller_agent.plan_analysis_iteration(
                topic=topic,
                context=context,
                visual_report=visual_report,
                search_report=search_report,
                iteration_history=iteration_history,
                input_mode=input_mode,
                round_index=round_index,
                max_rounds=max_rounds,
                confidence_threshold=confidence_threshold,
                retrieval_enabled=retrieval_enabled,
            )
        except Exception as exc:
            raise RuntimeError("Controller analysis planning LLM call failed.") from exc

    def _analyze_images_for_controller_plan(
        self,
        topic: str,
        context: str,
        controller_plan: str,
        previous_visual_report: str,
        image_paths: list[str],
        image_urls: list[str],
        progress: Callable[[str, str], None] | None,
    ) -> str:
        if not image_paths and not image_urls:
            return ""

        self._emit_progress(progress, "vision", "Re-examining image content for controller questions.")
        try:
            return self.meme_agent.analyze_images_for_plan(
                topic=topic,
                context=context,
                controller_plan=controller_plan,
                previous_visual_report=previous_visual_report,
                image_paths=image_paths,
                image_urls=image_urls,
            )
        except Exception as exc:
            raise RuntimeError("Image follow-up analysis LLM call failed.") from exc

    def _run_controller_analysis_loop(
        self,
        topic: str,
        context: str,
        visual_report: str,
        search_report: str,
        input_mode: str,
        image_paths: list[str],
        image_urls: list[str],
        max_rounds: int,
        confidence_threshold: float,
        allow_retrieval: bool,
        progress: Callable[[str, str], None] | None,
    ) -> tuple[str, str, str]:
        max_rounds = max(1, max_rounds)
        round_blocks: list[str] = []
        cumulative_visual_report = visual_report
        cumulative_search_report = search_report
        prior_no_search_request_fingerprints: set[str] = set()

        for round_index in range(1, max_rounds + 1):
            iteration_history = "\n\n".join(round_blocks)
            controller_plan = self._plan_analysis_iteration(
                topic=topic,
                context=context,
                visual_report=cumulative_visual_report,
                search_report=cumulative_search_report,
                iteration_history=iteration_history,
                input_mode=input_mode,
                round_index=round_index,
                max_rounds=max_rounds,
                confidence_threshold=confidence_threshold,
                retrieval_enabled=allow_retrieval,
                progress=progress,
            )
            confidence = self._controller_confidence(controller_plan)
            self._emit_progress(
                progress,
                "controller",
                (
                    f"Round {round_index} confidence {confidence:.2f}; "
                    f"threshold {confidence_threshold:.2f}."
                ),
            )
            round_blocks.append(
                f"## Controller Round {round_index}\n\n"
                f"{controller_plan}\n\n"
                f"Parsed confidence: {confidence:.2f}"
            )

            if self._controller_should_finalize(controller_plan, confidence_threshold):
                self._emit_progress(
                    progress,
                    "controller",
                    f"Confidence {confidence:.2f} reached threshold {confidence_threshold:.2f}.",
                )
                break
            if round_index >= max_rounds:
                self._emit_progress(
                    progress,
                    "controller",
                    f"Max controller rounds reached with confidence {confidence:.2f}.",
                )
                break

            if not allow_retrieval and not self._has_new_image_answerable_requests(
                controller_plan,
                prior_no_search_request_fingerprints,
            ):
                self._emit_progress(
                    progress,
                    "controller",
                    (
                        "No new image-answerable controller questions remain; "
                        "continuing with offline evidence."
                    ),
                )
                round_blocks.append(
                    "## No-Search Controller Stop\n\n"
                    "External retrieval is disabled and the remaining gaps are "
                    "not answerable by re-inspecting the current image."
                )
                break

            self._emit_progress(
                progress,
                "controller",
                (
                    "Confidence below threshold; requesting follow-up image analysis "
                    "and retrieval."
                    if allow_retrieval
                    else "Confidence below threshold; requesting follow-up image analysis."
                ),
            )
            followup_visual = self._analyze_images_for_controller_plan(
                topic=topic,
                context=context,
                controller_plan=controller_plan,
                previous_visual_report=cumulative_visual_report,
                image_paths=image_paths,
                image_urls=image_urls,
                progress=progress,
            )
            if followup_visual:
                cumulative_visual_report = (
                    f"{cumulative_visual_report}\n\n"
                    f"## Follow-up Image Analysis Round {round_index}\n\n"
                    f"{followup_visual}"
                ).strip()
                round_blocks.append(
                    f"## Follow-up Image Analysis Round {round_index}\n\n{followup_visual}"
                )

            if allow_retrieval:
                self._emit_progress(
                    progress,
                    "search",
                    f"Collecting controller-directed retrieval round {round_index + 1}.",
                )
                followup_context = (
                    f"{context}\n\n"
                    f"Controller plan for round {round_index + 1}:\n{controller_plan}"
                )
                if followup_visual:
                    followup_context += f"\n\nFollow-up image analysis:\n{followup_visual}"
                followup_report = self._label_search_report(
                    self.search_agent.run(topic=topic, context=followup_context),
                    round_index=round_index + 1,
                )
                cumulative_search_report = (
                    f"{cumulative_search_report}\n\n"
                    f"## Controller-Directed Retrieval Round {round_index + 1}\n\n"
                    f"{followup_report}"
                ).strip()
                round_blocks.append(
                    f"## Controller-Directed Retrieval Round {round_index + 1}\n\n"
                    f"{followup_report}"
                )

        return "\n\n".join(round_blocks), cumulative_visual_report, cumulative_search_report

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
        search_ready: Callable[[str, str, str, str, str], None] | None = None,
        iterative_search: bool = False,
        search_max_rounds: int = 3,
        controller_max_rounds: int = 3,
        controller_confidence_threshold: float = 0.8,
    ) -> WorkflowResult:
        image_paths = image_paths or []
        image_urls = image_urls or []
        search_report = ""
        visual_report = ""
        retrieval_plan = ""
        controller_report = ""
        memory_report = ""
        combined_context = ""
        input_mode = ""
        trajectory_run_id = ""

        try:
            combined_parts: list[str] = []
            input_mode = self._detect_input_mode(topic, context, image_paths, image_urls)
            has_images = input_mode in {"image_only", "text_and_image"}
            trajectory_run_id = self._start_trajectory(
                workflow_kind="analysis",
                topic=topic,
                image_paths=image_paths,
                image_urls=image_urls,
                context=context,
                input_mode=input_mode,
                options={
                    "use_search": use_search,
                    "force_search": force_search,
                    "stream_analysis": stream_analysis,
                    "iterative_search": iterative_search,
                    "search_max_rounds": search_max_rounds,
                    "controller_max_rounds": controller_max_rounds,
                    "controller_confidence_threshold": controller_confidence_threshold,
                },
            )
            self._record_trajectory(
                trajectory_run_id,
                stage="input",
                name="input_detected",
                payload={
                    "input_mode": input_mode,
                    "topic": topic,
                    "context": context,
                    "image_paths": image_paths,
                    "image_urls": image_urls,
                },
            )
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="memory",
                    name="memory_recalled",
                    payload={
                        "record_count": len(memory_records),
                        "has_topic_card": memory_card is not None,
                        "memory_report": memory_report,
                    },
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="vision",
                    name="initial_visual_report",
                    payload={"visual_report": visual_report},
                )
                self._emit_progress(progress, "vision", "Image-derived search context ready.")

            if use_search:
                retrieval_plan, plan_label = self._prepare_retrieval_plan(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    input_mode=input_mode,
                    progress=progress,
                )
                self._record_trajectory(
                    trajectory_run_id,
                    stage="planning",
                    name="retrieval_plan_ready",
                    payload={"plan_label": plan_label, "retrieval_plan": retrieval_plan},
                )
                if retrieval_plan:
                    combined_parts.append(f"{plan_label}:\n" + retrieval_plan)
                self._emit_progress(progress, "planning", "External retrieval forced.")
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="search",
                    name="search_report_ready",
                    payload={"search_report": search_report},
                )
            else:
                self._emit_progress(progress, "planning", "External retrieval disabled.")
                self._record_trajectory(
                    trajectory_run_id,
                    stage="planning",
                    name="retrieval_skipped",
                    payload={"reason": "use_search is false"},
                )

            controller_report, visual_report, search_report = self._run_controller_analysis_loop(
                topic=topic,
                context=context,
                visual_report=visual_report,
                search_report=search_report,
                input_mode=input_mode,
                image_paths=image_paths,
                image_urls=image_urls,
                max_rounds=controller_max_rounds,
                confidence_threshold=controller_confidence_threshold,
                allow_retrieval=use_search,
                progress=progress,
            )
            self._record_trajectory(
                trajectory_run_id,
                stage="controller",
                name="controller_loop_ready",
                payload={
                    "controller_report": controller_report,
                    "visual_report": visual_report,
                    "search_report": search_report,
                },
            )
            if controller_report:
                combined_parts.append(
                    "Controller planning and confidence report. Use these focus notes, "
                    "but still ground the final answer in source labels:\n"
                    + controller_report
                )
            if visual_report:
                combined_parts.append(
                    "Cumulative image analysis after controller-directed passes:\n"
                    + visual_report
                )
            if search_report:
                combined_parts.append(
                    "Cumulative internet search findings after controller-directed passes. "
                    "Cite source labels exactly as shown:\n"
                    + search_report
                )

            combined_context = "\n\n".join(part for part in combined_parts if part).strip()
            self._record_trajectory(
                trajectory_run_id,
                stage="context",
                name="combined_context_ready",
                payload={"combined_context": combined_context},
            )
            if search_ready:
                search_ready(
                    input_mode,
                    search_report,
                    visual_report,
                    retrieval_plan,
                    controller_report,
                )
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
            self._record_trajectory(
                trajectory_run_id,
                stage="analysis",
                name="final_analysis_ready",
                payload={"analysis": analysis, "streamed": stream_analysis},
            )

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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="memory",
                    name="analysis_saved",
                    payload={"saved": True},
                )
            output_payload = {
                "input_mode": input_mode,
                "analysis": analysis,
                "combined_context": combined_context,
                "memory_report": memory_report,
                "visual_report": visual_report,
                "retrieval_plan": retrieval_plan,
                "search_report": search_report,
                "controller_report": controller_report,
            }
            self._finish_trajectory(trajectory_run_id, output=output_payload)
            self._emit_progress(progress, "analysis", "Final analysis ready.")
            return WorkflowResult(
                search_report=search_report,
                analysis=analysis,
                combined_context=combined_context,
                visual_report=visual_report,
                retrieval_plan=retrieval_plan,
                controller_report=controller_report,
                memory_report=memory_report,
                input_mode=input_mode,
                trajectory_run_id=trajectory_run_id,
            )
        except Exception as exc:
            self._fail_trajectory(
                trajectory_run_id,
                error=exc,
                output={
                    "input_mode": input_mode,
                    "combined_context": combined_context,
                    "memory_report": memory_report,
                    "visual_report": visual_report,
                    "retrieval_plan": retrieval_plan,
                    "search_report": search_report,
                    "controller_report": controller_report,
                },
            )
            raise

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
        search_ready: Callable[[str, str, str, str, str], None] | None = None,
        iterative_search: bool = False,
        search_max_rounds: int = 3,
        controller_max_rounds: int = 3,
        controller_confidence_threshold: float = 0.8,
    ) -> MultiHeadWorkflowResult:
        image_paths = image_paths or []
        image_urls = image_urls or []
        head_names = normalize_head_names(task_heads)
        search_report = ""
        visual_report = ""
        retrieval_plan = ""
        controller_report = ""
        memory_report = ""
        combined_context = ""
        formatted_output = ""
        head_results: list[HeadResult] = []
        input_mode = ""
        trajectory_run_id = ""

        try:
            combined_parts: list[str] = []
            input_mode = self._detect_input_mode(topic, context, image_paths, image_urls)
            has_images = input_mode in {"image_only", "text_and_image"}
            trajectory_run_id = self._start_trajectory(
                workflow_kind="multi_head",
                topic=topic,
                image_paths=image_paths,
                image_urls=image_urls,
                context=context,
                input_mode=input_mode,
                options={
                    "task_heads": head_names,
                    "use_search": use_search,
                    "force_search": force_search,
                    "iterative_search": iterative_search,
                    "search_max_rounds": search_max_rounds,
                    "controller_max_rounds": controller_max_rounds,
                    "controller_confidence_threshold": controller_confidence_threshold,
                },
            )
            self._record_trajectory(
                trajectory_run_id,
                stage="input",
                name="input_detected",
                payload={
                    "input_mode": input_mode,
                    "topic": topic,
                    "context": context,
                    "image_paths": image_paths,
                    "image_urls": image_urls,
                    "task_heads": head_names,
                },
            )
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="memory",
                    name="memory_recalled",
                    payload={
                        "record_count": len(memory_records),
                        "has_topic_card": memory_card is not None,
                        "memory_report": memory_report,
                    },
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="vision",
                    name="initial_visual_report",
                    payload={"visual_report": visual_report},
                )
                self._emit_progress(progress, "vision", "Image-derived context ready.")

            if use_search:
                retrieval_plan, plan_label = self._prepare_retrieval_plan(
                    topic=topic,
                    context=context,
                    visual_report=visual_report,
                    input_mode=input_mode,
                    progress=progress,
                )
                self._record_trajectory(
                    trajectory_run_id,
                    stage="planning",
                    name="retrieval_plan_ready",
                    payload={"plan_label": plan_label, "retrieval_plan": retrieval_plan},
                )
                if retrieval_plan:
                    combined_parts.append(f"{plan_label}:\n" + retrieval_plan)
                self._emit_progress(progress, "planning", "External retrieval forced.")
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="search",
                    name="search_report_ready",
                    payload={"search_report": search_report},
                )
            else:
                self._emit_progress(progress, "planning", "External retrieval disabled.")
                self._record_trajectory(
                    trajectory_run_id,
                    stage="planning",
                    name="retrieval_skipped",
                    payload={"reason": "use_search is false"},
                )

            controller_report, visual_report, search_report = self._run_controller_analysis_loop(
                topic=topic,
                context=context,
                visual_report=visual_report,
                search_report=search_report,
                input_mode=input_mode,
                image_paths=image_paths,
                image_urls=image_urls,
                max_rounds=controller_max_rounds,
                confidence_threshold=controller_confidence_threshold,
                allow_retrieval=use_search,
                progress=progress,
            )
            self._record_trajectory(
                trajectory_run_id,
                stage="controller",
                name="controller_loop_ready",
                payload={
                    "controller_report": controller_report,
                    "visual_report": visual_report,
                    "search_report": search_report,
                },
            )
            if controller_report:
                combined_parts.append(
                    "Controller planning and confidence report. Use these focus notes, "
                    "but still ground each task head in source labels:\n"
                    + controller_report
                )
            if visual_report:
                combined_parts.append(
                    "Cumulative image analysis after controller-directed passes:\n"
                    + visual_report
                )
            if search_report:
                combined_parts.append(
                    "Cumulative internet search findings after controller-directed passes. "
                    "Cite source labels exactly as shown:\n"
                    + search_report
                )

            combined_context = "\n\n".join(part for part in combined_parts if part).strip()
            self._record_trajectory(
                trajectory_run_id,
                stage="context",
                name="combined_context_ready",
                payload={"combined_context": combined_context},
            )
            if search_ready:
                search_ready(
                    input_mode,
                    search_report,
                    visual_report,
                    retrieval_plan,
                    controller_report,
                )

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
            head_payload = [
                {"name": result.name, "title": result.title, "output": result.output}
                for result in head_results
            ]
            self._record_trajectory(
                trajectory_run_id,
                stage="heads",
                name="head_outputs_ready",
                payload={
                    "head_names": head_names,
                    "head_results": head_payload,
                    "formatted_output": formatted_output,
                },
            )
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
                self._record_trajectory(
                    trajectory_run_id,
                    stage="memory",
                    name="analysis_saved",
                    payload={"saved": True},
                )
            output_payload = {
                "input_mode": input_mode,
                "head_names": head_names,
                "head_results": head_payload,
                "formatted_output": formatted_output,
                "combined_context": combined_context,
                "memory_report": memory_report,
                "visual_report": visual_report,
                "retrieval_plan": retrieval_plan,
                "search_report": search_report,
                "controller_report": controller_report,
            }
            self._finish_trajectory(trajectory_run_id, output=output_payload)
            self._emit_progress(progress, "heads", "Task heads ready.")
            return MultiHeadWorkflowResult(
                head_results=head_results,
                formatted_output=formatted_output,
                search_report=search_report,
                combined_context=combined_context,
                visual_report=visual_report,
                retrieval_plan=retrieval_plan,
                controller_report=controller_report,
                memory_report=memory_report,
                input_mode=input_mode,
                trajectory_run_id=trajectory_run_id,
            )
        except Exception as exc:
            self._fail_trajectory(
                trajectory_run_id,
                error=exc,
                output={
                    "input_mode": input_mode,
                    "combined_context": combined_context,
                    "formatted_output": formatted_output,
                    "memory_report": memory_report,
                    "visual_report": visual_report,
                    "retrieval_plan": retrieval_plan,
                    "search_report": search_report,
                    "controller_report": controller_report,
                },
            )
            raise
