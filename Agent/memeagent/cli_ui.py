from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.rule import Rule
except ImportError:  # pragma: no cover - exercised in environments without rich.
    Console = None
    Live = None
    Markdown = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    TimeElapsedColumn = None
    Rule = None


@dataclass(frozen=True)
class RunSummary:
    model: str
    image_count: int
    timeout: float
    search_enabled: bool
    search_provider: str
    web_results: int
    news_results: int
    search_timeout: float


class MemeAgentCLI:
    """Terminal presentation helpers with a plain-text fallback."""

    def __init__(
        self,
        enabled: bool = True,
        stream_markdown: bool = False,
        trace_mode: str = "live",
    ) -> None:
        self._configure_text_streams()
        self.enabled = enabled and Console is not None
        self.stream_markdown = stream_markdown and self.enabled
        self.trace_mode = trace_mode
        self.console = Console() if self.enabled else None
        self._progress: Any = None
        self._live: Any = None
        self._task_id: Any = None
        self._stream_live: Any = None
        self._stream_text = ""
        self._activity_started_at = 0.0
        self._activity_events: list[tuple[str, str, float]] = []

    def _configure_text_streams(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if not callable(reconfigure):
                continue
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue

    def print_start(self, summary: RunSummary) -> None:
        if not self.enabled:
            print(
                f"Starting MemeAgent request with model={summary.model}, "
                f"images={summary.image_count}, timeout={summary.timeout}s..."
            )
            if summary.search_enabled:
                print(
                    f"Web search enabled with provider={summary.search_provider}: "
                    f"up to {summary.web_results} web results, "
                    f"{summary.news_results} news results, "
                    f"search timeout={summary.search_timeout}s."
                )
            return

        lines = [
            f"Model: [bold]{summary.model}[/bold]",
            f"Images: [bold]{summary.image_count}[/bold]",
            f"LLM timeout: [bold]{summary.timeout}s[/bold]",
        ]
        if summary.search_enabled:
            lines.extend(
                [
                    "",
                    f"Search provider: [bold]{summary.search_provider}[/bold]",
                    (
                        f"Result limits: [bold]{summary.web_results}[/bold] web, "
                        f"[bold]{summary.news_results}[/bold] news"
                    ),
                    f"Search timeout: [bold]{summary.search_timeout}s[/bold]",
                ]
            )
        else:
            lines.append("Search: [bold]disabled[/bold]")

        self.console.print(
            Panel.fit(
                "\n".join(lines),
                title="MemeAgent CLI",
                border_style="cyan",
            )
        )

    def start_activity(self) -> None:
        if not self.enabled or self.trace_mode == "off":
            return

        self._activity_started_at = time.monotonic()
        self._activity_events = []
        self._record_activity_event("start", "Preparing request...")
        self._live = Live(
            self._render_activity_panel(),
            console=self.console,
            refresh_per_second=6,
            transient=False,
        )
        self._live.start()

    def update(self, stage: str, message: str) -> None:
        label = f"[{stage}] {message}"
        if not self.enabled:
            print(label)
            return

        self._record_activity_event(stage, message)
        if self._live is not None:
            self._live.update(self._render_activity_panel())

    def _record_activity_event(self, stage: str, message: str) -> None:
        elapsed = max(0.0, time.monotonic() - self._activity_started_at)
        self._activity_events.append((stage, message, elapsed))
        self._activity_events = self._activity_events[-12:]

    def _render_activity_panel(self) -> Any:
        if not self._activity_events:
            body = "Preparing request..."
            return Panel(body, title="MemeAgent Workflow", border_style="cyan")

        stage_styles = {
            "start": "cyan",
            "input": "cyan",
            "memory": "blue",
            "vision": "magenta",
            "planning": "yellow",
            "controller": "yellow",
            "search": "green",
            "reflection": "yellow",
            "analysis": "green",
            "heads": "green",
        }
        current_stage, current_message, current_elapsed = self._activity_events[-1]
        lines = [
            f"Elapsed: [bold]{current_elapsed:5.1f}s[/bold]",
            (
                "Current: "
                f"[bold {stage_styles.get(current_stage, 'white')}]{current_stage}[/] "
                f"{current_message}"
            ),
            "",
            "[bold]Workflow timeline[/bold]",
        ]
        for index, (stage, message, elapsed) in enumerate(self._activity_events, start=1):
            style = stage_styles.get(stage, "white")
            lines.append(
                f"{index:02d}. [dim]{elapsed:5.1f}s[/dim] "
                f"[{style}]{stage:<10}[/{style}] {message}"
            )
        return Panel(
            "\n".join(lines),
            title="MemeAgent Workflow",
            border_style="cyan",
        )

    def stop_activity(self) -> None:
        if self._live is not None:
            self._live.stop()
            if self.trace_mode == "live" and self.console is not None:
                self.console.clear()
        self._live = None
        self._progress = None
        self._task_id = None
        self._activity_events = []

    def start_stream(self) -> None:
        self._stream_text = ""
        if self.stream_markdown:
            self._stream_live = Live(
                Panel(
                    Markdown(""),
                    title="Final Analysis",
                    border_style="green",
                ),
                console=self.console,
                refresh_per_second=4,
            )
            self._stream_live.start()
            return

        if self.enabled:
            self.console.print(Rule("Final Analysis", style="green"))
            return

        print("\n=== Final Analysis ===")

    def stream_delta(self, text: str) -> None:
        self._stream_text += text
        if not self.enabled:
            print(text, end="", flush=True)
            return

        if self.stream_markdown and self._stream_live is not None:
            self._stream_live.update(
                Panel(
                    Markdown(self._stream_text),
                    title="Final Analysis",
                    border_style="green",
                )
            )
            return

        self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def stop_stream(self) -> None:
        if self._stream_live is not None:
            self._stream_live.stop()
            self._stream_live = None
            return

        print()

    def print_result(
        self,
        analysis: str,
        input_mode: str,
        search_report: str = "",
        visual_report: str = "",
        retrieval_plan: str = "",
        controller_report: str = "",
        show_search: bool = False,
        analysis_title: str = "Final Analysis",
    ) -> None:
        if not self.enabled:
            print(f"Input mode: {input_mode}")
            if show_search and search_report:
                if visual_report:
                    print("\n=== Image-Derived Search Context ===")
                    print(visual_report)
                if retrieval_plan:
                    print("\n=== Supplemental Retrieval Plan ===")
                    print(retrieval_plan)
                if controller_report:
                    print("\n=== Controller Planning Report ===")
                    print(controller_report)
                print("\n=== Search Report ===")
                print(search_report)
                if analysis:
                    print(f"\n=== {analysis_title} ===")
            if analysis:
                print(analysis)
            return

        self.console.print(Rule(f"Input Mode: {input_mode}", style="cyan"))
        if show_search and search_report:
            if visual_report:
                self.console.print(
                    Panel(
                        Markdown(visual_report),
                        title="Image-Derived Search Context",
                        border_style="blue",
                    )
                )
            if retrieval_plan:
                self.console.print(
                    Panel(
                        Markdown(retrieval_plan),
                        title="Supplemental Retrieval Plan",
                        border_style="yellow",
                    )
                )
            if controller_report:
                self.console.print(
                    Panel(
                        Markdown(controller_report),
                        title="Controller Planning Report",
                        border_style="yellow",
                    )
                )
            self.console.print(
                Panel(
                    Markdown(search_report),
                    title="Search Report",
                    border_style="magenta",
                )
            )

        if analysis:
            self.console.print(
                Panel(
                    Markdown(analysis),
                    title=analysis_title,
                    border_style="green",
                )
            )

    def print_error(self, message: str) -> None:
        if self.enabled:
            self.console.print(f"[bold red]{message}[/bold red]")
        else:
            print(message)
