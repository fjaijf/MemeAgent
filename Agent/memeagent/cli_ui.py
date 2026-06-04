from __future__ import annotations

from dataclasses import dataclass
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

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and Console is not None
        self.console = Console() if self.enabled else None
        self._progress: Any = None
        self._live: Any = None
        self._task_id: Any = None

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
        if not self.enabled:
            return

        self._progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            TimeElapsedColumn(),
            transient=False,
        )
        self._task_id = self._progress.add_task("Preparing request...", total=None)
        self._live = Live(self._progress, console=self.console, refresh_per_second=8)
        self._live.start()

    def update(self, stage: str, message: str) -> None:
        label = f"[{stage}] {message}"
        if not self.enabled:
            print(label)
            return

        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=label)

    def stop_activity(self) -> None:
        if self._live is not None:
            self._live.stop()
        self._live = None
        self._progress = None
        self._task_id = None

    def print_result(
        self,
        analysis: str,
        input_mode: str,
        search_report: str = "",
        visual_report: str = "",
        show_search: bool = False,
    ) -> None:
        if not self.enabled:
            print(f"Input mode: {input_mode}")
            if show_search and search_report:
                if visual_report:
                    print("\n=== Image-Derived Search Context ===")
                    print(visual_report)
                print("\n=== Search Report ===")
                print(search_report)
                print("\n=== Final Analysis ===")
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
            self.console.print(
                Panel(
                    Markdown(search_report),
                    title="Search Report",
                    border_style="magenta",
                )
            )

        self.console.print(
            Panel(
                Markdown(analysis),
                title="Final Analysis",
                border_style="green",
            )
        )

    def print_error(self, message: str) -> None:
        if self.enabled:
            self.console.print(f"[bold red]{message}[/bold red]")
        else:
            print(message)
