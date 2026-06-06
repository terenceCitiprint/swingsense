"""Pretty terminal rendering of an analysis using rich."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_RELATION_STYLE = {
    "supports": "green",
    "contradicts": "red",
    "neutral": "yellow",
}
_CONFIDENCE_STYLE = {"low": "red", "medium": "yellow", "high": "green"}


def render_analysis(analysis: dict, swing_id: int | None = None) -> None:
    title = "SwingSense analysis"
    if swing_id is not None:
        title += f" — swing #{swing_id}"
    confidence = analysis.get("confidence", "unknown")
    conf_style = _CONFIDENCE_STYLE.get(confidence, "white")

    console.print(
        Panel(
            analysis.get("summary", "(no summary)"),
            title=title,
            subtitle=f"[{conf_style}]confidence: {confidence}[/{conf_style}]",
            border_style="cyan",
        )
    )

    translations = analysis.get("feel_translation", [])
    if translations:
        t = Table(title="Feel → candidate mechanics", show_lines=False, expand=True)
        t.add_column("Body region", style="magenta", no_wrap=True)
        t.add_column("Candidate mechanic", style="bold")
        t.add_column("Why")
        for item in translations:
            t.add_row(
                item.get("body_region", ""),
                item.get("mechanic", ""),
                item.get("explanation", ""),
            )
        console.print(t)

    refs = analysis.get("physics_cross_reference", [])
    if refs:
        t = Table(title="Physics cross-reference", expand=True)
        t.add_column("Principle", style="bold cyan")
        t.add_column("Relation")
        t.add_column("Note")
        for item in refs:
            rel = item.get("relation", "neutral")
            style = _RELATION_STYLE.get(rel, "white")
            t.add_row(
                item.get("principle", ""),
                f"[{style}]{rel}[/{style}]",
                item.get("note", ""),
            )
        console.print(t)

    contradictions = analysis.get("contradictions", [])
    if contradictions:
        body = "\n".join(f"• {c}" for c in contradictions)
        console.print(
            Panel(body, title="⚠ Feel-vs-real to watch", border_style="red")
        )

    suggestions = analysis.get("suggestions", [])
    if suggestions:
        t = Table(title="Suggestions to test", expand=True)
        t.add_column("Try this", style="bold green")
        t.add_column("Rationale")
        t.add_column("Source", style="dim")
        for item in suggestions:
            t.add_row(
                item.get("cue_or_drill", ""),
                item.get("rationale", ""),
                item.get("source", ""),
            )
        console.print(t)

    measure = analysis.get("what_to_measure_next", [])
    if measure:
        body = "\n".join(f"• {m}" for m in measure)
        console.print(
            Panel(body, title="📹 Capture next to confirm", border_style="blue")
        )
