"""Pretty terminal rendering of an analysis using rich."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def render_features(features: dict) -> None:
    """Show the measured swing features from the video pipeline."""
    conf = features.get("confidence", "unknown")
    conf_style = _CONFIDENCE_STYLE.get(conf, "white")
    events = features.get("events", {})
    metrics = features.get("metrics", {})

    reliable = events.get("reliable")
    rel_txt = (
        "[green]events reliable[/green]"
        if reliable
        else "[red]events UNRELIABLE[/red]"
    )
    console.print(
        Panel(
            f"Measured from video — [{conf_style}]confidence: {conf}[/{conf_style}]"
            f"  ·  {rel_txt}",
            title="📐 Video features (2D single-camera proxy)",
            border_style="blue",
        )
    )

    if events:
        t = Table(title="Swing events", expand=True)
        t.add_column("Event", style="bold")
        t.add_column("Frame")
        t.add_column("Time (s)")
        for name in (
            "address",
            "top",
            "transition",
            "impact",
            "follow_through",
            "finish",
        ):
            ev = events.get(name)
            if isinstance(ev, dict):
                t.add_row(name, str(ev.get("frame", "")), str(ev.get("t", "")))
        console.print(t)

    tempo = metrics.get("tempo", {})
    rot = metrics.get("rotation", {})
    if tempo or rot:
        t = Table(title="Key metrics", expand=True)
        t.add_column("Metric", style="cyan")
        t.add_column("Value")
        if tempo:
            flag = "" if tempo.get("reliable", True) else " [red](approx)[/red]"
            t.add_row(
                "Tempo (back:down)",
                f"{tempo.get('ratio_back_to_down')}:1  "
                f"({tempo.get('backswing_s')}s / {tempo.get('downswing_s')}s){flag}",
            )
        for label in ("address", "top", "impact"):
            r = rot.get(label)
            if r:
                t.add_row(
                    f"Separation @ {label}", f"{r.get('separation_deg')}° (proxy)"
                )
        energy = metrics.get("energy", {})
        if energy:
            t.add_row(
                "Energy (pendulum)",
                f"rise {energy.get('hand_rise')} · pause {energy.get('pause_at_top_s')}s "
                f"· whip peak {energy.get('peak_hand_speed')}",
            )
        bal = metrics.get("finish_balance", {})
        if bal.get("held_still") is not None:
            held = "[green]held still[/green]" if bal["held_still"] else "[yellow]moved[/yellow]"
            t.add_row(
                "Finish balance",
                f"{held} over {bal.get('hold_window_s')}s "
                f"(ankle drift {bal.get('ankle_drift_pct')}%)",
            )
        if "hand_visibility" in metrics:
            t.add_row("Hand tracking", f"{metrics['hand_visibility']} (0-1)")
        console.print(t)

    notes = features.get("notes", [])
    if notes:
        body = "\n".join(f"• {n}" for n in notes)
        console.print(Panel(body, title="Caveats", border_style="yellow"))


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
