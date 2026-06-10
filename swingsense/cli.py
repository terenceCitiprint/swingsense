"""SwingSense command-line interface (Phase 0: text-only feel translator)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from pathlib import Path

from . import __version__, config, db
from .engine import EngineError, analyze_feel
from .knowledge import load_knowledge
from .prompts import build_user_prompt
from .render import render_analysis, render_features

app = typer.Typer(
    help="Bridge what your golf swing FEELS like and what's actually happening.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def feel(
    description: str = typer.Argument(..., help="How the swing felt, in plain words."),
    club: str | None = typer.Option(None, "--club", "-c", help="e.g. driver, 7i."),
    tag: list[str] = typer.Option([], "--tag", "-t", help="Repeatable tag."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the assembled prompt; do not call the API."
    ),
    no_history: bool = typer.Option(
        False, "--no-history", help="Don't include recent swings as context."
    ),
):
    """Translate a feel into candidate mechanics, cross-referenced with physics."""
    kb = load_knowledge()
    history = None if no_history else db.list_swings(limit=5, club=club)

    if dry_run:
        console.print(build_user_prompt(description, kb, club, history))
        raise typer.Exit()

    try:
        analysis = analyze_feel(description, kb, club=club, history=history)
    except EngineError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    swing_id = db.add_swing(
        feel=description, analysis=analysis, club=club, tags=list(tag)
    )
    render_analysis(analysis, swing_id=swing_id)
    console.print(f"\n[dim]Saved as swing #{swing_id}. View later: "
                  f"swingsense show {swing_id}[/dim]")


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to a swing video (mp4/mov)."),
    feel: str | None = typer.Option(
        None, "--feel", "-f", help="How the swing felt (optional)."
    ),
    club: str | None = typer.Option(None, "--club", "-c", help="e.g. driver, 7i."),
    tag: list[str] = typer.Option([], "--tag", "-t", help="Repeatable tag."),
    model: str = typer.Option(
        "full", "--model", help="Pose model: lite | full | heavy."
    ),
    features_only: bool = typer.Option(
        False, "--features-only", help="Extract video features; skip the LLM."
    ),
    report: str | None = typer.Option(
        None, "--report", "-r",
        help="Also write a visual HTML report (panorama, charts, 3D loop) here.",
    ),
):
    """Extract biomechanics from a swing video and reason over them with your feel."""
    path = Path(video).expanduser()
    if not path.exists():
        console.print(f"[red]No such video:[/red] {path}")
        raise typer.Exit(code=1)

    # Heavy CV deps are imported lazily so the rest of the CLI stays light.
    try:
        from .vision.features import compute_features
        from .vision.pose import extract_pose
    except ImportError as exc:
        console.print(
            f"[red]Video support needs extra deps (opencv, mediapipe):[/red] {exc}"
        )
        raise typer.Exit(code=1)

    with console.status(f"Extracting pose from {path.name} ..."):
        track = extract_pose(str(path), model_variant=model)
        feats = compute_features(track).to_dict()

    render_features(feats)

    def _write_report(analysis_dict: dict | None) -> None:
        if not report:
            return
        from .report import build_report

        with console.status("Rendering visual report ..."):
            out = build_report(track, feats, analysis_dict, str(path), report)
        console.print(f"[green]Report written:[/green] {out}")

    if features_only:
        swing_id = db.add_swing(
            feel=feel or "(video only)",
            analysis={"_features": feats},
            club=club,
            video_path=str(path),
            tags=list(tag),
        )
        console.print(f"[dim]Saved measurements as swing #{swing_id}.[/dim]")
        _write_report(None)
        return

    kb = load_knowledge()
    history_ctx = db.list_swings(limit=5, club=club)
    try:
        analysis = analyze_feel(
            feel or "", kb, club=club, history=history_ctx, features=feats
        )
    except EngineError as exc:
        console.print(f"\n[yellow]Skipping reasoning step:[/yellow] {exc}")
        console.print("[dim]Measured features above are still valid.[/dim]")
        _write_report(None)
        raise typer.Exit(code=1)

    analysis["_features"] = feats  # persist measurements alongside the reasoning
    swing_id = db.add_swing(
        feel=feel or "(video only)",
        analysis=analysis,
        club=club,
        video_path=str(path),
        tags=list(tag),
    )
    console.print()
    render_analysis(analysis, swing_id=swing_id)
    console.print(f"\n[dim]Saved as swing #{swing_id}.[/dim]")
    _write_report(analysis)


@app.command()
def history(
    last: int = typer.Option(10, "--last", "-n", help="How many to show."),
    club: str | None = typer.Option(None, "--club", "-c", help="Filter by club."),
):
    """List recent swings and their one-line reads."""
    swings = db.list_swings(limit=last, club=club)
    if not swings:
        console.print("[yellow]No swings logged yet. Try: swingsense feel \"...\"[/yellow]")
        return
    t = Table(title="Swing history")
    t.add_column("#", style="bold")
    t.add_column("When (UTC)", style="dim")
    t.add_column("Club")
    t.add_column("Feel")
    t.add_column("Read")
    for s in swings:
        t.add_row(
            str(s.id),
            s.created_at[:19].replace("T", " "),
            s.club or "—",
            s.feel,
            s.analysis.get("summary", ""),
        )
    console.print(t)


@app.command()
def show(swing_id: int = typer.Argument(..., help="Swing id from `history`.")):
    """Re-render the full analysis for a logged swing."""
    swing = db.get_swing(swing_id)
    if not swing:
        console.print(f"[red]No swing #{swing_id}.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold]Feel:[/bold] {swing.feel}")
    if swing.club:
        console.print(f"[bold]Club:[/bold] {swing.club}")
    if swing.video_path:
        console.print(f"[bold]Video:[/bold] {swing.video_path}")
    feats = swing.analysis.get("_features")
    if feats:
        render_features(feats)
        console.print()
    render_analysis(swing.analysis, swing_id=swing.id)


@app.command()
def kb():
    """List the loaded knowledge base (physics + coaching)."""
    entries = load_knowledge()
    if not entries:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return
    t = Table(title="Knowledge base")
    t.add_column("Category", style="cyan")
    t.add_column("Name", style="bold")
    t.add_column("Summary")
    t.add_column("Source", style="dim")
    for e in entries:
        t.add_row(e.category, e.name, e.summary.strip(), e.source)
    console.print(t)


@app.command()
def compare(
    video_a: str = typer.Argument(..., help="First swing video (the 'before')."),
    video_b: str | None = typer.Argument(
        None, help="Second swing video (the 'after'). Omit when using --pro."),
    out: str = typer.Option("compare.html", "--out", "-o", help="Report path."),
    label_a: str = typer.Option("before", help="Label for the first swing."),
    label_b: str = typer.Option("after", help="Label for the second swing."),
    pro: str | None = typer.Option(
        None, "--pro", help="Compare against a stored reference "
        "(see `swingsense pro list`) instead of a second video."),
    model: str = typer.Option("full", "--model", help="Pose model variant."),
):
    """Compare two swings, or one swing against a stored pro reference."""
    try:
        from .vision.features import compute_features
        from .vision.pose import extract_pose
    except ImportError as exc:
        console.print(f"[red]Video support needs extra deps:[/red] {exc}")
        raise typer.Exit(code=1)

    path_a = Path(video_a).expanduser()
    if not path_a.exists():
        console.print(f"[red]No such video:[/red] {path_a}")
        raise typer.Exit(code=1)

    if pro:
        from .pro import build_pro_comparison, load_reference

        ref = load_reference(pro)
        if not ref:
            console.print(f"[red]No reference named '{pro}'.[/red] "
                          "See: swingsense pro list")
            raise typer.Exit(code=1)
        with console.status(f"Extracting pose from {path_a.name} ..."):
            t = extract_pose(str(path_a), model_variant=model)
            f = compute_features(t).to_dict()
        with console.status("Rendering comparison vs reference ..."):
            result = build_pro_comparison(t, f, str(path_a), ref, out,
                                          label=label_a)
        console.print(f"[green]Comparison written:[/green] {result}")
        return

    if not video_b:
        console.print("[red]Provide a second video, or use --pro NAME.[/red]")
        raise typer.Exit(code=1)
    paths = [path_a, Path(video_b).expanduser()]
    if not paths[1].exists():
        console.print(f"[red]No such video:[/red] {paths[1]}")
        raise typer.Exit(code=1)

    tracks, feats = [], []
    for p in paths:
        with console.status(f"Extracting pose from {p.name} ..."):
            t = extract_pose(str(p), model_variant=model)
            tracks.append(t)
            feats.append(compute_features(t).to_dict())

    from .compare import build_comparison

    with console.status("Rendering comparison ..."):
        result = build_comparison(
            tracks[0], feats[0], str(paths[0]),
            tracks[1], feats[1], str(paths[1]),
            out, labelA=label_a, labelB=label_b,
        )
    console.print(f"[green]Comparison written:[/green] {result}")


pro_app = typer.Typer(help="Manage the pro/reference swing library.",
                      no_args_is_help=True)
app.add_typer(pro_app, name="pro")


@pro_app.command("add")
def pro_add(
    video: str = typer.Argument(..., help="Reference swing video you have "
                                "rights to use."),
    name: str = typer.Option(..., "--name", "-n",
                             help="Reference name, e.g. 'tour_dtl_driver'."),
    source: str = typer.Option("", "--source", help="Where the clip is from."),
    model: str = typer.Option("full", "--model", help="Pose model variant."),
):
    """Ingest a reference swing into the local library (stores derived
    signals only, not the video)."""
    path = Path(video).expanduser()
    if not path.exists():
        console.print(f"[red]No such video:[/red] {path}")
        raise typer.Exit(code=1)
    try:
        from .vision.features import compute_features
        from .vision.pose import extract_pose
    except ImportError as exc:
        console.print(f"[red]Video support needs extra deps:[/red] {exc}")
        raise typer.Exit(code=1)
    from .pro import add_reference

    with console.status(f"Extracting pose from {path.name} ..."):
        track = extract_pose(str(path), model_variant=model)
        feats = compute_features(track).to_dict()
    if not feats["events"].get("reliable"):
        console.print("[yellow]Warning: events unreliable on this clip — the "
                      "stored reference will be low quality.[/yellow]")
    out = add_reference(track, feats, name, source=source)
    console.print(f"[green]Reference saved:[/green] {out}")


@pro_app.command("list")
def pro_list():
    """List stored references."""
    from .pro import list_references

    refs = list_references()
    if not refs:
        console.print("[yellow]No references yet. Add one: "
                      "swingsense pro add <video> --name <name>[/yellow]")
        return
    t = Table(title="Reference library")
    t.add_column("Name", style="bold")
    t.add_column("Tempo")
    t.add_column("Order")
    t.add_column("Conf")
    t.add_column("Source", style="dim")
    for r in refs:
        t.add_row(r["name"], str(r.get("tempo_ratio")),
                  " → ".join(r.get("firing_order", [])),
                  str(r.get("confidence")), r.get("source", ""))
    console.print(t)


@app.command()
def trends(
    out: str = typer.Option("trends.html", "--out", "-o", help="Report path."),
    last: int = typer.Option(100, "--last", "-n", help="How many swings back."),
    club: str | None = typer.Option(None, "--club", "-c", help="Filter by club."),
):
    """Chart tempo and separation across your logged video swings."""
    from .trends import build_trends

    result = build_trends(out, limit=last, club=club)
    if result is None:
        console.print(
            "[yellow]No video swings with measurements in history yet. "
            "Run: swingsense analyze <video> first.[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]Trends written:[/green] {result}")


@app.command()
def version():
    """Show version and active configuration."""
    console.print(f"SwingSense {__version__}")
    console.print(f"Model: {config.model()}")
    console.print(f"Data home: {config.data_home()}")
    console.print(
        "API key: "
        + ("[green]set[/green]" if config.api_key() else "[red]not set[/red]")
    )


if __name__ == "__main__":
    app()
