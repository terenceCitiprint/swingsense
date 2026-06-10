"""Self-contained HTML swing report: simple read on top, visuals for nerds below.

Layout thesis: the panoramic strip is the hero — proof that a phone video became
measurement. Tier 1 ("The read") is plain language from the reasoning engine.
Tier 2 ("For nerds") is visual: kinematic sequence chart, hand-speed profile,
hip–shoulder separation curve, and an interactive 3D loop of the frames where
the improvement lives.

Charts use Plotly (loaded from CDN, so the file stays small; open with internet
the first time and it caches).
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np

from .vision.kinematics import Kinematics, separation_series, sequence_read
from .vision.pose import PoseTrack

_SEG_COLORS = {
    "pelvis": "#2E6B46",   # fairway green — the engine
    "thorax": "#4178B4",   # steel blue
    "arm": "#B07A1E",      # amber
    "hands": "#E8590C",    # signal orange — the payload
}

_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def _sequence_fig(kin: Kinematics, events: dict):
    import plotly.graph_objects as go

    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    a = events["address"]["frame"]
    lo = max(0, a - 5)
    hi = min(len(kin.t) - 1, imp + int((imp - top) * 1.5) + 5)

    fig = go.Figure()
    series = {
        "pelvis": kin.hip_ang_vel,
        "thorax": kin.shoulder_ang_vel,
        "arm": kin.arm_ang_vel,
    }
    for name, sig in series.items():
        fig.add_trace(go.Scatter(
            x=kin.t[lo:hi + 1], y=sig[lo:hi + 1], name=name,
            line=dict(color=_SEG_COLORS[name], width=2.5),
        ))
    # Hands on a secondary axis (different units: speed, not deg/s).
    fig.add_trace(go.Scatter(
        x=kin.t[lo:hi + 1], y=kin.hand_speed[lo:hi + 1], name="hands (speed)",
        yaxis="y2", line=dict(color=_SEG_COLORS["hands"], width=2.5, dash="dot"),
    ))
    fig.add_vrect(x0=kin.t[top], x1=kin.t[imp],
                  fillcolor="#E8590C", opacity=0.06, line_width=0)
    for name, fr in (("top", top), ("impact", imp)):
        fig.add_vline(x=kin.t[fr], line=dict(color="#888", dash="dash", width=1),
                      annotation_text=name, annotation_position="top")
    fig.update_layout(
        title="Kinematic sequence — who fires when",
        xaxis_title="time (s)",
        yaxis=dict(title="segment angular speed (deg/s)"),
        yaxis2=dict(title="hand speed (proxy)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", y=1.12),
        template="plotly_white", height=420, margin=dict(t=80, b=40),
    )
    return fig


def _hand_speed_fig(kin: Kinematics, events: dict):
    import plotly.graph_objects as go

    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    lo, hi = max(0, top - 3), min(len(kin.t) - 1, imp + 5)
    seg = kin.hand_speed[lo:hi + 1]
    peak = lo + int(np.argmax(seg)) if seg.size else imp

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kin.t[lo:hi + 1], y=seg, mode="lines",
                             line=dict(color=_SEG_COLORS["hands"], width=3),
                             name="hand speed"))
    fig.add_vline(x=kin.t[imp], line=dict(color="#888", dash="dash", width=1),
                  annotation_text="impact")
    fig.add_trace(go.Scatter(x=[kin.t[peak]], y=[float(kin.hand_speed[peak])],
                             mode="markers+text", text=["peak"],
                             textposition="top center",
                             marker=dict(size=10, color="#14211A"),
                             showlegend=False))
    early = (imp - peak) / max(imp - top, 1)
    note = ("peak well before impact → energy spent early (cast tendency)"
            if early > 0.45 else
            "peak near impact → release timing looks efficient")
    fig.update_layout(
        title=f"Hand-speed profile through the downswing — {note}",
        xaxis_title="time (s)", yaxis_title="speed (frame-heights/s, proxy)",
        template="plotly_white", height=360, margin=dict(t=70, b=40),
        showlegend=False,
    )
    return fig


def _separation_fig(track: PoseTrack, kin: Kinematics, events: dict):
    import plotly.graph_objects as go

    sep = np.abs(separation_series(track))
    a = events["address"]["frame"]
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    lo, hi = max(0, a - 5), min(len(kin.t) - 1, imp + 8)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kin.t[lo:hi + 1], y=sep[lo:hi + 1], mode="lines",
                             line=dict(color=_SEG_COLORS["pelvis"], width=3)))
    for name, fr in (("top", top), ("impact", imp)):
        fig.add_vline(x=kin.t[fr], line=dict(color="#888", dash="dash", width=1),
                      annotation_text=name)
    fig.update_layout(
        title="Hip–shoulder separation (2D proxy) — the stored coil",
        xaxis_title="time (s)", yaxis_title="separation (deg, image-plane)",
        template="plotly_white", height=360, margin=dict(t=70, b=40),
    )
    return fig


_JOINT_GROUPS = [
    # (legend label, landmark indices, marker color, size)
    ("Head", [0], "#FF8A3D", 11),
    ("Upper torso", [11, 12, 13, 14, 15, 16], "#E8590C", 7),
    ("Pelvis", [23, 24], "#D9500B", 8),
    ("Stance", [25, 26, 27, 28], "#C2470A", 7),
]
_BONE = "#C9B18C"   # tan skeleton lines, as in mocap-style overlays
_GHOST = "#8C949B"


def _loop3d_fig(track: PoseTrack, events: dict, window_s: float = 1.0,
                ghost=None):
    """Interactive 3D skeleton animation around the transition (top of swing).

    Styled like a mocap overlay: tan bones, orange joints with white rings,
    joint groups labeled in the legend (Head / Upper torso / Pelvis / Stance),
    on a dark scene with the hand path traced through the window. If `ghost`
    (a correction.Ghost with changed=True) is given, a grey skeleton shows the
    re-timed corrected motion at the same instant.
    """
    import plotly.graph_objects as go

    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    fps = track.fps
    lo = max(0, top - int(fps * window_s * 0.6))
    hi = min(track.n_frames - 1, max(imp, top + int(fps * window_s * 0.6)))
    step = max(1, (hi - lo) // 40)  # cap ~40 animation frames
    idxs = list(range(lo, hi + 1, step))

    def xyz(lm, i):
        return lm[i, 0], lm[i, 2], 1 - lm[i, 1]  # depth on Y, image-y flipped up

    def bones_trace(lm, color, width, name=None, dash=None):
        xs, ys, zs = [], [], []
        for a, b in _CONNECTIONS:
            pa, pb = xyz(lm, a), xyz(lm, b)
            xs += [pa[0], pb[0], None]
            ys += [pa[1], pb[1], None]
            zs += [pa[2], pb[2], None]
        line = dict(color=color, width=width)
        if dash:
            line["dash"] = dash
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line,
                            name=name, showlegend=bool(name),
                            hoverinfo="skip")

    def joints_traces(lm):
        out = []
        for label, joints, color, size in _JOINT_GROUPS:
            pts = [xyz(lm, j) for j in joints]
            out.append(go.Scatter3d(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                z=[p[2] for p in pts], mode="markers", name=label,
                marker=dict(size=size, color=color,
                            line=dict(color="#F2F2F2", width=2)),
            ))
        return out

    show_ghost = ghost is not None and getattr(ghost, "changed", False)

    def frame_data(fi):
        data = [bones_trace(track.landmarks[fi], _BONE, 8)]
        data += joints_traces(track.landmarks[fi])
        if show_ghost:
            data.append(bones_trace(ghost.landmarks[fi], _GHOST, 5,
                                    name="ghost (re-timed)", dash="dot"))
        return data

    frames = [go.Frame(data=frame_data(fi), name=f"{fi / fps:.2f}s")
              for fi in idxs]

    # Static hand path through the window — appended AFTER the animated traces
    # so frame updates (which replace traces by index) never touch it.
    wrist = (track.landmarks[lo:hi + 1, 15, :] +
             track.landmarks[lo:hi + 1, 16, :]) / 2.0
    path_trace = go.Scatter3d(
        x=wrist[:, 0], y=wrist[:, 2], z=1 - wrist[:, 1], mode="lines",
        line=dict(color="#FFC845", width=4), name="hand path",
        opacity=0.65,
    )

    title = ("3D loop — transition window (rotate me; depth is MediaPipe's "
             "estimate, illustrative not measured)")
    if show_ghost:
        title = ("3D loop — joints labeled; grey ghost = your motion re-timed "
                 "to fire in order. Depth illustrative.")
    fig = go.Figure(data=frame_data(idxs[0]) + [path_trace], frames=frames)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E8E8E4", size=15)),
        paper_bgcolor="#10150F",
        legend=dict(font=dict(color="#D8DCD4"), orientation="h", y=0.02,
                    bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="data",
            bgcolor="#10150F",
            camera=dict(eye=dict(x=0.0, y=-2.2, z=0.15)),  # face-on default
        ),
        height=560, margin=dict(t=70, b=10),
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0,
            font=dict(color="#14211A"), bgcolor="#E8E8E4",
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False))]),
            ],
        )],
        sliders=[dict(
            font=dict(color="#D8DCD4"),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True))])
                   for f in frames],
            y=-0.02, len=0.9,
        )],
    )
    return fig


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def _esc(s) -> str:
    return html.escape(str(s or ""))


def _tier1_html(analysis: dict | None, features: dict, seq: dict,
                norms: list[str] | None = None) -> str:
    """Plain-language read: tendency, simple changes, drills."""
    parts = []
    if analysis:
        parts.append(f"<p class='summary'>{_esc(analysis.get('summary'))}</p>")
        contras = analysis.get("contradictions") or []
        if contras:
            items = "".join(f"<li>{_esc(c)}</li>" for c in contras)
            parts.append(f"<div class='block warn'><h3>Feel vs real — watch for"
                         f"</h3><ul>{items}</ul></div>")
        suggs = analysis.get("suggestions") or []
        if suggs:
            rows = "".join(
                f"<li><strong>{_esc(s.get('cue_or_drill'))}</strong> — "
                f"{_esc(s.get('rationale'))} "
                f"<span class='src'>({_esc(s.get('source'))})</span></li>"
                for s in suggs)
            parts.append(f"<div class='block'><h3>Simple changes & drills</h3>"
                         f"<ul>{rows}</ul></div>")
    else:
        parts.append(
            "<p class='summary'>Measured-only report (no reasoning call). "
            "The tendency read below comes straight from the numbers.</p>")

    order = " → ".join(seq["order"])
    verdict = seq["verdict"]
    tempo = features.get("metrics", {}).get("tempo", {})
    parts.append(
        f"<div class='block'><h3>Tendency from the measurements</h3><ul>"
        f"<li>Downswing firing order: <strong>{_esc(order)}</strong> "
        f"(ideal: pelvis → thorax → arm → hands) — <strong>{_esc(verdict)}"
        f"</strong></li>"
        f"<li>Tempo {_esc(tempo.get('ratio_back_to_down'))}:1 "
        f"(classic efficient swings sit near 3:1)</li></ul></div>")
    if norms:
        items = "".join(f"<li>{_esc(n)}</li>" for n in norms)
        parts.append(
            f"<div class='block'><h3>vs tour norms (published timing bands)"
            f"</h3><ul>{items}</ul></div>")
    return "\n".join(parts)


def build_report(
    track: PoseTrack,
    features: dict,
    analysis: dict | None,
    video_path: str,
    out_path: str,
    panorama_path: str | None = None,
) -> str:
    """Write the HTML report; returns out_path."""
    from .vision.correction import build_ghost
    from .vision.kinematics import compute_kinematics
    from .vision.visuals import panorama_strip

    kin = compute_kinematics(track)
    events = features["events"]
    seq = sequence_read(kin, events["top"]["frame"], events["impact"]["frame"])
    ghost = build_ghost(track, kin, events)
    try:
        from .pro import norms_assessment
        norms = norms_assessment(features, kin, events)
    except Exception:
        norms = None

    pano = panorama_path or str(Path(out_path).with_suffix(".panorama.png"))
    panorama_strip(track, video_path, events, kin, pano)
    pano_b64 = base64.b64encode(Path(pano).read_bytes()).decode()

    figs = [
        _sequence_fig(kin, events),
        _hand_speed_fig(kin, events),
        _separation_fig(track, kin, events),
        _loop3d_fig(track, events, ghost=ghost),
    ]
    charts = [
        f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
        for i, f in enumerate(figs)
    ]

    conf = features.get("confidence", "unknown")
    notes = features.get("notes", [])
    notes_html = "".join(f"<li>{_esc(n)}</li>" for n in notes)
    reliable = events.get("reliable", False)
    badge_cls = "ok" if reliable else "bad"
    badge_txt = (f"confidence: {conf} · events reliable" if reliable
                 else f"confidence: {conf} · events unreliable — treat visuals "
                      "as illustrative")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SwingSense report</title>
<style>
  :root {{
    --paper:#F6F7F4; --ink:#14211A; --green:#2E6B46; --orange:#E8590C;
    --blue:#4178B4; --line:#D8DCD4;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
         font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 20px 80px; }}
  header {{ padding:36px 0 10px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase;
              color:var(--green); font-weight:700; }}
  h1 {{ font-size:clamp(28px,5vw,44px); margin:.1em 0 .2em; letter-spacing:-.02em; }}
  .badge {{ display:inline-block; font:600 12px/1 ui-monospace, monospace;
            padding:6px 10px; border-radius:4px; }}
  .badge.ok  {{ background:#E4F0E8; color:var(--green); }}
  .badge.bad {{ background:#FBE9E0; color:var(--orange); }}
  .hero {{ margin:22px -20px 0; overflow-x:auto; background:#10150F;
           border-top:3px solid var(--green); border-bottom:3px solid var(--green); }}
  .hero img {{ display:block; height:380px; width:auto; max-width:none; }}
  .legend {{ font:12px ui-monospace, monospace; color:#5A6258; margin-top:8px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
          margin:0 4px -1px 10px; }}
  section h2 {{ margin:48px 0 6px; font-size:13px; letter-spacing:.18em;
                text-transform:uppercase; color:var(--green);
                border-bottom:1px solid var(--line); padding-bottom:8px; }}
  .summary {{ font-size:21px; line-height:1.5; max-width:46em; }}
  .block {{ background:#fff; border:1px solid var(--line); border-radius:8px;
            padding:16px 20px; margin:14px 0; }}
  .block.warn {{ border-left:4px solid var(--orange); }}
  .block h3 {{ margin:0 0 8px; font-size:15px; }}
  .block ul {{ margin:0; padding-left:20px; }}
  .block li {{ margin:6px 0; }}
  .src {{ color:#5A6258; font-size:13px; }}
  .chart {{ background:#fff; border:1px solid var(--line); border-radius:8px;
            padding:8px; margin:18px 0; }}
  .caveats {{ font-size:14px; color:#5A6258; }}
  footer {{ margin-top:60px; font-size:13px; color:#5A6258;
            border-top:1px solid var(--line); padding-top:14px; }}
</style></head><body><div class="wrap">

<header>
  <div class="eyebrow">SwingSense · feel ain't real</div>
  <h1>Swing report</h1>
  <span class="badge {badge_cls}">{_esc(badge_txt)}</span>
</header>

<div class="hero"><img alt="Swing panorama with velocity-colored skeleton"
  src="data:image/png;base64,{pano_b64}"></div>
<div class="legend">skeleton color = joint speed:
  <span class="dot" style="background:#4678B4"></span>slow
  <span class="dot" style="background:#B0851E"></span>quick
  <span class="dot" style="background:#E8590C"></span>fast
  &nbsp;·&nbsp; orange arc = hand path</div>

<section>
  <h2>The read</h2>
  {_tier1_html(analysis, features, seq, norms)}
</section>

<section>
  <h2>For nerds</h2>
  <div class="chart">{charts[0]}</div>
  <div class="chart">{charts[1]}</div>
  <div class="chart">{charts[2]}</div>
  <div class="chart">{charts[3]}
    <p class="caveats" style="padding:0 12px">{_esc(ghost.description)}</p>
  </div>
  <div class="block caveats"><h3>Measurement caveats</h3>
    <ul>{notes_html or "<li>None recorded.</li>"}</ul>
    <p>All numbers are single-camera 2D proxies. Timing and shape are
    trustworthy; absolute magnitudes and anything depth-dependent are not.
    The club is not tracked.</p>
  </div>
</section>

<footer>Generated by SwingSense from {_esc(Path(video_path).name)} ·
charts by Plotly (CDN)</footer>
</div></body></html>"""

    Path(out_path).write_text(doc)
    return out_path
