"""Compare two swings: time-aligned sequence overlay + metric deltas + panoramas.

Alignment: each swing is resampled onto a normalized phase axis where
0 = address, 0.75 = top, 1.0 = impact. That removes raw-tempo differences so
the *shape* and *order* of the motion can be compared directly; the tempo
difference itself is reported as a number.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np

from .vision.kinematics import Kinematics, compute_kinematics, sequence_read
from .vision.pose import PoseTrack

_PHASE_TOP = 0.75  # where "top of backswing" lands on the normalized axis


def _esc(s) -> str:
    return html.escape(str(s or ""))


def _phase_resample(sig: np.ndarray, a: int, top: int, imp: int,
                    n: int = 200) -> np.ndarray:
    """Resample a per-frame signal onto the normalized phase axis."""
    phase = np.linspace(0, 1, n)
    back_n = int(n * _PHASE_TOP)
    back_idx = np.linspace(a, top, back_n, endpoint=False)
    down_idx = np.linspace(top, imp, n - back_n)
    idx = np.concatenate([back_idx, down_idx])
    return np.interp(idx, np.arange(len(sig)), sig), phase


def _overlay_fig(kinA: Kinematics, evA: dict, kinB: Kinematics, evB: dict,
                 labelA: str, labelB: str):
    import plotly.graph_objects as go

    series = [
        ("pelvis", "hip_ang_vel", "#2E6B46"),
        ("thorax", "shoulder_ang_vel", "#4178B4"),
        ("hands", "hand_speed", "#E8590C"),
    ]
    fig = go.Figure()
    for (kin, ev, label, dash) in ((kinA, evA, labelA, None),
                                   (kinB, evB, labelB, "dot")):
        a, top, imp = (ev["address"]["frame"], ev["top"]["frame"],
                       ev["impact"]["frame"])
        for name, attr, color in series:
            sig = getattr(kin, attr)
            y, phase = _phase_resample(sig, a, top, imp)
            # Normalize each trace to its own peak so A and B share a scale.
            peak = float(np.max(y)) or 1.0
            fig.add_trace(go.Scatter(
                x=phase, y=y / peak, name=f"{name} — {label}",
                line=dict(color=color, width=2.5, dash=dash),
            ))
    fig.add_vline(x=_PHASE_TOP, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="top")
    fig.add_vline(x=1.0, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="impact")
    fig.update_layout(
        title=f"Time-aligned sequence — {labelA} (solid) vs {labelB} (dotted), "
              "each trace normalized to its own peak",
        xaxis_title="swing phase (0 = address, 1 = impact)",
        yaxis_title="relative intensity",
        template="plotly_white", height=460,
        legend=dict(orientation="h", y=1.14),
        margin=dict(t=100, b=40),
    )
    return fig


def _delta_rows(featsA: dict, featsB: dict, seqA: dict, seqB: dict) -> str:
    def sep_at_top(feats):
        return feats.get("metrics", {}).get("rotation", {}).get(
            "top", {}).get("separation_deg")

    rows = [
        ("Tempo (back:down)",
         featsA["metrics"]["tempo"].get("ratio_back_to_down"),
         featsB["metrics"]["tempo"].get("ratio_back_to_down")),
        ("Separation @ top (deg, proxy)", sep_at_top(featsA), sep_at_top(featsB)),
        ("Firing order", " → ".join(seqA["order"]), " → ".join(seqB["order"])),
        ("Sequence verdict", seqA["verdict"], seqB["verdict"]),
        ("Confidence", featsA.get("confidence"), featsB.get("confidence")),
    ]
    out = []
    for name, a, b in rows:
        out.append(f"<tr><td>{_esc(name)}</td><td>{_esc(a)}</td>"
                   f"<td>{_esc(b)}</td></tr>")
    return "".join(out)


def build_comparison(
    trackA: PoseTrack, featsA: dict, videoA: str,
    trackB: PoseTrack, featsB: dict, videoB: str,
    out_path: str,
    labelA: str = "swing A", labelB: str = "swing B",
) -> str:
    from .vision.visuals import panorama_strip

    kinA, kinB = compute_kinematics(trackA), compute_kinematics(trackB)
    evA, evB = featsA["events"], featsB["events"]
    seqA = sequence_read(kinA, evA["top"]["frame"], evA["impact"]["frame"])
    seqB = sequence_read(kinB, evB["top"]["frame"], evB["impact"]["frame"])

    panos = []
    for tag, track, video, ev, kin in (("A", trackA, videoA, evA, kinA),
                                       ("B", trackB, videoB, evB, kinB)):
        p = str(Path(out_path).with_suffix(f".pano{tag}.png"))
        panorama_strip(track, video, ev, kin, p, panel_height=360)
        panos.append(base64.b64encode(Path(p).read_bytes()).decode())

    fig = _overlay_fig(kinA, evA, kinB, evB, labelA, labelB)
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SwingSense comparison</title>
<style>
  :root {{ --paper:#F6F7F4; --ink:#14211A; --green:#2E6B46; --orange:#E8590C;
           --line:#D8DCD4; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
         font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 20px 80px; }}
  header {{ padding:36px 0 16px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase;
              color:var(--green); font-weight:700; }}
  h1 {{ font-size:clamp(26px,5vw,40px); margin:.1em 0; letter-spacing:-.02em; }}
  h2 {{ margin:40px 0 6px; font-size:13px; letter-spacing:.18em;
        text-transform:uppercase; color:var(--green);
        border-bottom:1px solid var(--line); padding-bottom:8px; }}
  .hero {{ margin:8px -20px; overflow-x:auto; background:#10150F;
           border-top:3px solid var(--green);
           border-bottom:3px solid var(--green); }}
  .hero img {{ display:block; height:280px; width:auto; max-width:none; }}
  .tag {{ font:600 12px ui-monospace, monospace; color:#5A6258; margin:14px 0 2px; }}
  table {{ border-collapse:collapse; width:100%; background:#fff;
           border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  th, td {{ text-align:left; padding:10px 14px;
            border-bottom:1px solid var(--line); }}
  th {{ background:#EDF0EA; font-size:13px; letter-spacing:.06em;
        text-transform:uppercase; }}
  .chart {{ background:#fff; border:1px solid var(--line); border-radius:8px;
            padding:8px; margin:18px 0; }}
  footer {{ margin-top:50px; font-size:13px; color:#5A6258;
            border-top:1px solid var(--line); padding-top:14px; }}
</style></head><body><div class="wrap">
<header>
  <div class="eyebrow">SwingSense · comparison</div>
  <h1>{_esc(labelA)} vs {_esc(labelB)}</h1>
</header>

<div class="tag">{_esc(labelA)} — {_esc(Path(videoA).name)}</div>
<div class="hero"><img src="data:image/png;base64,{panos[0]}"></div>
<div class="tag">{_esc(labelB)} — {_esc(Path(videoB).name)}</div>
<div class="hero"><img src="data:image/png;base64,{panos[1]}"></div>

<h2>What changed</h2>
<table><tr><th>Metric</th><th>{_esc(labelA)}</th><th>{_esc(labelB)}</th></tr>
{_delta_rows(featsA, featsB, seqA, seqB)}</table>

<h2>Time-aligned motion</h2>
<div class="chart">{chart}</div>

<footer>Phase-aligned so tempo differences don't mask shape differences;
all values are 2D single-camera proxies.</footer>
</div></body></html>"""
    Path(out_path).write_text(doc)
    return out_path
