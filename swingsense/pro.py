"""Pro reference library + published tour norms.

Two layers of "compare me to the greats":

1. TOUR_NORMS — quantitative bands from published golf-biomechanics research
   (kinematic-sequence and tempo studies). These are camera-independent
   quantities only: tempo ratio, downswing duration, and the time gaps between
   segment peaks. Angle magnitudes are deliberately excluded — our 2D
   image-plane proxies are not comparable to lab 3D values.

2. A reference library — any swing video the user has rights to use can be
   ingested with `swingsense pro add`. The clip is processed by the SAME
   pipeline as the player's swings (same proxies, same camera assumptions), so
   phase-aligned curve overlays ARE apples-to-apples. The library stores only
   the derived, phase-normalized signals (small JSON), not the video.
"""

from __future__ import annotations

import json
import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .vision.kinematics import (
    Kinematics,
    compute_kinematics,
    separation_series,
    sequence_read,
)
from .vision.pose import PoseTrack

# Published, approximate bands (kinematic-sequence / tempo literature).
# Tempo: elite swings cluster near 3:1 back:down. Downswing: ~0.23–0.29 s.
# Peak gaps: pelvis→thorax→arm peaks spaced roughly 40–120 ms apart.
TOUR_NORMS = {
    "tempo_ratio": (2.7, 3.3),
    "downswing_s": (0.22, 0.30),
    "peak_gap_ms": (40, 120),
    "note": (
        "Approximate bands from published kinematic-sequence and tempo "
        "research; camera-independent timing quantities only."
    ),
}

_N_PHASE = 200
_PHASE_TOP = 0.75  # keep aligned with compare.py


def references_dir() -> Path:
    d = config.data_home() / "references"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _phase_resample(sig: np.ndarray, a: int, top: int, imp: int) -> np.ndarray:
    back_n = int(_N_PHASE * _PHASE_TOP)
    back_idx = np.linspace(a, top, back_n, endpoint=False)
    down_idx = np.linspace(top, imp, _N_PHASE - back_n)
    idx = np.concatenate([back_idx, down_idx])
    return np.interp(idx, np.arange(len(sig)), sig)


def add_reference(
    track: PoseTrack, features: dict, name: str, source: str = ""
) -> Path:
    """Process a reference swing into stored phase-normalized signals."""
    kin = compute_kinematics(track)
    ev = features["events"]
    a, top, imp = (ev["address"]["frame"], ev["top"]["frame"],
                   ev["impact"]["frame"])
    seq = sequence_read(kin, top, imp)
    sep = np.abs(separation_series(track))

    def rs(sig):
        return [round(float(v), 4) for v in _phase_resample(sig, a, top, imp)]

    data = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fps": track.fps,
        "confidence": features.get("confidence"),
        "tempo_ratio": features["metrics"]["tempo"].get("ratio_back_to_down"),
        "downswing_s": features["metrics"]["tempo"].get("downswing_s"),
        "firing_order": seq["order"],
        "signals": {
            "pelvis": rs(kin.hip_ang_vel),
            "thorax": rs(kin.shoulder_ang_vel),
            "arm": rs(kin.arm_ang_vel),
            "hands": rs(kin.hand_speed),
            "separation": rs(sep),
        },
        "notes": [
            "Signals are phase-normalized (0=address, 0.75=top, 1=impact) and "
            "produced by the same 2D single-camera pipeline as player swings.",
        ],
    }
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = references_dir() / f"{safe}.json"
    path.write_text(json.dumps(data, indent=1))
    return path


def list_references() -> list[dict]:
    out = []
    for p in sorted(references_dir().glob("*.json")):
        try:
            d = json.loads(p.read_text())
            d["_path"] = str(p)
            out.append(d)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_reference(name: str) -> dict | None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    p = references_dir() / f"{safe}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def norms_assessment(features: dict, kin: Kinematics, events: dict) -> list[str]:
    """Plain-language lines comparing this swing to TOUR_NORMS bands."""
    lines: list[str] = []
    tempo = features["metrics"]["tempo"]
    ratio = tempo.get("ratio_back_to_down")
    lo, hi = TOUR_NORMS["tempo_ratio"]
    if ratio is not None:
        if lo <= ratio <= hi:
            lines.append(f"Tempo {ratio}:1 sits inside the tour band "
                         f"({lo}–{hi}:1).")
        else:
            side = "quicker" if ratio < lo else "slower"
            lines.append(f"Tempo {ratio}:1 is outside the tour band "
                         f"({lo}–{hi}:1) — backswing relatively {side}.")
    ds = tempo.get("downswing_s")
    dlo, dhi = TOUR_NORMS["downswing_s"]
    if ds is not None and tempo.get("reliable", False):
        inside = dlo <= ds <= dhi
        lines.append(f"Downswing {ds}s — tour band is {dlo}–{dhi}s"
                     f"{'' if inside else ' (outside)'}.")
    # Peak gaps in ms (only meaningful when the order is right).
    top, imp = events["top"]["frame"], events["impact"]["frame"]
    peaks = kin.segment_peaks(top, imp)
    glo, ghi = TOUR_NORMS["peak_gap_ms"]
    frame_ms = (kin.t[1] - kin.t[0]) * 1000.0 if len(kin.t) > 1 else 33.3
    gaps = {
        "pelvis→thorax": (peaks["thorax"] - peaks["pelvis"]) * frame_ms,
        "thorax→arm": (peaks["arm"] - peaks["thorax"]) * frame_ms,
    }
    for label, g in gaps.items():
        if g < 0:
            lines.append(f"Peak gap {label}: reversed — the distal segment "
                         f"peaked first (tour swings: +{glo}–{ghi} ms).")
        elif g == 0:
            lines.append(f"Peak gap {label}: peaked simultaneously — tour "
                         f"swings show a +{glo}–{ghi} ms gap.")
        elif g < glo or g > ghi:
            lines.append(f"Peak gap {label}: {g:.0f} ms vs tour {glo}–{ghi} ms.")
        else:
            lines.append(f"Peak gap {label}: {g:.0f} ms — inside tour band.")
    return lines


# --------------------------------------------------------------------------- #
# Player-vs-reference overlay report
# --------------------------------------------------------------------------- #

def _esc(s) -> str:
    return html.escape(str(s or ""))


def build_pro_comparison(
    track: PoseTrack,
    features: dict,
    video_path: str,
    ref: dict,
    out_path: str,
    label: str = "you",
) -> str:
    """Phase-aligned overlay of the player's swing against a stored reference."""
    import plotly.graph_objects as go
    from .vision.visuals import panorama_strip

    kin = compute_kinematics(track)
    ev = features["events"]
    a, top, imp = ev["address"]["frame"], ev["top"]["frame"], ev["impact"]["frame"]
    seq = sequence_read(kin, top, imp)
    phase = np.linspace(0, 1, _N_PHASE)

    colors = {"pelvis": "#2E6B46", "thorax": "#4178B4",
              "arm": "#B07A1E", "hands": "#E8590C"}
    player = {
        "pelvis": _phase_resample(kin.hip_ang_vel, a, top, imp),
        "thorax": _phase_resample(kin.shoulder_ang_vel, a, top, imp),
        "arm": _phase_resample(kin.arm_ang_vel, a, top, imp),
        "hands": _phase_resample(kin.hand_speed, a, top, imp),
    }

    fig = go.Figure()
    for name in ("pelvis", "thorax", "hands"):
        you = np.asarray(player[name])
        pro = np.asarray(ref["signals"][name], dtype=float)
        ypk = float(np.max(you)) or 1.0
        ppk = float(np.max(pro)) or 1.0
        fig.add_trace(go.Scatter(x=phase, y=you / ypk,
                                 name=f"{name} — {label}",
                                 line=dict(color=colors[name], width=2.5)))
        fig.add_trace(go.Scatter(x=phase, y=pro / ppk,
                                 name=f"{name} — {ref['name']}",
                                 line=dict(color=colors[name], width=2.5,
                                           dash="dot")))
    fig.add_vline(x=_PHASE_TOP, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="top")
    fig.add_vline(x=1.0, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="impact")
    fig.update_layout(
        title=f"Phase-aligned: {label} (solid) vs {ref['name']} (dotted) — "
              "each trace normalized to its own peak",
        xaxis_title="swing phase (0 = address, 1 = impact)",
        yaxis_title="relative intensity",
        template="plotly_white", height=470,
        legend=dict(orientation="h", y=1.16), margin=dict(t=110, b=40),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Separation overlay.
    sep_you = _phase_resample(np.abs(separation_series(track)), a, top, imp)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=phase, y=sep_you, name=label,
                              line=dict(color="#2E6B46", width=3)))
    fig2.add_trace(go.Scatter(x=phase, y=ref["signals"]["separation"],
                              name=ref["name"],
                              line=dict(color="#5A6258", width=3, dash="dot")))
    fig2.add_vline(x=_PHASE_TOP, line=dict(color="#888", dash="dash", width=1))
    fig2.update_layout(
        title="Hip–shoulder separation through the swing (2D proxy, both "
              "measured by the same pipeline)",
        xaxis_title="swing phase", yaxis_title="separation (deg, proxy)",
        template="plotly_white", height=380, margin=dict(t=70, b=40),
    )
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    import base64
    pano = str(Path(out_path).with_suffix(".pano.png"))
    panorama_strip(track, video_path, ev, kin, pano, panel_height=360)
    pano_b64 = base64.b64encode(Path(pano).read_bytes()).decode()

    norms = norms_assessment(features, kin, ev)
    norms_html = "".join(f"<li>{_esc(n)}</li>" for n in norms)
    tempo = features["metrics"]["tempo"]
    rows = [
        ("Tempo (back:down)", tempo.get("ratio_back_to_down"),
         ref.get("tempo_ratio")),
        ("Downswing (s)", tempo.get("downswing_s"), ref.get("downswing_s")),
        ("Firing order", " → ".join(seq["order"]),
         " → ".join(ref.get("firing_order", []))),
        ("Capture confidence", features.get("confidence"),
         ref.get("confidence")),
    ]
    table = "".join(f"<tr><td>{_esc(n)}</td><td>{_esc(x)}</td>"
                    f"<td>{_esc(y)}</td></tr>" for n, x, y in rows)

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SwingSense — vs {_esc(ref['name'])}</title>
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
  table {{ border-collapse:collapse; width:100%; background:#fff;
           border:1px solid var(--line); }}
  th, td {{ text-align:left; padding:10px 14px;
            border-bottom:1px solid var(--line); }}
  th {{ background:#EDF0EA; font-size:13px; letter-spacing:.06em;
        text-transform:uppercase; }}
  .chart {{ background:#fff; border:1px solid var(--line); border-radius:8px;
            padding:8px; margin:18px 0; }}
  .block {{ background:#fff; border:1px solid var(--line); border-radius:8px;
            padding:16px 20px; margin:14px 0; font-size:15px; }}
  footer {{ margin-top:50px; font-size:13px; color:#5A6258;
            border-top:1px solid var(--line); padding-top:14px; }}
</style></head><body><div class="wrap">
<header>
  <div class="eyebrow">SwingSense · vs reference</div>
  <h1>{_esc(label)} vs {_esc(ref['name'])}</h1>
</header>

<div class="hero"><img src="data:image/png;base64,{pano_b64}"></div>

<h2>Numbers side by side</h2>
<table><tr><th>Metric</th><th>{_esc(label)}</th><th>{_esc(ref['name'])}</th></tr>
{table}</table>

<h2>vs published tour norms</h2>
<div class="block"><ul>{norms_html}</ul>
<p style="color:#5A6258">{_esc(TOUR_NORMS['note'])}</p></div>

<h2>Phase-aligned motion</h2>
<div class="chart">{chart}</div>
<div class="chart">{chart2}</div>

<footer>Reference processed by the identical 2D pipeline, so proxy curves are
directly comparable. Angle magnitudes are still proxies, not lab values.
</footer>
</div></body></html>"""
    Path(out_path).write_text(doc)
    return out_path
