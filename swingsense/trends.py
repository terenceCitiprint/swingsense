"""Trend tracking: tempo, separation, and confidence across logged swings.

Reads the local SQLite history. Only swings that carry measured `_features`
(i.e. analyzed from video) contribute numeric points; feel-only entries are
skipped.
"""

from __future__ import annotations

import html
from pathlib import Path

from . import db


def _esc(s) -> str:
    return html.escape(str(s or ""))


def collect(limit: int = 100, club: str | None = None) -> list[dict]:
    """Pull (swing_id, date, tempo, separation@top, confidence) per swing."""
    rows = []
    for s in reversed(db.list_swings(limit=limit, club=club)):  # oldest first
        feats = s.analysis.get("_features")
        if not feats:
            continue
        tempo = feats.get("metrics", {}).get("tempo", {})
        sep = (feats.get("metrics", {}).get("rotation", {})
               .get("top", {}).get("separation_deg"))
        rows.append({
            "id": s.id,
            "date": s.created_at[:10],
            "club": s.club,
            "tempo": tempo.get("ratio_back_to_down"),
            "tempo_reliable": tempo.get("reliable", False),
            "separation_top": sep,
            "confidence": feats.get("confidence"),
            "feel": s.feel,
        })
    return rows


def build_trends(out_path: str, limit: int = 100,
                 club: str | None = None) -> str | None:
    """Write the trends HTML; returns out_path, or None if no video swings."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = collect(limit=limit, club=club)
    if not rows:
        return None

    x = [f"#{r['id']}" for r in rows]
    hover = [f"#{r['id']} · {r['date']} · {r['club'] or '—'}<br>{r['feel']}"
             for r in rows]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Tempo ratio (back:down)",
                                        "Hip–shoulder separation @ top (proxy)"),
                        vertical_spacing=0.14)
    fig.add_trace(go.Scatter(
        x=x, y=[r["tempo"] for r in rows], mode="lines+markers",
        name="tempo", text=hover, hoverinfo="text+y",
        line=dict(color="#2E6B46", width=2.5),
        marker=dict(size=9, symbol=["circle" if r["tempo_reliable"] else "x"
                                    for r in rows])), row=1, col=1)
    fig.add_hline(y=3.0, line=dict(color="#888", dash="dash", width=1),
                  annotation_text="3:1 reference", row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=[abs(r["separation_top"]) if r["separation_top"] is not None
                else None for r in rows],
        mode="lines+markers", name="separation", text=hover,
        hoverinfo="text+y",
        line=dict(color="#4178B4", width=2.5), marker=dict(size=9)),
        row=2, col=1)
    fig.update_layout(template="plotly_white", height=620, showlegend=False,
                      margin=dict(t=60, b=40),
                      title=f"Trends across {len(rows)} video swings"
                            + (f" · {club}" if club else ""))
    chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

    note = ("Markers shown as × are swings where tempo fell outside the "
            "camera's reliable resolution — read those points loosely.")
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SwingSense trends</title>
<style>
  body {{ margin:0; background:#F6F7F4; color:#14211A;
         font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:0 20px 60px; }}
  header {{ padding:36px 0 10px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase;
              color:#2E6B46; font-weight:700; }}
  h1 {{ font-size:clamp(26px,5vw,38px); margin:.1em 0; }}
  .chart {{ background:#fff; border:1px solid #D8DCD4; border-radius:8px;
            padding:8px; margin:18px 0; }}
  .note {{ font-size:14px; color:#5A6258; }}
</style></head><body><div class="wrap">
<header><div class="eyebrow">SwingSense · trends</div>
<h1>Is the drill working?</h1></header>
<div class="chart">{chart}</div>
<p class="note">{_esc(note)}</p>
</div></body></html>"""
    Path(out_path).write_text(doc)
    return out_path
