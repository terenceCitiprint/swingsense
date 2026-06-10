"""Prompt construction for the reasoning engine.

Kept separate from the API call so the assembled prompt can be inspected offline
(via `swingsense feel --dry-run`) without spending a token.
"""

from __future__ import annotations

import json

from .db import Swing
from .knowledge import KBEntry, render_for_prompt

SYSTEM_PROMPT = """\
You are SwingSense, an analyst that bridges a golfer's subjective *feel* and the \
objective biomechanics/physics of their swing.

Core principle: FEEL AIN'T REAL. A described feel is a hypothesis about what the \
body might be doing, never ground truth. Your job is to translate the feel into \
candidate mechanics, cross-reference those against the provided physics \
principles and coaching frameworks, surface contradictions, and propose cues or \
drills the golfer can test.

Be specific and grounded. Cite the physics principle or coaching framework you \
draw on by name. You may be given MEASURED FEATURES extracted from a swing \
video; when present, weight them according to their stated per-swing confidence \
and the notes/caveats attached — single-camera 2D pose is a proxy, the club is \
not tracked, and low frame rates blur impact. If the measured data is flagged \
unreliable, treat it as weak evidence and say so. Where a measured number and \
the player's feel disagree, surface that explicitly as a feel-vs-real flag. \
Never fabricate measurements you were not given.
"""

# Phase 0 has no measured data, so we ask the model to return a structured
# analysis we can render and store. Output is a single JSON object.
OUTPUT_SCHEMA = """\
Return ONLY a single JSON object (no markdown fences, no prose around it) with \
this exact shape:

{
  "summary": "one or two sentence plain-language read of what's likely going on",
  "feel_translation": [
    {"mechanic": "the candidate body mechanic", "body_region": "e.g. pelvis/lead wrist/thorax", "explanation": "why this feel maps here"}
  ],
  "physics_cross_reference": [
    {"principle": "name of the physics principle", "relation": "supports|contradicts|neutral", "note": "how it relates"}
  ],
  "contradictions": ["explicit feel-vs-real tensions to watch for, if any"],
  "suggestions": [
    {"cue_or_drill": "what to try", "rationale": "why", "source": "coaching framework or principle name"}
  ],
  "what_to_measure_next": ["the single most useful thing to capture on video/sensor to confirm or refute this"],
  "confidence": "low|medium|high"
}
"""


def build_user_prompt(
    feel: str,
    kb: list[KBEntry],
    club: str | None,
    history: list[Swing] | None = None,
    features: dict | None = None,
) -> str:
    parts: list[str] = []
    parts.append("KNOWLEDGE BASE (reason over these, cite by name):")
    parts.append(render_for_prompt(kb))
    parts.append("")

    if features:
        parts.append(
            "MEASURED FEATURES from the swing video (2D single-camera proxies — "
            "respect the 'confidence' field and notes):"
        )
        parts.append(json.dumps(features, indent=2))
        parts.append("")

    if history:
        parts.append("RECENT SWING HISTORY (most recent first, for continuity):")
        for s in history:
            club_str = f" [{s.club}]" if s.club else ""
            summary = s.analysis.get("summary", "")
            parts.append(f"- #{s.id}{club_str} felt: \"{s.feel}\" -> {summary}")
        parts.append("")

    club_line = f"\nClub: {club}" if club else ""
    parts.append("CURRENT SWING TO ANALYZE:")
    feel_line = (
        f'Feel description: "{feel}"'
        if feel
        else "Feel description: (none given — analyze from the measured features)"
    )
    parts.append(f"{feel_line}{club_line}")
    parts.append("")
    parts.append(OUTPUT_SCHEMA)
    return "\n".join(parts)
