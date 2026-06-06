"""The reasoning engine: send the assembled prompt to Claude, parse the result.

Phase 0 is text-only — no video features yet — so the engine reasons purely from
the feel description plus the knowledge base and recent history.
"""

from __future__ import annotations

import json

from . import config
from .db import Swing
from .knowledge import KBEntry
from .prompts import SYSTEM_PROMPT, build_user_prompt


class EngineError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """Parse the model's JSON, tolerating stray prose or code fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip a leading ```json / ``` fence and the trailing fence.
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise EngineError(f"Model did not return JSON. Raw output:\n{text}")
    return json.loads(text[start : end + 1])


def analyze_feel(
    feel: str,
    kb: list[KBEntry],
    club: str | None = None,
    history: list[Swing] | None = None,
) -> dict:
    """Call Claude and return the parsed structured analysis."""
    key = config.api_key()
    if not key:
        raise EngineError(
            "ANTHROPIC_API_KEY is not set. Export it, or use --dry-run to preview "
            "the prompt without calling the API."
        )

    # Imported lazily so --dry-run and tests work without the SDK installed.
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    user_prompt = build_user_prompt(feel, kb, club, history)

    try:
        resp = client.messages.create(
            model=config.model(),
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:  # surface a clean message to the CLI
        raise EngineError(f"Claude API call failed: {exc}") from exc

    raw = "".join(block.text for block in resp.content if block.type == "text")
    analysis = _extract_json(raw)
    analysis["_model"] = config.model()
    return analysis
