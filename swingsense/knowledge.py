"""Load the curated knowledge base (physics rules + coaching frameworks).

The KB is plain YAML on disk so it is version-controlled and *you* own the
philosophy. The engine reasons over whatever is loaded here; growing the KB
never requires touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config


@dataclass
class KBEntry:
    id: str
    name: str
    category: str  # "physics" or "coaching"
    summary: str
    details: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)


def _load_dir(path: Path, category: str) -> list[KBEntry]:
    entries: list[KBEntry] = []
    if not path.exists():
        return entries
    for yaml_file in sorted(path.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text()) or {}
        for raw in data.get("entries", []):
            entries.append(
                KBEntry(
                    id=raw.get("id", ""),
                    name=raw.get("name", ""),
                    category=category,
                    summary=raw.get("summary", ""),
                    details=raw.get("details", ""),
                    source=raw.get("source", ""),
                    tags=raw.get("tags", []) or [],
                )
            )
    return entries


def load_knowledge() -> list[KBEntry]:
    base = config.kb_dir()
    return _load_dir(base / "physics", "physics") + _load_dir(
        base / "coaching", "coaching"
    )


def render_for_prompt(entries: list[KBEntry]) -> str:
    """Format KB entries as compact reference text for the model."""
    lines: list[str] = []
    for e in entries:
        src = f" (source: {e.source})" if e.source else ""
        lines.append(f"- [{e.category}] {e.name}{src}: {e.summary}")
        if e.details:
            lines.append(f"    {e.details}")
    return "\n".join(lines)
