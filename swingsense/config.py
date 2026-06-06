"""Configuration and filesystem locations for SwingSense.

Everything lives under a single data directory so the tool is self-contained and
easy to back up. The location can be overridden with SWINGSENSE_HOME.
"""

from __future__ import annotations

import os
from pathlib import Path

# Default reasoning model. Override with SWINGSENSE_MODEL. Sonnet is the cost /
# quality default; point it at an Opus id for deeper reasoning.
DEFAULT_MODEL = "claude-sonnet-4-6"


def data_home() -> Path:
    """Root directory for the local database and any user overrides."""
    override = os.environ.get("SWINGSENSE_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".swingsense"
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path() -> Path:
    return data_home() / "history.db"


def model() -> str:
    return os.environ.get("SWINGSENSE_MODEL", DEFAULT_MODEL)


def api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def kb_dir() -> Path:
    """The bundled knowledge base directory (physics + coaching)."""
    return Path(__file__).resolve().parent / "kb"
