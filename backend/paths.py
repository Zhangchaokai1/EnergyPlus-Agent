"""Runtime root shared by the backend and the core simulation modules.

The runtime root holds the core modules (``train_office_surrogate.py`` ...)
together with ``models/``, ``weather/`` and optional ``datasets/``.

Resolution order:
1. ``PAG_RUNTIME_ROOT`` environment variable;
2. the folder containing this repository, if it already provides the core
   modules (research-workspace layout);
3. the bundled ``core/`` folder of this repository (default for a fresh clone).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]


def _detect_root() -> Path:
    env = os.environ.get("PAG_RUNTIME_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if (REPO_DIR.parent / "train_office_surrogate.py").exists():
        return REPO_DIR.parent
    return REPO_DIR / "core"


ROOT = _detect_root()
