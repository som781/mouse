"""Mouse configuration loader.

Resolution order (later sources override earlier):
    1. Built-in defaults
    2. User config:    ~/.mouse/config.json
    3. Project config: <project_root>/mouse.config.json
    4. $MOUSE_CONFIG env var (path to a JSON file)

The schema is validated manually — no extra dependencies. Unknown keys
cause a clear error so typos don't silently disappear.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from mouse.errors import MouseError
from mouse.project import PROJECT_ROOT


# ─── Schema ─────────────────────────────────────────────────────────

# field name → (python type, human description)
_SCHEMA: dict[str, tuple[type, str]] = {
    "model":           (str,  "LLM model identifier (litellm format)"),
    "max_steps":       (int,  "Maximum agent loop iterations per turn"),
    "max_retries":     (int,  "Retry attempts on LLM API errors"),
    "auto_approve":    (bool, "Skip permission prompts for ASK-tier tools"),
    "system_prompt":   (str,  "Override the default system prompt (empty = default)"),
    "mcp_config_path": (str,  "Override MCP config path (empty = auto-detect)"),
}


@dataclass
class MouseConfig:
    """Resolved Mouse configuration."""
    model: str = "gpt-5.4-mini"
    max_steps: int = 25
    max_retries: int = 2
    auto_approve: bool = False
    system_prompt: str = ""
    mcp_config_path: str = ""

    @classmethod
    def defaults(cls) -> MouseConfig:
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Errors ─────────────────────────────────────────────────────────

class ConfigError(MouseError):
    """Raised when a config file is invalid."""


# ─── Validation ─────────────────────────────────────────────────────

def _validate(raw: dict, source: str) -> dict:
    """Validate a raw config dict against the schema. Returns the cleaned dict.

    Rules:
      - Unknown keys are an error (catches typos).
      - Type mismatches are an error.
      - bool is checked before int to avoid `True` being accepted as int.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top-level value must be a JSON object")

    cleaned: dict = {}
    valid_keys = set(_SCHEMA.keys())

    for key, value in raw.items():
        if key not in valid_keys:
            hint = ", ".join(sorted(valid_keys))
            raise ConfigError(
                f"{source}: unknown config key '{key}'. Valid keys: {hint}"
            )
        expected_type, _desc = _SCHEMA[key]

        # Bool must be checked first because bool is a subclass of int.
        if expected_type is bool:
            if not isinstance(value, bool):
                raise ConfigError(
                    f"{source}: '{key}' must be a boolean, got {type(value).__name__}"
                )
        elif expected_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(
                    f"{source}: '{key}' must be an integer, got {type(value).__name__}"
                )
        elif expected_type is str:
            if not isinstance(value, str):
                raise ConfigError(
                    f"{source}: '{key}' must be a string, got {type(value).__name__}"
                )
        else:
            # Schema misconfiguration — should never happen in practice.
            raise ConfigError(f"{source}: unsupported schema type for '{key}'")

        cleaned[key] = value

    return cleaned


def _load_file(path: Path) -> dict:
    """Read and validate a single config file. Missing file → {}."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON — {e}") from e
    return _validate(raw, str(path))


# ─── Public API ─────────────────────────────────────────────────────

def config_search_paths() -> list[Path]:
    """Return the ordered list of config files Mouse will read (lowest → highest precedence)."""
    paths = [
        Path.home() / ".mouse" / "config.json",
        Path(PROJECT_ROOT) / "mouse.config.json",
    ]
    env_path = os.environ.get("MOUSE_CONFIG", "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser())
    return paths


def load_config() -> MouseConfig:
    """Load and merge config from all sources. Raises ConfigError on invalid input."""
    merged = MouseConfig.defaults().to_dict()

    for path in config_search_paths():
        layer = _load_file(path)
        merged.update(layer)

    # Construct via field names to be safe against schema drift.
    field_names = {f.name for f in fields(MouseConfig)}
    return MouseConfig(**{k: v for k, v in merged.items() if k in field_names})
