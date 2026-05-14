"""User preferences — persistent style and tool preferences on disk.

The preferences store is a tiny JSON-backed key/value bag that survives
across sessions. It holds things the agent learns (or the user sets
explicitly) about how this user likes to work::

    memory/preferences.json

        {
          "coding": {
            "language": "python",
            "formatter": "ruff",
            "line_length": 88
          },
          "tools": {
            "preferred_shell": "zsh",
            "avoid": ["curl"]
          },
          "style": {
            "tone": "terse"
          }
        }

Design notes:
- Dotted keys (``coding.formatter``) map to nested dicts under the hood.
  This keeps the file hand-editable while letting callers address
  specific settings without walking the structure.
- Writes are atomic: the store serializes to a temp file in the same
  directory, then ``os.replace`` to the final path. This avoids
  corrupting the file if the process is killed mid-write.
- The store never mutates unknown top-level keys — callers can set any
  path they like. We're deliberately schemaless; the memory injector
  in Sprint 4 Step 4 decides what to surface.
- No file locking. Single-agent-per-dir is the assumed deployment.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mouse.errors import MouseError


class PreferencesError(MouseError):
    """Raised on preferences I/O or malformed-file failures."""


def default_preferences_path() -> Path:
    """``~/.mouse/memory/preferences.json`` — the default file location."""
    return Path.home() / ".mouse" / "memory" / "preferences.json"


_MISSING = object()


class Preferences:
    """File-backed JSON preferences store.

    One instance per agent process. Reads are served from an in-memory
    copy; writes flush atomically to disk. Call :meth:`reload` to
    re-read if an external process may have modified the file.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path).expanduser() if path else default_preferences_path()
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self.reload()

    # ── I/O ──

    def reload(self) -> None:
        """Re-read the on-disk file into memory (empty dict if missing)."""
        if not self.path.exists():
            self._data = {}
            return
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as e:
            raise PreferencesError(f"failed to read {self.path}: {e}") from e
        if not text.strip():
            self._data = {}
            return
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise PreferencesError(
                f"preferences file {self.path} is not valid JSON: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise PreferencesError(
                f"preferences file {self.path} must contain a JSON object"
            )
        self._data = parsed

    def _flush(self) -> None:
        """Atomic write of the in-memory dict to disk."""
        try:
            # NamedTemporaryFile in the same directory guarantees the
            # replace is atomic on the same filesystem.
            fd, tmp_path = tempfile.mkstemp(
                prefix=".preferences.", suffix=".tmp", dir=self.path.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, sort_keys=True)
                    f.write("\n")
                os.replace(tmp_path, self.path)
            except Exception:
                # Best-effort cleanup of the temp file.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            raise PreferencesError(f"failed to write {self.path}: {e}") from e

    # ── Read ──

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at ``key`` (dotted path), or ``default``."""
        node = self._resolve(key)
        if node is _MISSING:
            return default
        return node

    def has(self, key: str) -> bool:
        """True if ``key`` exists (even if its value is falsy/None)."""
        return self._resolve(key) is not _MISSING

    def all(self) -> dict[str, Any]:
        """Deep copy of the full preferences dict."""
        return json.loads(json.dumps(self._data))

    def _resolve(self, key: str) -> Any:
        parts = self._split(key)
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return _MISSING
            node = node[part]
        return node

    # ── Write ──

    def set(self, key: str, value: Any) -> None:
        """Set ``key`` (dotted path) to ``value`` and flush to disk."""
        parts = self._split(key)
        self._validate_json(value, key)
        node = self._data
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value
        self._flush()

    def update(self, values: dict[str, Any]) -> None:
        """Set multiple dotted keys in one flush.

        All keys and values are validated *before* any mutation so a
        bad entry later in the batch can't leave the store with a
        half-applied in-memory state. Either every key lands or none
        do.
        """
        if not isinstance(values, dict):
            raise PreferencesError("update: values must be a dict")

        # Validate everything up front — _split enforces key shape,
        # _validate_json enforces serializability. If any raise we
        # bail out before touching self._data.
        validated: list[tuple[list[str], Any]] = []
        for k, v in values.items():
            parts = self._split(k)
            self._validate_json(v, k)
            validated.append((parts, v))

        for parts, v in validated:
            node = self._data
            for part in parts[:-1]:
                existing = node.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    node[part] = existing
                node = existing
            node[parts[-1]] = v
        self._flush()

    def delete(self, key: str) -> bool:
        """Remove ``key``. Returns True if something was deleted."""
        parts = self._split(key)
        node = self._data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            return False
        del node[parts[-1]]
        self._flush()
        return True

    def clear(self) -> None:
        """Remove every preference."""
        self._data = {}
        self._flush()

    # ── Internals ──

    @staticmethod
    def _split(key: str) -> list[str]:
        if not isinstance(key, str) or not key.strip():
            raise PreferencesError("key must be a non-empty string")
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise PreferencesError(f"invalid key: {key!r}")
        return parts

    @staticmethod
    def _validate_json(value: Any, key: str) -> None:
        """Reject values that can't round-trip through JSON."""
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise PreferencesError(
                f"value for {key!r} is not JSON-serializable: {e}"
            ) from e
