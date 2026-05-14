"""Tests for the JSON-backed preferences store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mouse.errors import MouseError
from mouse.memory.preferences import Preferences, PreferencesError


@pytest.fixture
def prefs(tmp_path: Path) -> Preferences:
    return Preferences(tmp_path / "preferences.json")


# ─── Errors & construction ──────────────────────────────────────────


def test_error_is_mouse_error():
    assert issubclass(PreferencesError, MouseError)


def test_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "deep" / "preferences.json"
    assert not path.parent.exists()
    Preferences(path)
    assert path.parent.is_dir()


def test_missing_file_yields_empty_prefs(tmp_path: Path):
    p = Preferences(tmp_path / "preferences.json")
    assert p.all() == {}
    assert p.get("anything") is None


def test_loads_existing_file(tmp_path: Path):
    path = tmp_path / "preferences.json"
    path.write_text('{"coding": {"formatter": "ruff"}}', encoding="utf-8")
    p = Preferences(path)
    assert p.get("coding.formatter") == "ruff"


def test_empty_file_yields_empty_prefs(tmp_path: Path):
    path = tmp_path / "preferences.json"
    path.write_text("   \n", encoding="utf-8")
    p = Preferences(path)
    assert p.all() == {}


def test_invalid_json_raises(tmp_path: Path):
    path = tmp_path / "preferences.json"
    path.write_text("{not valid", encoding="utf-8")
    with pytest.raises(PreferencesError, match="not valid JSON"):
        Preferences(path)


def test_non_object_root_raises(tmp_path: Path):
    path = tmp_path / "preferences.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(PreferencesError, match="JSON object"):
        Preferences(path)


# ─── get / has / all ────────────────────────────────────────────────


def test_get_with_default(prefs: Preferences):
    assert prefs.get("missing", "fallback") == "fallback"


def test_get_nested_dotted_key(prefs: Preferences):
    prefs.set("a.b.c", 42)
    assert prefs.get("a.b.c") == 42
    assert prefs.get("a.b") == {"c": 42}


def test_has_distinguishes_none_from_missing(prefs: Preferences):
    prefs.set("explicit.none", None)
    assert prefs.has("explicit.none") is True
    assert prefs.has("explicit.absent") is False


def test_has_traverses_non_dict_safely(prefs: Preferences):
    prefs.set("k", "string")
    # Descending past a scalar is not a hit.
    assert prefs.has("k.deeper") is False
    assert prefs.get("k.deeper") is None


def test_all_returns_deep_copy(prefs: Preferences):
    prefs.set("a.b", [1, 2, 3])
    snapshot = prefs.all()
    snapshot["a"]["b"].append(99)
    assert prefs.get("a.b") == [1, 2, 3]


# ─── set / update / delete / clear ──────────────────────────────────


def test_set_persists_to_disk(tmp_path: Path):
    path = tmp_path / "preferences.json"
    p = Preferences(path)
    p.set("coding.language", "python")
    # Re-open to confirm the write landed.
    p2 = Preferences(path)
    assert p2.get("coding.language") == "python"


def test_set_rejects_non_json_value(prefs: Preferences):
    with pytest.raises(PreferencesError, match="JSON-serializable"):
        prefs.set("bad", {1, 2, 3})  # sets aren't JSON


def test_set_overwrites_scalar_with_dict(prefs: Preferences):
    prefs.set("a", "scalar")
    prefs.set("a.b", "nested")
    # The scalar at "a" was replaced with a dict to make room for "a.b".
    assert prefs.get("a") == {"b": "nested"}


def test_set_empty_key_raises(prefs: Preferences):
    with pytest.raises(PreferencesError, match="non-empty"):
        prefs.set("", "value")


def test_set_dots_only_key_raises(prefs: Preferences):
    with pytest.raises(PreferencesError):
        prefs.set("...", "value")


def test_update_sets_many_at_once(prefs: Preferences):
    prefs.update({
        "coding.formatter": "ruff",
        "coding.language": "python",
        "style.tone": "terse",
    })
    assert prefs.get("coding.formatter") == "ruff"
    assert prefs.get("coding.language") == "python"
    assert prefs.get("style.tone") == "terse"


def test_update_rejects_non_dict(prefs: Preferences):
    with pytest.raises(PreferencesError, match="must be a dict"):
        prefs.update("nope")  # type: ignore[arg-type]


def test_update_is_atomic_on_validation_failure(tmp_path: Path):
    """Regression: if any value in an update batch fails validation,
    no value in the batch should land — neither in memory nor on
    disk. The prior implementation validated mid-loop and left
    earlier keys dirty in self._data."""
    path = tmp_path / "preferences.json"
    p = Preferences(path)
    with pytest.raises(PreferencesError, match="JSON-serializable"):
        p.update({
            "coding.language": "python",
            "bad.value": {1, 2, 3},  # set → not JSON-serializable
            "coding.formatter": "ruff",
        })
    # Nothing from the batch should be visible in memory.
    assert p.get("coding.language") is None
    assert p.get("coding.formatter") is None
    # Reload from disk to confirm nothing was persisted either.
    p2 = Preferences(path)
    assert p2.all() == {}


def test_update_flushes_once(tmp_path: Path):
    """After update, reopening should see all values — and only the
    final file should exist (no leftover temp files)."""
    path = tmp_path / "preferences.json"
    p = Preferences(path)
    p.update({"a": 1, "b.c": 2})
    p2 = Preferences(path)
    assert p2.get("a") == 1
    assert p2.get("b.c") == 2
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".preferences.")]
    assert leftover == []


def test_delete_returns_true_when_removed(prefs: Preferences):
    prefs.set("a.b", 1)
    assert prefs.delete("a.b") is True
    assert prefs.has("a.b") is False
    # Parent dict is still there (empty).
    assert prefs.get("a") == {}


def test_delete_returns_false_when_missing(prefs: Preferences):
    assert prefs.delete("not.here") is False


def test_delete_through_scalar_returns_false(prefs: Preferences):
    prefs.set("k", "scalar")
    assert prefs.delete("k.inside.scalar") is False


def test_clear_empties_everything(prefs: Preferences):
    prefs.set("a", 1)
    prefs.set("b.c", 2)
    prefs.clear()
    assert prefs.all() == {}


# ─── Atomic writes & reload ─────────────────────────────────────────


def test_reload_picks_up_external_changes(tmp_path: Path):
    path = tmp_path / "preferences.json"
    p = Preferences(path)
    # Write from outside.
    path.write_text('{"outside": "yes"}', encoding="utf-8")
    assert p.get("outside") is None  # stale in-memory view
    p.reload()
    assert p.get("outside") == "yes"


def test_flush_produces_sorted_indented_json(tmp_path: Path):
    path = tmp_path / "preferences.json"
    p = Preferences(path)
    p.set("z", 1)
    p.set("a", 2)
    text = path.read_text(encoding="utf-8")
    # Keys sorted alphabetically.
    assert text.index('"a"') < text.index('"z"')
    # Pretty-printed (contains a newline between entries).
    assert "\n" in text
    # Valid JSON round-trip.
    assert json.loads(text) == {"a": 2, "z": 1}


def test_reload_after_truncated_file(tmp_path: Path):
    path = tmp_path / "preferences.json"
    p = Preferences(path)
    p.set("a", 1)
    # Simulate truncation.
    path.write_text("", encoding="utf-8")
    p.reload()
    assert p.all() == {}
