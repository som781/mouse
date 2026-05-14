"""Tests for mouse.config — schema validation and merge order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mouse.config import ConfigError, MouseConfig, _validate, load_config
from mouse.errors import MouseError


def test_defaults():
    c = MouseConfig.defaults()
    assert c.model == "gpt-5.4-mini"
    assert c.max_steps == 25
    assert c.max_retries == 2
    assert c.auto_approve is False


def test_validate_accepts_known_keys():
    raw = {"model": "gpt-4o", "max_steps": 10, "auto_approve": True}
    cleaned = _validate(raw, "test")
    assert cleaned == raw


def test_validate_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown config key 'modle'"):
        _validate({"modle": "typo"}, "test")


def test_validate_rejects_wrong_type():
    with pytest.raises(ConfigError, match="must be an integer"):
        _validate({"max_steps": "twenty"}, "test")


def test_validate_rejects_bool_as_int():
    # bool is a subclass of int in Python — schema must catch this.
    with pytest.raises(ConfigError, match="must be an integer"):
        _validate({"max_steps": True}, "test")


def test_validate_rejects_int_as_bool():
    with pytest.raises(ConfigError, match="must be a boolean"):
        _validate({"auto_approve": 1}, "test")


def test_validate_rejects_non_dict_root():
    with pytest.raises(ConfigError, match="must be a JSON object"):
        _validate(["not", "a", "dict"], "test")  # type: ignore[arg-type]


def test_config_error_is_mouse_error():
    assert issubclass(ConfigError, MouseError)


def test_load_config_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "override.json"
    cfg.write_text(json.dumps({"model": "claude-3-5-sonnet", "max_steps": 7}))
    monkeypatch.setenv("MOUSE_CONFIG", str(cfg))
    # Make sure no project-level mouse.config.json sneaks in.
    monkeypatch.chdir(tmp_path)
    c = load_config()
    assert c.model == "claude-3-5-sonnet"
    assert c.max_steps == 7
    # Unset fields fall back to defaults.
    assert c.max_retries == 2


def test_load_config_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "broken.json"
    cfg.write_text("{not valid json")
    monkeypatch.setenv("MOUSE_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config()
