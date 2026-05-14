"""Tests for the Electron API helpers."""

from __future__ import annotations

from pathlib import Path

from mouse.api.server import summarize_agent


class _FakeTool:
    def __init__(self, name: str, permission: str = 'safe', description: str = ''):
        self.name = name
        self.permission = type('P', (), {'value': permission})()
        self.description = description


class _FakeTools:
    def __init__(self):
        self._tools = {
            'bash': _FakeTool('bash', 'ask', 'Run shell commands'),
            'read_file': _FakeTool('read_file', 'safe', 'Read files'),
        }

    def list_names(self):
        return list(self._tools.keys())

    def get(self, name):
        return self._tools[name]


class _FakeSkill:
    def __init__(self, name: str, tier: str, description: str):
        self.name = name
        self.tier = tier
        self.description = description


class _FakeSkills:
    def all(self):
        return [_FakeSkill('test-skill', 'builtin', 'demo skill')]


class _FakeLLM:
    def __init__(self):
        self.model = 'gpt-test'


class _FakePermissions:
    def __init__(self):
        self.auto_approve = False


class _FakeAgent:
    def __init__(self):
        self.llm = _FakeLLM()
        self.permissions = _FakePermissions()
        self.stats = type('S', (), {'summary': lambda self: 'stats ok'})()
        self.messages = [1, 2]
        self.tools = _FakeTools()
        self.skills = _FakeSkills()


class _FakeSession:
    def __init__(self):
        self.id = 'abc123'
        self.title = 'Demo'
        self.status = 'active'
        self.created_at = 1.0
        self.updated_at = 2.0
        self.model = 'gpt-test'
        self.total_tokens = 42
        self.summary = 'hello'


class _FakeSessionMgr:
    def __init__(self):
        self.active = _FakeSession()

    def list_sessions(self, limit: int = 20):
        return [self.active]


def test_summarize_agent_returns_structured_state():
    state = summarize_agent(_FakeAgent(), _FakeSessionMgr())
    assert state['model'] == 'gpt-test'
    assert state['messages'] == 2
    assert state['session']['id'] == 'abc123'
    assert len(state['tools']) == 2
    assert len(state['skills']) == 1
    assert len(state['sessions']) == 1
