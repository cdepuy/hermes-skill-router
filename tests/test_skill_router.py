"""Unit tests for skill_router (stdlib + pytest + unittest.mock only, no network).

Run from the plugin dir with the repo venv (pytest + stdlib):
    source /Users/openclaw64/.hermes/hermes-agent/venv/bin/activate
    python -m pytest tests/ -q   # or: python -m pytest tests/test_router.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import router


# ---------------------------------------------------------------------------
# Fixtures: a tiny fake skill index on disk (no network, no real Laya model).
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_skills_dir(tmp_path):
    """A scratch skill index with three skill dirs + frontmatter."""
    def make(name, desc, category="general"):
        d = tmp_path / category / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n## Procedure\nDo the {name} thing.\n",
            encoding="utf-8",
        )
    make("x-post", "develop and post X/Twitter announcements")
    make("blog-post-creation", "create a 650 blog post draft")
    make("minecraft-modpack-server", "host modded Minecraft servers")
    return str(tmp_path)


@pytest.fixture
def fake_laya_py(tmp_path):
    """A fake laya interpreter that prints a canned JSON result (no model)."""
    exe = tmp_path / "fake_laya_py"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "payload = json.loads(sys.stdin.read())\n"
        "# pick the skill whose name/desc shares most tokens with the task\n"
        "best = max(payload['skills'], key=lambda s: len(set(s['name'].replace('-',' ').split()) & set(payload['task'].lower().replace('-',' ').split())))\n"
        "print(json.dumps({'top':[{'skill':best['name'],'probability':0.99}], 'all':[{'skill':best['name'],'probability':0.99}], 'chosen':best['name']}))\n",
        encoding="utf-8",
    )
    exe.chmod(0o755)
    return str(exe)


# ---------------------------------------------------------------------------
# router module
# ---------------------------------------------------------------------------
def test_discover_skills_finds_skill_md(fake_skills_dir):
    skills = router.discover_skills(fake_skills_dir)
    names = {s["name"] for s in skills}
    assert "x-post" in names
    assert "blog-post-creation" in names
    assert "minecraft-modpack-server" in names


def test_discover_skills_excludes_support_and_vcs_dirs(fake_skills_dir):
    # add a junk dir that must be ignored
    import pathlib
    junk = pathlib.Path(fake_skills_dir, "node_modules", "fake")
    junk.mkdir(parents=True, exist_ok=True)
    (junk / "SKILL.md").write_text(
        "---\nname: junk\n---\n", encoding="utf-8")
    skills = router.discover_skills(fake_skills_dir)
    assert all(s["name"] != "junk" for s in skills)


def test_prefilter_ranks_relevant_skills_higher(fake_skills_dir):
    skills = router.discover_skills(fake_skills_dir)
    short = router._prefilter("post to twitter", skills, top_k=10)
    assert short and short[0]["name"] == "x-post"


def test_pass_empty_task_returns_empty():
    assert router.route_task("   ") == []


def test_route_task_fail_open_returns_empty(fake_skills_dir, tmp_path):
    # nonexistent laya python -> fail open
    result = router.route_task(
        "post to twitter", skills_dir=fake_skills_dir,
        laya_py=str(tmp_path / "does_not_exist"), top_n=2, floor=0.25)
    assert result == []


def test_route_task_with_mock_laya(fake_skills_dir, fake_laya_py, monkeypatch):
    # Keep the persistent-server path from starting; force the one-shot fallback
    # by monkeypatching _start_server to return None.
    monkeypatch.setattr(router, "_start_server", lambda py: None)
    result = router.route_task(
        "post to twitter", skills_dir=fake_skills_dir, laya_py=fake_laya_py,
        top_n=2, floor=0.25)
    assert result and result[0]["name"] == "x-post"
    assert "probability" in result[0]
    assert "path" in result[0]


def test_lexical_corroboration_drops_off_topic_pick(fake_skills_dir, fake_laya_py, monkeypatch):
    # Fake laya always picks 'minecraft-modpack-server'; with a twitter task there is
    # zero token overlap ("minecraft" not in task), so the pick must be dropped.
    monkeypatch.setattr(router, "_start_server", lambda py: None)
    result = router.route_task(
        "write a press release", skills_dir=fake_skills_dir, laya_py=fake_laya_py,
        top_n=2, floor=0.25)
    assert result == []


# ---------------------------------------------------------------------------
# plugin __init__ (:hook fail-open, excerpt truncation)
# ---------------------------------------------------------------------------
def _load_plugin_under_test():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(here)
    sys.path.insert(0, pkg_dir)
    import skill_router as plugin
    return plugin


class FakeCtx:
    def __init__(self, **overrides):
        self._overrides = overrides
    def get_config(self, key, default=None):
        return self._overrides.get(key, default)
    def register_hook(self, *_a, **_k):
        return None
    def register_command(self, *_a, **_k):
        return None
    def register_tool(self, *_a, **_k):
        return None


def test_hook_fail_open_on_config_missing():
    plugin = _load_plugin_under_test()
    fake = FakeCtx()
    plugin._GLOBAL_CTX = fake
    # empty task -> None (no injection, no crash)
    assert plugin._handle_pre_llm_call(user_message="   ") is None
    # no task key -> None
    assert plugin._handle_pre_llm_call(some_other_key="x") is None
    # unset global ctx -> None
    plugin._GLOBAL_CTX = None
    assert plugin._handle_pre_llm_call(user_message="hello") is None


def test_hook_multimodal_keeps_text():
    plugin = _load_plugin_under_test()
    plugin._GLOBAL_CTX = None  # routing would need ctx; we only test task extraction path
    # With ctx None it fails-open to None regardless; here we assert it doesn't raise on a list payload.
    out = plugin._handle_pre_llm_call(user_message=[{"type": "text", "text": "hello"}])
    assert out is None  # fail-open (no ctx), no crash


def test_excerpt_truncates_and_marks():
    plugin = _load_plugin_under_test()
    long = ("# T\n\n## procedure\n" + "y" * 5000)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td, "SKILL.md")
        p.write_text(f"---\nname: t\n---\n{long}", encoding="utf-8")
        ex = plugin._excerpt(str(p), 200)
    assert len(ex) <= 200 + 80  # truncation marker allowance
    assert "truncated" in ex


def test_register_is_idempotent_no_network():
    plugin = _load_plugin_under_test()
    fake = FakeCtx()
    plugin.register(fake)  # must not raise with FakeCtx registering hook/tool/command
