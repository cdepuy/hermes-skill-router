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


# ---------------------------------------------------------------------------
# SemIf engine + AVAILABILITY GATE (mocked transport — no network/box needed)
# ---------------------------------------------------------------------------
def _mock_probe(available, reason=None, **extra):
    d = {"available": available, "reason": reason}
    d.update(extra)
    return d


def test_route_task_semif_unavailable_returns_empty(monkeypatch):
    # Box unreachable -> gate says unavailable -> [] (revert to normal, no injection).
    monkeypatch.setattr(router, "semif_probe",
                        lambda: _mock_probe(False, reason="box/server unreachable"))
    assert router.route_task_semif("post to twitter") == []


def test_route_task_semif_cold_model_returns_empty(monkeypatch):
    # Box reachable but model not warmed -> [].
    monkeypatch.setattr(router, "semif_probe",
                        lambda: _mock_probe(False, reason="model not warmed"))
    assert router.route_task_semif("post to twitter") == []


def test_route_task_semif_busy_gpu_returns_empty(monkeypatch):
    # GPU busy (util above threshold) -> [].
    monkeypatch.setattr(router, "semif_probe",
                        lambda: _mock_probe(False, reason="gpu busy: util=95%"))
    assert router.route_task_semif("post to twitter") == []


def test_route_task_semif_empty_task_returns_empty(monkeypatch):
    # Whitespace task never even probes -> [].
    monkeypatch.setattr(router, "semif_probe", lambda: _mock_probe(True))
    assert router.route_task_semif("   ") == []


def test_route_task_semif_available_routes(fake_skills_dir, monkeypatch):
    # Engine available -> routes through SemIf (mocked server returns probs).
    monkeypatch.setattr(router, "semif_probe",
                        lambda: _mock_probe(True, model_warm=True, gpu_util=5))
    # Fake the remote route: pick the skill sharing most tokens with the task.
    def fake_route(task, candidates, top_n=2, floor=0.01):
        def overlap(s):
            return len(set(s["name"].replace("-", " ").split())
                       & set(task.lower().replace("-", " ").split()))
        ranked = sorted(candidates, key=lambda s: -overlap(s))
        return [{"skill": ranked[0]["name"], "probability": 0.9}]
    monkeypatch.setattr(router, "discover_semif_route", fake_route)
    result = router.route_task_semif("post to twitter", skills_dir=fake_skills_dir)
    assert result and result[0]["name"] == "x-post"
    assert result[0]["probability"] == 0.9
    assert "path" in result[0]


def test_semif_probe_mocked_ping_happy(monkeypatch):
    monkeypatch.setattr(router, "_start_semif_server", lambda: {"proc": object(), "stdin": None, "stdout": object()})
    monkeypatch.setattr(router, "_semif_ping",
                        lambda h: {"ping": True, "model_warm": True, "gpu_util": 5, "gpu_mem_used_mb": 4000})
    p = router.semif_probe()
    assert p["available"] is True
    assert p["gpu_util"] == 5


def test_semif_probe_gpu_busy_threshold(monkeypatch):
    # GPU util above the configured max -> unavailable.
    monkeypatch.setattr(router, "_start_semif_server", lambda: {"proc": object(), "stdin": None, "stdout": object()})
    monkeypatch.setattr(router, "_semif_ping",
                        lambda h: {"ping": True, "model_warm": True, "gpu_util": 99, "gpu_mem_used_mb": 4000})
    p = router.semif_probe()
    assert p["available"] is False
    assert "busy" in p["reason"]


def test_semif_probe_dead_ping(monkeypatch):
    # Server up but no ping reply -> unavailable.
    monkeypatch.setattr(router, "_start_semif_server", lambda: {"proc": object(), "stdin": None, "stdout": object()})
    monkeypatch.setattr(router, "_semif_ping", lambda h: None)
    p = router.semif_probe()
    assert p["available"] is False


def test_readline_timeout_returns_none_on_dead_stream():
    # A stream that never produces a line -> None after timeout (no hang).
    import io
    import router as R
    assert R._readline_timeout(io.StringIO("hello\n"), 0.5) == "hello"
    assert R._readline_timeout(io.StringIO("\n"), 0.5) is None


def test_readline_timeout_bounds_wait():
    # A stream that blocks long must return None within ~timeout, not hang.
    import router as R
    import threading, time
    import io

    class Blocking(io.StringIO):
        def readline(self, *a):
            time.sleep(30)  # far exceeds the 0.3s budget
            return "late\n"

    t0 = time.time()
    got = R._readline_timeout(Blocking(), 0.3)
    el = time.time() - t0
    assert got is None
    assert el < 3.0  # bounded, not the 30s block


def test_discover_semif_route_ranks_and_respects_floor(monkeypatch):
    # SemIf returns {option_ids, probabilities}; discover_semif_route uses floor+top_n.
    monkeypatch.setattr(router, "_semif_server_or_raise", lambda: {"proc": object()})
    monkeypatch.setattr(router, "_semif_route",
                        lambda h, task, options, max_tokens=4096: {
                            "option_ids": ["a", "b", "c"],
                            "probabilities": [0.7, 0.2, 0.1],
                        })
    fake_skills = [
        {"name": "a", "description": "aaa"}, {"name": "b", "description": "bbb"},
        {"name": "c", "description": "ccc"},
    ]
    out = router.discover_semif_route("task", fake_skills, top_n=2, floor=0.15)
    assert out == [{"skill": "a", "probability": 0.7}, {"skill": "b", "probability": 0.2}]


def test_discover_semif_route_too_many_options_raises(monkeypatch):
    # SemIf caps at 16 options; >16 must fail closed via SemIfUnavailable.
    many = [{"name": "s%02d" % i, "description": "d"} for i in range(17)]
    import pytest as _pt
    with _pt.raises(router.SemIfUnavailable):
        router.discover_semif_route("task", many)


def test_discover_semif_route_server_fail_raises(monkeypatch):
    monkeypatch.setattr(router, "_semif_server_or_raise", lambda: {"proc": object()})
    monkeypatch.setattr(router, "_semif_route", lambda h, t, o, max_tokens=4096: {"error": "model load failed"})
    import pytest as _pt
    with _pt.raises(router.SemIfUnavailable):
        router.discover_semif_route("post to x", [{"name": "x-post", "description": "d"}])


def test_route_task_semif_cold_model_triggers_background_warm(monkeypatch):
    # Cold but reachable -> returns [] AND kicks a background warm (no stall).
    warmed = []
    monkeypatch.setattr(router, "semif_probe",
                        lambda: _mock_probe(False, reason="model not warmed"))
    monkeypatch.setattr(router, "_background_warm", lambda: warmed.append(True))
    assert router.route_task_semif("post to twitter") == []
    assert warmed == [True]


def test_route_task_semif_unreachable_does_not_warm(monkeypatch):
    # Unreachable -> [] and NO warm attempt.
    warmed = []
    monkeypatch.setattr(router, "semif_probe",
                        lambda: _mock_probe(False, reason="box/server unreachable"))
    monkeypatch.setattr(router, "_background_warm", lambda: warmed.append(True))
    assert router.route_task_semif("post to twitter") == []
    assert warmed == []


def test_semif_warm_wrapper(monkeypatch):
    # Warm success + failure map to bool.
    monkeypatch.setattr(router, "_semif_warm_sync", lambda: {"warmed": True})
    assert router.semif_warm() is True
    monkeypatch.setattr(router, "_semif_warm_sync", lambda: {"warmed": False, "error": "x"})
    assert router.semif_warm() is False


def test_background_warm_guards_duplicate(monkeypatch):
    # _background_warm only starts one thread at a time (idempotent guard).
    import router as R
    starts = []
    monkeypatch.setattr(R, "_BG_WARM_STARTED", False)
    # Replace the thread-target with a recorder WITHOUT actually starting a thread.
    monkeypatch.setattr(R, "_semif_warm_sync", lambda: starts.append(1))
    # Stub threading.Thread within router to just set the flag (no real thread).
    class _FakeThread:
        def __init__(self, *a, **k):
            pass
        def start(self):
            R._BG_WARM_STARTED = True
    monkeypatch.setattr(R, "threading", type("_thr", (), {"Thread": _FakeThread}))
    R._background_warm()
    R._background_warm()
    assert len(starts) == 1  # second call is guarded by _BG_WARM_STARTED
    R._BG_WARM_STARTED = False


