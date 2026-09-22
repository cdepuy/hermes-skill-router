"""skill_router: a Hermes plugin that pre-routes each task to the right skill.

A Jev-style approach to skill selection. Instead of relying on the main model
to (sometimes) remember to load a skill mid-task, a small calibrated local
model (Laya, a 421M System-1 decision model) classifies the user's task
against the skill index *before* the main model sees it, and the selected
skill(s) are injected into the request as active guidance. Guarantees the
relevant skill is in context deterministically — no "session forgot the skill".

- Plugin kind: general (General plugin with a pre_llm_call hook).
- Cache-safe: injects via the user-message channel (returns {"context": ...}),
  never mutating the system prompt, so prompt caching is preserved.
- Fail-open: if Laya is missing/crashed or routing finds nothing, the hook
  returns nothing and the session behaves exactly as stock Hermes.

Config (plugins.entries.skill_router.settings.*):
  enabled:      bool  (default true)
  top_n:        int   (default 2)     skills to route
  floor:        float (default 0.05)  min P(route) to accept a skill
  skills_dir:   str   (default ~/.hermes/skills)
  laya_py:      str   (default ~/.hermes/workspace/laya-venv/bin/python)
  excerpt_chars:int   (default 3000)  per-skill body chars injected
  timeout_s:    float (default 30)

Floor note (from eval_floor.py, 19-task sweep): the confidence floor is a
NON-discriminating knob for this router — Laya is confidently wrong (mean conf
of wrong top-1 = 0.917 vs right = 0.894), so raising the floor costs coverage
without buying precision (F0.5 is flat 0.84 from floor 0 to 0.6). Keep it low
(~0) and rely on top_n=2 to hedge confident-wrong top-1 picks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import router as _router_mod

PLUGIN_IDENT = "skill_router"
_GLOBAL_CTX = None


def _cfg(ctx, key: str, default):
    try:
        return ctx.get_config(key, default)
    except Exception:
        return default


def _excerpt(path: str, max_chars: int) -> str:
    """First ~max_chars of a skill body (post-frontmatter), with a read hint."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Strip YAML frontmatter so the excerpt is actionable guidance, not metadata.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Cut at a markdown header near the cap so we never clip mid-sentence.
    cut = text[:max_chars]
    last_hd = max(cut.rfind("\n## "), cut.rfind("\n### "), cut.rfind("\n# "))
    if last_hd > 0:
        cut = cut[:last_hd]
    return cut.rstrip() + (
        "\n\n[... skill '%s' truncated — read the full file to continue]" % os.path.basename(os.path.dirname(path))
    )


def _routed_context(task: str, ctx) -> str:
    """Route task -> skill(s), load each, return the injected context block.

    Uses the SemIf engine by default (engine=config 'engine', default 'semif').
    The AVAILABILITY GATE (semif_probe) runs first: if the 3080 box is
    unreachable, the model isn't warmed, or the GPU is busy, routing returns []
    -> no injection -> the session behaves exactly as stock Hermes (fail-open).
    """
    enabled = _cfg(ctx, "enabled", True)
    if not enabled:
        return ""
    top_n = int(_cfg(ctx, "top_n", 2))
    floor = float(_cfg(ctx, "floor", 0.05))
    skills_dir = _cfg(ctx, "skills_dir", os.path.join(os.path.expanduser("~"), ".hermes", "skills"))
    laya_py = _cfg(ctx, "laya_py", os.path.join(
        os.path.expanduser("~"), ".hermes", "workspace", "laya-mlx-bench", ".venv", "bin", "python"))
    excerpt_chars = int(_cfg(ctx, "excerpt_chars", 3000))
    timeout_s = float(_cfg(ctx, "timeout_s", 30))
    engine = str(_cfg(ctx, "engine", "semif")).lower()

    if engine == "semif":
        # AVAILABILITY GATE is internal to route_task_semif: any miss -> [].
        routed = _router_mod.route_task_semif(
            str(task), skills_dir=skills_dir, top_n=top_n, floor=floor,
        )
    else:  # 'laya' (legacy) — fallback engine, direct call.
        routed = _router_mod.route_task(
            str(task), skills_dir=skills_dir, laya_py=laya_py,
            top_n=top_n, floor=floor, timeout=timeout_s,
        )
    if not routed:
        return ""

    blocks = []
    for r in routed:
        name = r["name"]
        body = _excerpt(r["path"], excerpt_chars)
        prob = r.get("probability")
        pct = f"{prob:.0%}" if isinstance(prob, (int, float)) else "?"
        blocks.append(
            f"ROUTED SKILL (auto-selected by skill_router, P={pct}): {name}\n"
            f"--- begin {name} excerpt ---\n{body}\n--- end {name} excerpt ---"
        )
    return (
        "skill_router injected the following skill(s) for this task. Treat their "
        "instructions as ACTIVE guidance for this turn; you do not need to call "
        "skill_view for these.\n\n" + "\n\n".join(blocks)
    )


def _handle_pre_llm_call(**kwargs):
    """pre_llm_call hook. Returns {"context": ...} injected into the user message
    (never the system prompt -> cache-safe), or None on failure (fail-open)."""
    try:
        task = kwargs.get("user_message")
        if isinstance(task, (list, tuple)):
            # multimodal content: keep the text part(s) as the routing signal
            task = " ".join(
                ch.get("text", "") for ch in task if isinstance(ch, dict) and ch.get("text"))
        if not task or not str(task).strip():
            return None
        ctx = _GLOBAL_CTX
        if ctx is None:
            return None
        context = _routed_context(str(task), ctx)
        if not context:
            return None
        return {"context": context}
    except Exception:
        # Fail-open: a routing error must never break the turn.
        return None


def _route_tool(args, ctx):
    task = (args or {}).get("task", "")
    out = _routed_context(task, ctx)
    if not out:
        return json.dumps({"routed": [], "note": "no skill routed (fail-open)"})
    names = [
        ln.split("):")[0].split(":", 1)[-1].strip()
        for ln in out.splitlines() if ln.startswith("ROUTED SKILL")
    ]
    return json.dumps({"routed": names, "context": out})


def _cmd_status(raw_args: str = "") -> str:
    cfg = {
        "enabled": _cfg(_GLOBAL_CTX, "enabled", True),
        "top_n": _cfg(_GLOBAL_CTX, "top_n", 2),
        "floor": _cfg(_GLOBAL_CTX, "floor", 0.05),
        "excerpt_chars": _cfg(_GLOBAL_CTX, "excerpt_chars", 3000),
    }
    return "\n".join(f"{k}: {v}" for k, v in cfg.items())


def _cmd_warm(raw_args: str = "") -> str:
    """/skill-router-warm — preload the SemIf model on the 3080 box (blocks)."""
    try:
        ok = _router_mod.semif_warm()
        return ("SemIf model warmed OK on 3080." if ok
                else "SemIf warm FAILED (box down / model load error). Routing stays off until warmed.")
    except Exception as e:
        return "SemIf warm ERROR: %s" % e


def register(ctx):
    """Register the skill_router plugin: pre_llm_call hook + a route tool."""
    global _GLOBAL_CTX
    _GLOBAL_CTX = ctx

    ctx.register_hook("pre_llm_call", _handle_pre_llm_call)

    # In-session slash command: /skill-router-status
    ctx.register_command(
        "skill-router-status",
        _cmd_status,
        description="Show skill_router config",
        args_hint="",
    )

    ctx.register_command(
        "skill-router-warm",
        _cmd_warm,
        description="Preload the SemIf model on the 3080 box (blocks until warm)",
        args_hint="",
    )

    # A model-callable tool so the agent can route explicitly on demand.
    ctx.register_tool(
        name="skill_router_route",
        toolset="skills",
        schema={
            "name": "skill_router_route",
            "description": (
                "Route a task to the best-matching Hermes skill(s) and return them. "
                "Use when unsure which skill applies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task or question to route."},
                },
                "required": ["task"],
            },
        },
        handler=lambda args, **kw: _route_tool(args, ctx),
        check_fn=lambda: bool(_cfg(ctx, "enabled", True)),
    )
