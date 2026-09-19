# skill_router

A Hermes plugin that **pre-routes each task to the right skill** before the main
model sees it, using a small calibrated local model (Laya, a 421M System-1
decision model). It implements the idea from [JEV TIP 001](https://x.com/parcadei/status/2101132368949993830):
stop spending your main model's tokens (and trust) on figuring out which skill
to load — classify the task against skill names + short descriptions, then load
the winner deterministically.

**Fixes the "session forgot the skill" failure**: the relevant skill's guidance
is injected as active context every turn, so the main model never has to
remember to call `skill_view()`.

## What it does

- On each new user turn, the plugin's `pre_llm_call` hook runs a tiny routing
  model (Laya, local, `$0`) against the ~300-skill index.
- It classifies the task, picks the top 1-2 skills, reads their `SKILL.md`, and
  injects a compact excerpt as the user-message context.
- Fail-open: if Laya is unavailable, missing, or routing finds nothing, the hook
  injects nothing and Hermes behaves exactly as stock.

## Cache-safe by design

The skill is injected through the **user-message channel** (`pre_llm_call`
returns `{"context": ...}`), never by mutating the system prompt. This preserves
Hermes's byte-stable prompt-cache invariant — only the ephemeral per-turn user
context changes, exactly like the built-in memory-prefetch path.

## Installation (no custom code for you)

The plugin lives in `~/.hermes/plugins/skill_router/` (user state, safe across
`hermes update`). Install once:

```bash
hermes plugins enable skill_router      # if not auto-enabled
```

It requires the local Laya venv (already present on this machine):
`~/.hermes/workspace/laya-venv/bin/python` with `pip install laya`.

Verify: `hermes plugins doctor skill_router` → `OK`.

## Configuration

Set under `plugins.entries.skill_router.settings` in `config.yaml`:

| key           | default | meaning |
|---------------|---------|---------|
| `enabled`     | `true`  | on/off |
| `top_n`       | `2`     | max skills to route |
| `floor`       | `0.25`  | min P(route) to accept a skill |
| `skills_dir`  | `~/.hermes/skills` | skill index location |
| `laya_py`     | `~/.hermes/workspace/laya-venv/bin/python` | Laya interpreter |
| `excerpt_chars` | `3000` | per-skill body chars injected |
| `timeout_s`   | `30`    | routing timeout |

Check live config: `/skill-router-status` (slash) or
`hermes skill-router-status` (CLI).

## How routing works (2 stages)

1. **Lexical prefilter** — narrows ~300 skills to a candidate pool (top-12)
   using deterministic token overlap between the task and skill name/desc/category.
   Keeps the Laya call inside its option cap and cheap.
2. **Laya re-rank** — the candidate pool is posed as a single calibrated
   `choice` question; Laya returns a per-option probability. Skills above
   `floor`, up to `top_n`, are loaded. Requires the surfaced pick to share at
   least one task token (guards off-topic small-model picks).

A **persistent Laya server** (`laya_server.py`) keeps the model resident so
routing is ~0.1s after the first turn (first call ~20s model load).

## On-demand tool

The plugin also registers `skill_router_route` (toolset `skills`) — call it to
route an arbitrary task explicitly:

```
skill_router_route(task="post this announcement to X/Twitter")
```

## Honest limitations (v0.1)

- **Accuracy is ~good, not perfect.** A 421M System-1 model can misroute
  (observed e.g. a "post to twitter" task occasionally gravitating to a
  LinkedIn-posting skill). It's a helper, not an oracle; the injected text says
  "treat as ACTIVE guidance," and the main model still sees the task. It is
  never worse than stock (fail-open), but it can pick a *plausible-but-not-ideal*
  skill from time to time.
- **Cold-start latency**: the first routing call loads the 1.7 GB Laya model
  (~20s). Subsequent turns are ~0.1s while the server is alive.
- **Not "every tool AND skill on every prompt."** This guarantees the *right*
  skill is present; it intentionally does not dump all 300 skills (that would
  bloat context and defeat the purpose). Tool schemas are unchanged (Hermes
  already sends them every call).

## Development notes

Files:
- `__init__.py` — plugin entry: `register(ctx)`, `pre_llm_call` hook, tool, slash command.
- `router.py` — skill discovery, lexical prefilter, Laya client (persistent server + one-shot fallback), `route_task()`.
- `laya_predict.py` — one-shot Laya CLI (laya-venv interpreter).
- `laya_server.py` — persistent Laya stdio server (keeps model resident).
- `plugin.yaml` — manifest.

For an upstream contribution: this is a **general (standalone) plugin**, so it
lands via the plugin catalog / `~/.hermes/plugins/` rather than the core tree.
It does not touch `run_agent.py`, `cli.py`, or prompt assembly.
