# skill_router

A Hermes plugin that **pre-routes each task to the right skill** before the main
model sees it, using a small calibrated local model (SemIf Qwen3.5-4B by
default, or legacy Laya). It implements the idea from
[JEV TIP 001](https://x.com/parcadei/status/2101132368949993830):
stop spending your main model's tokens (and trust) on figuring out which skill
to load — classify the task against skill names + short descriptions, then load
the winner deterministically.

**Fixes the "session forgot the skill" failure**: the relevant skill's guidance
is injected as active context every turn, so the main model never has to
remember to call `skill_view()`.

## What it does

- On each new user turn, the plugin's `pre_llm_call` hook runs a tiny routing
  model (SemIf on the 3080 box, or legacy Laya) against the ~300-skill index.
- It classifies the task, picks the top 1-2 skills, reads their `SKILL.md`, and
  injects a compact excerpt as the user-message context.
- Fail-open: if the engine is unavailable, busy, or routing finds nothing, the
  hook injects nothing and Hermes behaves exactly as stock.

## Cache-safe by design

The skill is injected through the **user-message channel** (`pre_llm_call`
returns `{"context": ...}`), never by mutating the system prompt. This preserves
Hermes's byte-stable prompt-cache invariant — only the ephemeral per-turn user
context changes, exactly like the built-in memory-prefetch path.

## Engines: SemIf (default) and Laya

Routing engine is config-driven (`engine`, default `semif`):

- **SemIf (default)** — `Qwen/Qwen3.5-4B` running on the win11 RTX 3080 box
  (`192.168.1.253`), direct-logit readout. Reads the task+options and returns
  usable graded probabilities (~0.2s warm after a ~100s one-time model load).
  Measured 67% top-1 vs Laya's 56% on the same pools, and never one-hots.
- **Laya (legacy)** — the original local 421M model at `laya_py`.

Both are wrapped by the same **availability gate** (below).

## AVAILABILITY GATE (the core of this version)

Before any routing work, the plugin probes whether the SemIf engine is actually
usable **this turn**. It checks, in order:

1. **Box reachable** — the 3080 answers over SSH; if not, unavailable.
2. **Server alive** — the `semif_server.py` stdio server is talking.
3. **Model warmed** — Qwen3.5-4B is loaded into VRAM (a cold model needs ~100s).
4. **GPU free** — `nvidia-smi` GPU util / memory within configured thresholds
   (defaults: util ≤ 40%, mem ≤ 16000MB), so we don't contend with a running
   render/LatentSync job.

If **any** check fails, the turn **reverts to normal session behavior — no skill
injection** (never worse than stock Hermes). The probe is fast (~0.3s; the
unreachable-box case ~1.5s), so it never stalls a normal turn.

Special case — **cold but reachable**: if the box is up and the GPU is free but
the model isn't loaded, the current turn reverts to normal (no stall) while a
**non-blocking background warm** loads the model for the *next* turn. You can
also pre-warm explicitly with `/skill-router-warm` (blocks ~15-100s) e.g. right
after a box reboot.

## Installation (no custom code for you)

The plugin lives in `~/.hermes/plugins/skill_router/` (user state, safe across
`hermes update`). Install once:

```bash
hermes plugins enable skill_router      # if not auto-enabled
```

SemIf needs the 3080 box's `semif_server.py` deployed (see
`SEMIF_BENCHMARK.md` / `semif-on-windows-3080` skill). Laya needs its local venv
(`~/.hermes/workspace/laya-venv/bin/python` with `pip install laya`).

Verify: `hermes plugins doctor skill_router` → `OK`.

## Configuration

Set under `plugins.entries.skill_router.settings` in `config.yaml`:

| key           | default | meaning |
|---------------|---------|---------|
| `enabled`     | `true`  | on/off |
| `engine`      | `semif` | routing engine: `semif` or `laya` |
| `top_n`       | `2`     | max skills to route |
| `floor`       | `0.05`  | min P(route) to accept a skill |
| `skills_dir`  | `~/.hermes/skills` | skill index location |
| `laya_py`     | `~/.hermes/workspace/laya-venv/bin/python` | Laya interpreter |
| `excerpt_chars` | `3000` | per-skill body chars injected |
| `timeout_s`   | `30`    | routing timeout |
| `semif_host` / `semif_port` / etc. | 192.168.1.253 / 2233 ... | SemIf box + SSH (env-overridable) |

SemIf box/SSH knobs default to env `SEMIF_*` or built-ins; thresholds are env
`SEMIF_GPU_UTIL_MAX` (40), `SEMIF_GPU_MEM_MAX_MB` (16000), timeouts
`SEMIF_PROBE_TIMEOUT_S` (1.5), `SEMIF_ROUTE_TIMEOUT_S` (3.0).

Check live config: `/skill-router-status` (slash) or
`hermes skill-router-status` (CLI).

## How routing works (2 stages)

1. **Lexical prefilter** — narrows ~300 skills to a candidate pool (top-12/16)
   using deterministic token overlap between the task and skill name/desc/category.
   Keeps the decision inside the engine's option cap and cheap.
2. **SemIf re-rank** — the candidate pool is posed as a single decision with the
   task as `state` and skills as `options`; SemIf reads option logits in one
   forward pass and returns per-option probability. Skills above `floor`, up to
   `top_n`, are loaded.

A **persistent SemIf server** (`semif_server.py` on the box, driven over a live
SSH session) keeps the model resident so routing is ~0.3s after the first turn
(first call triggers the ~100s model load, or `/skill-router-warm`).

## On-demand tool

The plugin also registers `skill_router_route` (toolset `skills`) — call it to
route an arbitrary task explicitly:

```
skill_router_route(task="post this announcement to X/Twitter")
```

## Honest limitations

- **Accuracy is ~good, not perfect.** SemIf is better than Laya (67% vs 56%
  top-1 on the benchmark pools) and returns unbiased confidence, but it still
  misroutes some meta/technical tasks. It's a helper, not an oracle; the main
  model still sees the task, and it's never worse than stock (fail-open).
- **Cold-start**: first routing needs the Qwen3.5-4B model loaded (~100s, or
  ~15s if the OS page cache has it). Run `/skill-router-warm` after a box reboot
  so interactive turns route immediately.
- **Depends on the 3080 box being up + free.** If it's off or mid-render, the
  availability gate reverts to stock Hermes (no routing) — by design.
- **Not "every tool AND skill on every prompt."** This guarantees the *right*
  skill is present; it intentionally does not dump all 300 skills.

## Development notes

Files:
- `__init__.py` — plugin entry: `register(ctx)`, `pre_llm_call` hook, tool, slash commands.
- `router.py` — skill discovery, lexical prefilter, SemIf client + availability gate, `route_task_semif()`.
- `semif_server.py` — stdio server (runs ON the 3080 box; reaches SemIf over SSH).
- `laya_server.py` / `laya_predict.py` — legacy Laya persistent/one-shot backends.
- `plugin.yaml` — manifest.

For an upstream contribution: this is a **general (standalone) plugin**, so it
lands via the plugin catalog / `~/.hermes/plugins/` rather than the core tree.
It does not touch `run_agent.py`, `cli.py`, or prompt assembly.
