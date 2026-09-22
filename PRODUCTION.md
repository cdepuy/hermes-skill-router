# skill_router production evaluation — 6-day live review (2026-09-21)

Follow-up to EVAL.md. EVAL.md measured a controlled 19-task eval and claimed
~84-90% accuracy + "floor is a non-knob". Chris rightfully asked for proof from
LIVE production. This is that audit. **It overturns the eval.**

## Method

Queried `~/.hermes/state.db` `messages.api_content` (the actual text sent to
the model, which carries the injected "ROUTED SKILL ... P=NN% NAME" block) for
all rows in Sep 15-21 2026 containing the router's injection header. Parsed
every pick. This is the real, persisted production record.

## Volume

- **67 total router injections in the window**
  - **52 in cron sessions** (every scheduled job got a pick)
  - **15 in interactive Telegram sessions** (across 9 distinct sessions)
- Router first fired Sep 19 14:52 (cron); telegram firing burst Sep 20 21:42-21:56.
- Plugin was enabled the whole window; last firing Sep 21 18:06 (cron).

## Results

### Interactive Telegram (the case that matters): ~7-25% accuracy
Of 15 real interactive turns, **1 was strictly correct** (`smb-nas-mount-resiliency`
for the nas-zip-drain task). Even granting half-credit to defensible picks,
accuracy is ~25%. Most misses were **confident (P=100)** and often nonsensical:

- "is this setup working on local Hermes?" -> hermes-model-provider-setup (P=100)
- "run it, make the curve, find optimal" -> agent-validation-loop (P=100)
- "getting back on this, is it working? commands to run?" -> debugging-hermes-tui-commands (P=100)
- "Compare to Laya [x]" -> github-code-review (P=94)
- "Figure new uses of Laya/Jev" -> lmstudio-status (P=75)

### Cron: 52 injections, high "correct"-but-low-information
Cron picks mostly match the job (dgx job -> dgx-spark-management, linkedin job ->
linkedin-post-writing-style). But many are trivially correct: the task text literally
begins "The user has invoked the '<skill>' skill" — the router reads the answer off
the prompt. No intelligence demonstrated.

## Root cause (confirmed by re-running the router on the real tasks)

1. **PREFILTER RECALL FAILURE (stage 1).** A deterministic token-overlap prefilter
   narrows ~300 skills to a top-12 pool before Laya re-ranks. For 2 of 5 sampled
   failed tasks, the correct skill was **dropped from the pool** before Laya saw it
   (e.g. `hermes-model-provider-setup` and `x-article-idea-briefing` absent). This is
   not Laya's fault — the pool made a correct answer impossible.
2. **LAYA DISCERNMENT FAILURE (stage 2).** Even when the correct skill was IN the
   pool, Laya picked an unrelated one (nas task -> `camoufox-maintenance`).
3. **GRANULARITY / PREMISE FAILURE (architecture).** Short ambiguous Telegram turns
   ("is this working?", "hi") have **no clean single-skill answer**. Forcing exactly
   one pick per turn injects noise on most interactive turns. Some turns need zero
   skills, some need several; the router has no "none of the above" and no multi-skill
   discrimination.

The confidence floor is a red herring — already proven non-discriminating in EVAL.md,
and production confirms it (confidently wrong).

## Why EVAL.md misled us

The eval used curated, skill-shaped task phrasings ("draft a 650 blog post about
optical components") that map cleanly to a skill. Real interactive prompts are short,
conversational, context-heavy, and often about "is X working?" status — which is
genuinely ambiguous and usually needs NOTHING. The eval measured the router's best
case; production exposes its real, failure-dominated behavior.

## Recommendation

- **Disable interactive (telegram) pre-routing.** It is adding noise to most turns.
  Hermes already loads the full skill index + main-model skill_view; that is the
  correct routing mechanism for chat.
- Cron routing can stay IF wanted (self-referential picks = low value, but harmless).
- If skill pre-routing is ever improved, the fixes are: (a) a recall-complete
  candidate selector (embedding/vector, not lexical), (b) a bigger router model or
  the main model itself, (c) explicit "no skill needed" rejection and multi-skill
  support, (d) route on the semantic intent, not noisy raw message text.

Status: plugin DISABLED 2026-09-21 per Chris. This doc supersedes the optimistic
accuracy claims in EVAL.md.
