# skill_router floor evaluation — findings (2026-09-19)

Question asked (Chris): "is there a way to know the optimal value [for the
confidence floor]?" Answer: run a labeled sweep and measure precision vs
coverage at every floor. This is that sweep.

## Method

- **19 realistic tasks** spanning Chris's actual daily workflows (blog,
  whitepaper, newsletter, X-post, LinkedIn, press release, ssh, minecraft,
  youtube, earnings chart, market data, DGX spark serve, weather, github PR,
  email, llm bench, spotify, video edit, competitor research).
- Each went through the **real** production path (`_prefilter` → Laya top-12 pool),
  once per task. The full per-skill probability distribution was captured
  (`all`), then **re-thresholded at every floor 0.00–1.00** against the stored
  distributions — no repeated model runs.
- Ground truth: evaluator-judged accepted skill set per task (tight; later
  relaxed to accept umbrella skills that genuinely cover the task — `github`
  for the PR task, `pdf` for whitepaper-as-PDF).
- Metric: **top-1 correctness** gated by the floor, reported as precision
  (of emitted picks, how many right) and coverage (of tasks, how many right).

## Results (fair labeling)

| floor | emit% | prec | cover | F0.5 |
|---|---|---|---|---|
| 0.00–0.35 | 100 | 0.842 | 0.842 | 0.842 |
| 0.45–0.55 | 95 | 0.833 | 0.789 | 0.824 |
| 0.60–0.70 | 84 | 0.875 | 0.737 | 0.843 |
| 0.75–0.80 | 79 | 0.867 | 0.684 | 0.823 |
| 0.85–0.90 | 74 | 0.857 | 0.632 | 0.800 |
| 0.95 | 68 | 0.846 | 0.579 | 0.775 |
| 1.00 | 53 | 0.900 | 0.474 | 0.763 |

**Best F0.5 = 0.843, tied at floor 0.00 and floor 0.60.**

## Verdict

**The confidence floor is a non-discriminating knob.** Precision never rises
above 0.90 even at floor 1.0 (which silences half the turns and collapses
coverage to 0.47). The reason is diagnostic:

- Correct top-1 picks: mean confidence **0.894**
- Wrong top-1 picks: mean confidence **0.917** — Laya is *more* confident
  when it's wrong. 4 of 5 early misroutes sat at confidence **1.0**.

A floor that's supposed to separate signal from noise can't work when the
noise is louder than the signal. Raising it chops correct picks and wrong
picks at the same rate.

## What IS the right setting

- **floor ≈ 0** (keep 0.05 default as a tie-break only). Raising it buys
  ~nothing and costs coverage.
- **top_n = 2** is the real hedge: it catches the correct skill even when
  top-1 is a confident-wrong pick (e.g. ssh task → top-1
  `remote-engine-swap-over-ssh` wrong, but the correct `ssh-connection-fixer`
  sits at #2).
- **Real accuracy ceiling: ~84% top-1 / ~90% at top-2** on this suite — set by
  the 421M model's comprehension, NOT by any threshold. If we want more, the
  lever is a bigger routing model or category-aware candidate selection, not
  floor tuning.

## Residual misroutes (the real errors, all at high confidence)

- chrisdepuy LinkedIn post → `blog-post-creation` (confusable: both "writing")
- ssh failing → `remote-engine-swap-over-ssh` (confusable: both "ssh/remote")
- spotify playlist → `office-app-automation` (pure comprehension miss)

## Artifacts

- `eval_floor.py` — the harness (re-runnable; outputs `eval_raw.csv`)
- `eval_raw.csv` — stored per-task distributions (floor-agnostic, so the
  sweep is re-derivable without re-running the model)

## Honest caveat

This is an evaluator-labeled 19-task set, not a statistically validated
benchmark. Results are a directional guide for THIS router, not a certified
score. Real production behavior also routes top-2 + lexical corroboration, so
live precision is a bit higher than the raw top-1 number here.
