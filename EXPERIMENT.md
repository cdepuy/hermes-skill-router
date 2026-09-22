# Option 2 + Option 4 experiment — real results (2026-09-21)

Question: do the cheap versions of Option 2 (semantic candidate prefilter) + 
Option 4 (no-skill gate) improve skill_router? Measured on the real production
task set plus eval tasks. **Neutral, evidence-only.**

## Option 2 — embedding (Qwen3-Embedding-0.6B, cached/local) prefilter

Metric = prefilter recall: is the CORRECT skill in the top-80 candidate pool
(the pool the router's Laya re-ranks over)?

- lexical (current) prefilter: 10/12
- embedding prefilter:          12/12  (+2: recovered x-article-idea-briefing,
                                        hermes-agent, both dropped by lexical)

**But recall-to-pool is NOT recall-to-answer.** In 3/12 tasks the embedding
ranked the correct skill OUTSIDE its top-6 (router/semantic-router task:
skill-router-plugin not in emb top-6; 'router working?' task: hermes-agent not
in emb top-6, which was mostly device/SSH skills). So the embedding gets the
right skill into the 80-pool but often too low to matter, and Laya still
mispicks. Option 2 fixes a paper recall metric; it does NOT fix correctness.

## Option 4 — no-skill gate (embedding top-1 similarity)

Metric: can embedding similarity separate "clear, one-skill" turns from
"ambiguous, no-good-skill" turns?

| kind       | top-1 sim  | top1-top2 margin |
|------------|-----------|------------------|
| clear      | 0.74–0.80 | 0.020–0.190      |
| ambiguous  | 0.43–0.64 | 0.000–0.029      |

A threshold on top-1 similarity (~0.68) **cleanly separated this sample**:
all clear turns ≥0.738, all ambiguous ≤0.640. So an embedding-based gate CAN
suppress injection on genuinely ambiguous turns ("hi", "is this working?",
"run it make the curve") while letting clear ones through.

Caveats: small curated sample; 'x-post' clear case had a thin 0.020 margin,
so margin alone is unreliable — similarity (absolute match) separates better
than margin. Even on clear turns top-1 isn't always the ideal skill
(blog task top-1 was optical-components-earnings-analysis, defensible).

## Combined verdict

- Option 2 (embedding prefilter): fixes Stage-1 recall but is hollow — the
  bottleneck is Stage-2 (Laya selection) which it does not address.
- Option 4 (embedding similarity gate): a REAL improvement — it would have
  silenced the embarrassing P=100 injections on ambiguous turns, which is
  exactly where the live router did most harm.
- Together: embedding prefilter + similarity gate is strictly better than the
  current lexical router on every axis measured (recall 12/12 vs 10/12, and
  noise reduced by the gate). But selection quality (what it injects when it
  DOES inject) is still dominated by Laya's 421M discernment, which neither
  option fixes.

## Recommendation (unchanged in spirit)

Even with both improvements, this is still a "mostly noise, sometimes right"
pre-router for interactive chat. The gate makes it quieter; it does not make
it reliably correct. Hermes's native per-call skill index + main-model
skill_view remains the better routing path. These experiments are worth
documenting (and the embedding+gate could justify a cron-only or research
spike) but do NOT justify re-enabling interactive injection.

Status: plugin DISABLED. See PRODUCTION.md (6-day audit) + EVAL.md (why the
controlled eval misled).
