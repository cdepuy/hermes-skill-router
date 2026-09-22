#!/usr/bin/env python3
"""Evaluate the skill_router audit log accumulated over the cron window.

Reads ~/.hermes/skill_router_audit.jsonl (written by the plugin's pre_llm_call
hook) and produces an honest summary for the 3-day report:

  - how many routing decisions were logged (routed vs skip)
  - how many injected a skill vs reverted to stock (skip w/ reasons)
  - the top skills picked, by count
  - the benchmark 9-task set: if any of those exact tasks appeared, whether
    SemIf's top-1 matched the expected correct skill (the "6/9" check)
  - an overall routed score

This is a DATA report, not a labeled-accuracy guarantee: production tasks are
unlabeled, so "correct" is only measurable where a known ground-truth task
appeared (the benchmark tasks) or where the pick is self-evidently on-topic.
"""
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

AUDIT = os.path.expanduser("~/.hermes/skill_router_audit.jsonl")

# Ground-truth task -> correct skill for the benchmark 9-task set (so we can
# check the "6/9 times right" question against real traffic).
BENCHMARK = {
    "i'm still getting lots of nas-zip-drain failure messages. what should i do to fix it?": "smb-nas-mount-resiliency",
    "make the curve and find optimal floor value for the router": "skill-router-plugin",
    "compare SemanticRouter to Laya router approach": "skill-router-plugin",
    "figure if any of these are new uses or Laya / Jev type approaches": "jev-local-system-one-engine",
    "draft a 650 Group blog post about optical components earnings": "blog-post-creation",
    "write a formal 650 Group whitepaper as a PDF": "whitepaper-authoring",
    "post an announcement to X/Twitter as cdepuy": "x-post",
    "diagnose why ssh to the remote server is failing": "ssh-connection-fixer",
    "set up a modded minecraft server": "minecraft-modpack-server",
}


def _ts(ts):
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


def main():
    if not os.path.exists(AUDIT):
        print("AUDIT: no skill_router audit log found at %s (plugin may not have logged yet)." % AUDIT)
        print("(Plugin was just enabled; no routing data has accumulated.)")
        return

    events = []
    with open(AUDIT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue

    if not events:
        print("AUDIT: audit log exists but is empty (no routing decisions logged yet).")
        return

    routed = [e for e in events if e.get("event") == "routed"]
    skips = [e for e in events if e.get("event") == "skip"]
    tasks_with_skip = {e.get("task") for e in skips}

    print("=" * 62)
    print("skill_router — 3-DAY AUDIT REPORT")
    print("=" * 62)
    print("Window: %s → %s" % (_ts(events[0].get("ts")), _ts(events[-1].get("ts"))))
    print("Total routing decisions logged: %d" % len(events))
    print("  Routed (injected a skill): %d" % len(routed))
    print("  Skipped (reverted to stock): %d" % len(skips))
    print()

    # Skip reasons
    if skips:
        reasons = Counter(e.get("reason") or "unknown" for e in skips)
        print("Skip reasons:")
        for reason, n in reasons.most_common():
            print("  %-40s %d" % (reason, n))
        print()

    # Top skills picked
    if routed:
        pick_counts = Counter()
        for e in routed:
            for p in e.get("picks", []):
                pick_counts[p.get("skill")] += 1
        print("Most-injected skills (healthiest if on-topic):")
        for skill, n in pick_counts.most_common(8):
            print("  %-40s %d" % (skill, n))
        print()

    # Benchmark 9-task check ("6/9 times right")
    bench_hits = 0
    bench_found = 0
    bench_rows = []
    for task, correct in BENCHMARK.items():
        # find any routed event for this exact task text
        matches = [e for e in routed if e.get("task") == task]
        if not matches:
            continue
        bench_found += 1
        picks = matches[-1].get("picks", [])
        top1 = picks[0]["skill"] if picks else None
        ok = top1 == correct
        bench_hits += int(ok)
        bench_rows.append((task, correct, top1, [p.get("skill") for p in picks]))

    print("BENCHMARK 9-TASK CHECK (the '6/9' question):")
    if not bench_found:
        print("  None of the exact benchmark tasks appeared in real traffic")
        print("  -> cannot re-measure the 6/9 figure from production data this window.")
        print("     (6/9 was the measured SemIf top-1 on those 9 pools in the lab benchmark.)")
    else:
        for task, correct, top1, ordered in bench_rows:
            mark = "OK " if top1 == correct else "x"
            print("  [%s] '%s...' -> %s (correct=%s; picks=%s)" % (
                mark, task[:35], top1, correct, ordered))
        print("  Benchmark tasks seen: %d/%d, top-1 correct: %d/%d" % (
            bench_found, len(BENCHMARK), bench_hits, bench_found))
        pct = (bench_hits / bench_found * 100) if bench_found else 0
        print("  (extrapolated accuracy %.0f%% on traffic seen)" % pct)
    print()

    # Our 9 lab tasks minus what appeared in traffic
    missing = [t for t in BENCHMARK if not any(e.get("task") == t for e in routed)]
    print("Benchmark tasks NOT seen in traffic this window: %d" % len(missing))
    print()

    print("NOTE: this is a data report. Production tasks are unlabeled; accuracy is")
    print("only directly measurable on the benchmark tasks above. '6/9' was the lab")
    print("top-1 over the 9 benchmark pools (SemIf 6/9 vs Laya 5/9).")


if __name__ == "__main__":
    main()
