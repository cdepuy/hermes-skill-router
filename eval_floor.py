"""skill_router floor-sweep evaluation harness.

Goal: find the "optimal" confidence floor by measuring precision vs coverage
across a labeled set of realistic tasks, from ONE Laya pass per task (the full
`all` probability distribution), then re-thresholding at every floor value on
the stored distributions — no repeated model runs.

Methodology notes (be honest in the report):
 - Ground truth is a single human judgment per task (Chris's workflows). It is
   *evaluator* ground truth, not statistically labeled, so results are a
   directional guide, not an oracle benchmark.
 - We evaluate the TOP-1 pick (what the hook effectively injects first). The
   plugin's real behavior is top-N above floor with lexical corroboration; here
   we isolate the router's raw signal: is its #1 pick correct, and does raising
   the floor silence wrong picks faster than right ones?
 - "Correct" = top pick name in that task's accepted set.

Output: a table of floor -> precision (of predictions emitted, how many are
right) and coverage (of test turns, how many got a right top-pick), plus the
rate of turns where the router injects anything at all.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router

LAYA_PY = router.DEFAULT_LAYA_PY
LAYA_PREDICT = router.LAYA_PREDICT

# ---------------------------------------------------------------------------
# Labeled test set: (task, [accepted skill names]).
# Span the user's real daily workflows + a few distractors. Accepted sets are
# intentionally tight (the single best / the clearly-relevant skills).
# ---------------------------------------------------------------------------
TASKS = [
    ("draft a 650 Group blog post about optical components earnings",
     ["blog-post-creation", "optical-components-earnings-analysis", "chris-depuy-writing-style"]),
    ("write a formal 650 Group whitepaper as a PDF",
     ["whitepaper-authoring", "whitepaper-generation", "whitepaper-cover-generation"]),
    ("prepare the weekly 650 Group email newsletter draft",
     ["newsletter-handoff", "b2b-newsletter-design", "chris-depuy-writing-style"]),
    ("post an announcement to X/Twitter as cdepuy",
     ["x-post", "x-article-idea-briefing", "x-mcp-integration"]),
    ("write a chrisdepuy LinkedIn post about the market report",
     ["linkedin-post-automation", "linkedin-post-writing-style", "linkedin-draft-review"]),
    ("create a 650 Group press release",
     ["press-release-creation"]),
    ("diagnose why ssh to the remote server is failing",
     ["ssh-connection-fixer", "macos-lan-ssh", "windows-openssh-remote-door"]),
    ("set up a modded minecraft server for the fam",
     ["minecraft-modpack-server"]),
    ("transcribe this youtube video into a summary",
     ["transcription-workflow", "youtube-content"]),
    ("build an earnings revision chart for the report",
     ["earnings-revision-chart", "financial-data-refresh", "optical-components-earnings-analysis"]),
    ("pull the latest stock prices and market data",
     ["financial-data-refresh"]),
    ("serve deepseek 0731 on the DGX sparks with vllm",
     ["deepseek-dspark-spark-serve", "serving-llms-vllm", "dgx-spark-orchestration"]),
    ("check the weather for the event venue friday",
     ["weather", "air-quality-research"]),
    ("make a pull request to fix a bug in a github repo",
     ["github-pr-workflow", "github-code-review", "github-fork-contribution"]),
    ("send an email to a client",
     ["email", "mir-email"]),
    ("benchmark a local LLM with the lm-eval harness",
     ["evaluating-llms-harness"]),
    ("create a spotify playlist for the office",
     ["spotify"]),
    ("edit this video locally with ffmpeg",
     ["local-video-editing-ffmpeg"]),
    ("research a competitor company and write structured notes",
     ["company-research", "ir-notes", "competitive-intelligence-maintenance"]),
]


def run_laya_full(task: str) -> list[dict]:
    """Return the full probability distribution for this task's candidate pool.

    Uses the persistent Laya server (one model load for the whole sweep, then
    fast per-task inference) — mirrors production, and avoids 19 cold loads.
    """
    all_skills = router.discover_skills()
    candidates = router._prefilter(task, all_skills, top_k=router.PREFILTER_TOP_K)
    candidates = candidates[:12]
    if not candidates:
        return []
    payload = router.render_payload(task, candidates, top_n=12, floor=0.0)
    # persistent server first
    handle = router._start_server(LAYA_PY)
    data = None
    if handle is not None:
        data = router._server_predict(handle, payload)
    if data is None or data.get("error"):
        # fall back to a one-shot cold subprocess
        try:
            proc = subprocess.run(
                [LAYA_PY, LAYA_PREDICT], input=payload, capture_output=True,
                text=True, timeout=40, env={**os.environ, "USE_TF": "0"},
            )
            data = json.loads(proc.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return []
    if data.get("error"):
        return []
    out = {s["name"]: s for s in candidates}
    dist = []
    for item in (data.get("all") or []):
        name = str(item.get("skill", "")).replace("...", "")
        prob = float(item.get("probability", 0.0))
        if name in out:
            dist.append({"name": name, "probability": prob, "path": out[name]["path"]})
    return sorted(dist, key=lambda x: x["probability"], reverse=True)


def main() -> None:
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_raw.csv")
    rows = []
    print(f"Running {len(TASKS)} tasks through Laya (one pass each; ~seconds each)...")
    for i, (task, _accepted) in enumerate(TASKS, 1):
        dist = run_laya_full(task)
        rows.append({"task": task, "dist": dist})
        top = dist[0]["name"] if dist else "(none)"
        print(f"  [{i}/{len(TASKS)}] {top!r}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "dist_json"])
        for r in rows:
            w.writerow([r["task"], json.dumps(r["dist"])])
    print(f"\nSaved raw distributions to {out_csv}")

    # ---- Floor sweep over stored distributions ----
    print("\n=== Floor sweep (top-1 pick) ===")
    print(f"{'floor':>6} {'emit%':>6} {'prec':>6} {'cover':>6} {'right':>5} {'wrong':>5}")
    accepted_map = {t: a for t, a in TASKS}
    best = None
    for floor in [f / 100 for f in range(0, 101, 5)]:
        emitted = right = wrong = 0
        for r in rows:
            dist = r["dist"]
            if not dist:
                continue
            top = dist[0]
            if top["probability"] >= floor:
                emitted += 1
                if top["name"] in accepted_map[r["task"]]:
                    right += 1
                else:
                    wrong += 1
        n = len(TASKS)
        emit_pct = emitted / n * 100
        prec = right / emitted if emitted else float("nan")
        cover = right / n
        print(f"{floor:6.2f} {emit_pct:6.1f} {prec:6.3f} {cover:6.3f} {right:5d} {wrong:5d}")
        if emitted and cover > 0:
            # F0.5: precision-weighted F-score (prefers precision, resists collapse)
            f05 = (1.25 * prec * cover) / (0.25 * prec + cover + 1e-9)
            if best is None or f05 > best[1]:
                best = (floor, f05, prec, cover)
    if best:
        print(f"\nBest F0.5 floor = {best[0]:.2f}  (F0.5={best[1]:.3f}, prec={best[2]:.3f}, cover={best[3]:.3f})")


if __name__ == "__main__":
    main()
