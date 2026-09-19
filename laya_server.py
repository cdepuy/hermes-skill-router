#!/usr/bin/env python3
"""skill_router persistent Laya server.

Loads the Laya model ONCE at startup, then serves routing jobs over stdio:
  stdin  -> one JSON payload per line:  {"task": str, "skills": [{name,description},...],
                                         "top_n": int, "floor": float}
  stdout -> one JSON result per line:   {"top":[{skill,probability},...], "all":[...],
                                         "chosen": str}   (or {"error": str})

One model load (~20s), then each job is a fast inference (~1-3s). The router
spawns this once and keeps it alive, eliminating per-turn cold model loads.

Run (inside laya-venv):  USE_TF=0 python laya_server.py
"""
from __future__ import annotations

import json
import sys
import signal
import time

from laya_predict import build_questions

_MODEL = "convaiinnovations/laya"


def init_agent():
    import laya
    return laya.load(_MODEL)


def run_job(agent, payload: dict) -> dict:
    task = str(payload.get("task", ""))
    skills = payload.get("skills") or []
    top_n = int(payload.get("top_n", 2))
    floor = float(payload.get("floor", 0.05))
    if not task or not skills:
        return {"error": "task and skills required"}
    state = {"task": task}
    qs = {"route": build_questions(task, skills)}
    r = agent.predict(state, qs)
    route_ans = r["answers"]["route"]
    probs = route_ans.get("probabilities") or {}
    ranked = [{"skill": label, "probability": float(p)} for label, p in probs.items()]
    ranked.sort(key=lambda x: x["probability"], reverse=True)
    picked = [x for x in ranked if x["probability"] >= floor][:top_n]
    return {"top": picked, "all": ranked, "chosen": route_ans.get("choice")}


def main() -> int:
    # Exit cleanly on SIGTERM so the router can shut us down.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    agent = init_agent()
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            result = run_job(agent, payload)
        except Exception as exc:
            result = {"error": f"job failed: {exc}"}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
