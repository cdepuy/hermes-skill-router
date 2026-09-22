#!/usr/bin/env python3
"""skill_router Laya subprocess entry.

Runs INSIDE the laya-venv (Hermes' own interpreter lacks `laya`). Reads a task
+ a skill index on stdin (JSON), returns top-N skill names + P(choose) on
stdout (JSON). Used by the skill_router plugin's pre_llm_call hook; kept a
standalone CLI so the plugin never imports laya into its own process.

Usage:
  USE_TF=0 python laya_predict.py < payload.json
    payload.json: {"task": str, "skills": [{name, description}, ...], "top_n": int, "floor": float}

Example payload:
  {"task": "draft a 650 group blog post about optical components earnings",
   "skills": [{"name": "chris-depuy-writing-style", "description": "650 blog voice"},
              {"name": "optical-components-earnings-analysis", "description": "predict earnings season trends"}],
   "top_n": 2, "floor": 0.05}
"""
from __future__ import annotations
import json, os, sys


def build_questions(task: str, skills: list[dict]) -> dict:
    """One Laya `choice` question whose options are the candidate skills.

    Laya's choice schema: {"type": "choice", "instructions": str, "criteria": {LABEL: reason}}.
    Each candiate skill is a criterion. Laya returns a probability per option.
    """
    criteria = {}
    for s in skills:
        name = s["name"]
        desc = s.get("description", "")
        # Keep labels short but unique; fold description into the reason so the
        # router has signal beyond the bare name.
        label = name if len(name) <= 40 else name[:37] + "..."
        criteria[label] = f"{name}: {desc}"[:160]
    return {
        "type": "choice",
        "instructions": (
            "Given the USER TASK, which skill should be loaded to guide "
            "completing it? Rank the candidate skills. Select the one(s) most "
            "relevant to this task."
        ),
        "criteria": criteria,
    }


def main() -> int:
    import laya_mlx as laya  # laya-mlx port (same convaiinnovations/laya weights)
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(json.dumps({"error": f"bad stdin JSON: {exc}"}))
        return 2
    task = str(payload.get("task", ""))
    skills = payload.get("skills") or []
    top_n = int(payload.get("top_n", 2))
    floor = float(payload.get("floor", 0.05))
    if not task or not skills:
        print(json.dumps({"error": "task and skills required"}))
        return 2

    state = {"task": task}
    qs = {"route": build_questions(task, skills)}
    try:
        agent = laya.load("aac6fef/laya-mlx", dtype="float16")
        r = agent.predict(state, qs)
    except Exception as exc:
        print(json.dumps({"error": f"laya predict failed: {exc}"}))
        return 3

    # Laya `choice` answer shape: {"choice": <picked_label>, "probabilities": {label: P}}.
    route_ans = r["answers"]["route"]
    probs = route_ans.get("probabilities") or {}
    ranked = [{"skill": label, "probability": float(p)} for label, p in probs.items()]
    ranked.sort(key=lambda x: x["probability"], reverse=True)
    picked = [x for x in ranked if x["probability"] >= floor][:top_n]
    print(json.dumps({"top": picked, "all": ranked, "chosen": route_ans.get("choice")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
