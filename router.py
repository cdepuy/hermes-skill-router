"""skill_router: scan the Hermes skill index, surface it to Laya, pick top-N.

Pure/import-light by design. The plugin (__init__.py) owns the Hermes
pre_llm_call hook; this module owns:
  - discovering enabled SKILL.md files + their short descriptions
   (mirrors Hermes' own index: name + <=60-char description, category-first)
  - shelling out to laya_predict.py (running in the laya-venv) for a calibrated
    per-skill probability given the user's task
  - returning the top-N skill identifiers above a confidence floor

FAIL-OPEN: any error -> returns [] so an uninstalled/crashed model is never
worse than not having the plugin.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Where SKILL.md files live for this profile. Mirrors get_skills_dir().
DEFAULT_SKILLS_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "skills")
# The laya-venv python (holds `laya`); Hermes' own interpreter does not.
DEFAULT_LAYA_PY = os.path.join(
    os.path.expanduser("~"), ".hermes", "workspace", "laya-venv", "bin", "python"
)
# This module's own dir - laya_predict.py ships alongside it.
_HERE = os.path.dirname(os.path.abspath(__file__))
LAYA_PREDICT = os.path.join(_HERE, "laya_predict.py")
LAYA_SERVER = os.path.join(_HERE, "laya_server.py")

# Persistent Laya server: loaded once, reused across routing calls.
_SERVER = None  # {proc, stdout, stdin} once live
_SERVER_LOCK = None
try:
    import threading
    _SERVER_LOCK = threading.Lock()
except Exception:
    pass

_SKILL_MD = "SKILL.md"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_NAME_RE = re.compile(r"^name:\s*(.+)$", re.M)
_DESC_RE = re.compile(r"^description:\s*(.+)$", re.M)
_EXCLUDED_DIRS = {
    ".git", ".github", ".hub", ".archive", ".curator_backups",
    ".venv", "venv", "node_modules", "__pycache__", "_org",
}

DESC_LIMIT = 80
# Laya's choice-question head limits the number of options we can route among.
LAYA_MAX_OPTIONS = 100  # comfortably under head_max_len=192 once * name+desc text
# How many candidates to carry into the Laya re-rank after lexical prefilter.
PREFILTER_TOP_K = 80

# Very common words that add no routing signal.
_STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "and", "or", "on", "in", "with", "at",
    "by", "is", "are", "was", "were", "be", "it", "this", "that", "these",
    "those", "from", "as", "about", "i", "you", "we", "my", "your", "please",
    "can", "could", "would", "should", "do", "does", "did", "have", "has",
    "had", "will", "shall", "use", "using", "need", "want", "help", "me",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _prefilter(task: str, skills: list[dict], top_k: int = PREFILTER_TOP_K) -> list[dict]:
    """Deterministic lexical prefilter: task tokens vs skill name+desc tokens.

    Returns a shortlist (<= top_k) ordered by descending overlap. This narrows
    the full skill set down to a candidate pool that fits Laya's option cap,
    before the calibrated System-1 re-rank. Also always carries any skill whose
    category token appears in the task (e.g. 'writing', 'mlops').
    """
    if not skills:
        return []
    task_toks = _tokenize(task)
    if not task_toks:
        # No lexical signal: fall back to a spread across categories so Laya
        # still has a representative option set (always < cap).
        picked, seen = [], set()
        for s in sorted(skills, key=lambda x: x["category"]):
            if s["name"] not in seen:
                seen.add(s["name"]); picked.append(s)
                if len(picked) >= top_k:
                    break
        return picked

    scored = []
    for s in skills:
        text = " ".join([s.get("name", ""), s.get("category", ""), s.get("description", "")])
        toks = _tokenize(text)
        overlap = len(task_toks & toks)
        # Boost exact name hit and category hit
        if s.get("name", "").lower().replace("-", " ") in task.lower():
            overlap += 3
        if s.get("category", "") in task.lower():
            overlap += 2
        if overlap > 0:
            scored.append((overlap, s))
    scored.sort(key=lambda x: (-x[0], x[1]["name"]))
    return [s for _, s in scored[:top_k]]


def _parse_frontmatter(text: str) -> dict:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    fm = m.group(1)
    def first(pat):
        mm = re.search(pat, fm)
        return mm.group(1).strip().strip('"\'') if mm else ""
    return {"name": first(_NAME_RE), "description": first(_DESC_RE)}


def discover_skills(skills_dir: str | None = None) -> list[dict]:
    """All enabled SKILL.md files as [{name, description, path}], by category."""
    root = Path(skills_dir or DEFAULT_SKILLS_DIR)
    if not root.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(root.rglob(_SKILL_MD)):
        parts = p.relative_to(root).parts
        if any(seg in _EXCLUDED_DIRS for seg in parts):
            continue
        # Keep only name + first-line description for the router's "short
        # descriptions" input (the X-post idea: classify against names + short
        # descriptions). Category folded into name for disambiguation.
        try:
            fm = _parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        name = fm.get("name") or p.parent.name
        category = parts[0] if len(parts) >= 2 else "general"
        desc = (fm.get("description") or "").strip()
        if len(desc) > DESC_LIMIT:
            desc = desc[: DESC_LIMIT - 3] + "..."
        out.append({
            "name": name,
            "category": category,
            "description": desc,
            "path": str(p),
            ## provide a bare-name option too (some skills are ambiguous)
        })
    return out


def render_payload(task: str, skills: list[dict], top_n: int, floor: float) -> str:
    """JSON string for laya_predict.py stdin. Only name+description go to Laya
    (keeps the prompt tiny and the decision fast), path stays for later load."""
    return json.dumps({
        "task": task,
        "skills": [{"name": s["name"], "description": s["description"]} for s in skills],
        "top_n": top_n,
        "floor": floor,
    })


def _start_server(laya_py: str) -> dict | None:
    """Spawn the persistent Laya server; return a live handle or None."""
    global _SERVER
    if _SERVER is not None and _SERVER.get("proc") is not None and _SERVER["proc"].poll() is None:
        return _SERVER
    if not os.path.exists(laya_py):
        return None
    try:
        proc = subprocess.Popen(
            [laya_py, LAYA_SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env={**os.environ, "USE_TF": "0"},
        )
    except (OSError, ValueError):
        return None
    assert proc.stdout is not None
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            if json.loads(line).get("ready"):
                _SERVER = {"proc": proc, "stdout": proc.stdout, "stdin": proc.stdin}
                return _SERVER
        except json.JSONDecodeError:
            continue
    try:
        proc.terminate()
    except Exception:
        pass
    return None


def _server_predict(handle: dict, payload: str) -> dict | None:
    """Send one job to the persistent server and read the single result line."""
    stdin, stdout = handle.get("stdin"), handle.get("stdout")
    if stdin is None or stdout is None:
        return None
    try:
        stdin.write(payload + "\n")
        stdin.flush()
        for _ in range(3):
            line = stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except (BrokenPipeError, OSError, ValueError):
        return None
    return None


def _run_laya(task: str, skills: list[dict], top_n: int, floor: float, laya_py: str | None,
              timeout: float = 30.0) -> list[dict]:
    """Call Laya; prefers the persistent server (model loaded once), falls back
    to a one-shot cold subprocess. Returns top picks [{skill, probability}]."""
    py = laya_py or DEFAULT_LAYA_PY
    if not os.path.exists(py):
        return []
    payload = render_payload(task, skills, top_n, floor)

    # 1) Persistent server (fast path) — reuses a live model.
    lock = _SERVER_LOCK
    if lock is not None:
        with lock:
            handle = _start_server(py)
            if handle is not None:
                data = _server_predict(handle, payload)
                if data is not None and not data.get("error"):
                    top = data.get("top") or []
                    if isinstance(top, list):
                        return [t for t in top if isinstance(t, dict)]

    # 2) One-shot fallback (cold subprocess).
    try:
        proc = subprocess.run(
            [py, LAYA_PREDICT],
            input=payload, capture_output=True, text=True, timeout=timeout, check=False,
            env={**os.environ, "USE_TF": "0"},
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return []
    if data.get("error"):
        return []
    top = data.get("top") or []
    if not isinstance(top, list):
        return []
    return [t for t in top if isinstance(t, dict)]


def route_task(
    task: str,
    *,
    skills_dir: str | None = None,
    laya_py: str | None = None,
    top_n: int = 2,
    floor: float = 0.05,
    timeout: float = 30.0,
) -> list[dict]:
    """Task -> top-N routed skills [{name, probability, path}].

    FAIL-OPEN: empty list on any failure. A down/Cay-missing model means the
    plugin injects nothing and the session behaves exactly as stock Hermes.
    """
    if not task or not task.strip():
        return []
    all_skills = discover_skills(skills_dir)
    if not all_skills:
        return []
    # Narrow 300+ skills to a candidate pool that fits Laya's option cap.
    candidates = _prefilter(task, all_skills, top_k=PREFILTER_TOP_K)
    if len(candidates) > LAYA_MAX_OPTIONS:
        candidates = candidates[:LAYA_MAX_OPTIONS]
    if not candidates:
        return []
    # Laya one-hots with large pools: route on a TIGHTER pool so it returns a
    # calibrated, useful distribution (top-12 keeps candidate quality, Laya's
    # estimation is more stable, and latency stays ~seconds).
    if len(candidates) > 12:
        candidates = candidates[:12]
    picks = _run_laya(task, candidates, top_n, floor, laya_py, timeout)
    name_to_skill = {s["name"]: s for s in candidates}
    # Lexical corroboration: drop any Laya pick with zero token overlap of the
    # task (guards against off-topic small-model picks). Best-effort; an empty
    # task-token set (no signal) keeps picks as-is.
    task_toks = _tokenize(task)
    result = []
    for p in picks:
        name = str(p.get("skill", "")).replace("...", "")
        prob = p.get("probability")
        if name not in name_to_skill:
            continue
        if task_toks:
            cand_text = " ".join([
                name_to_skill[name]["name"], name_to_skill[name]["category"],
                name_to_skill[name]["description"],
            ])
            if not (_tokenize(cand_text) & task_toks):
                continue
        result.append({
            "name": name,
            "probability": prob,
            "path": name_to_skill[name]["path"],
            "category": name_to_skill[name]["category"],
        })
    return result


if __name__ == "__main__":
    # Manual test driver: route a quick task against the real index.
    t = sys.argv[1] if len(sys.argv) > 1 else "draft a 650 blog post about optical components earnings"
    import time
    s = time.time()
    ans = route_task(t)
    print(f"task: {t!r}  ({len(ans)} routed in {time.time()-s:.2f}s)")
    for a in ans:
        print(f"  {a['probability']:.2f}  {a['name']}  [{a['category']}]")
