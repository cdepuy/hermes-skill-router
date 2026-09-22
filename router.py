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
# The laya-mlx venv python (holds `laya_mlx`); Hermes' own interpreter does not.
DEFAULT_LAYA_PY = os.path.join(
    os.path.expanduser("~"), ".hermes", "workspace", "laya-mlx-bench", ".venv", "bin", "python"
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


# ============================================================================
# SemIf engine (win11 RTX 3080 box) + AVAILABILITY GATE
# ----------------------------------------------------------------------------
# The routing engine can be SemIf (Qwen3.5-4B, direct-logit readout) running
# as a persistent stdio server on the 3080 box (semif_server.py), reached
# over a LIVE SSH session (the Laya-server pattern, but remote).
#
# AVAILABILITY GATE: before doing ANY routing work we probe whether the box is
# reachable AND the SemIf model is loaded AND the GPU isn't busy. If any check
# fails, the gate returns "unavailable" and the caller skips routing entirely
# (revert to normal session behavior — NO skill injection). NEVER worse than
# not having the plugin.
# ============================================================================

SEMIF_HOST = os.environ.get("SEMIF_HOST", "192.168.1.253")
SEMIF_PORT = int(os.environ.get("SEMIF_PORT", "2233"))
SEMIF_SSH_USER = os.environ.get("SEMIF_SSH_USER", "wav2lip")
SEMIF_SSH_KEY = os.environ.get(
    "SEMIF_SSH_KEY", os.path.join(os.path.expanduser("~"), ".ssh", "id_wav2lip_win"))
SEMIF_BOX_PY = os.environ.get("SEMIF_BOX_PY", "C:/wav2lip/py311/python.exe")
SEMIF_BOX_SERVER = os.environ.get("SEMIF_BOX_SERVER", "C:/wav2lip/semif_server.py")
SEMIF_SYSPATH = os.environ.get("SEMIF_SYSPATH", "C:/wav2lip/semifpkg")
# Availability thresholds (nvidia-smi on the box).
SEMIF_GPU_UTIL_MAX = int(os.environ.get("SEMIF_GPU_UTIL_MAX", "40"))   # % GPU util allowed
SEMIF_GPU_MEM_MAX_MB = int(os.environ.get("SEMIF_GPU_MEM_MAX_MB", "16000"))  # box has 10240; use <10GB busy
# Probe/budget timeouts (seconds). A turn must not stall waiting on routing.
SEMIF_PROBE_TIMEOUT = float(os.environ.get("SEMIF_PROBE_TIMEOUT_S", "1.5"))
SEMIF_ROUTE_TIMEOUT = float(os.environ.get("SEMIF_ROUTE_TIMEOUT_S", "3.0"))
# Warm timeout: allow a cold-model first route more time (model must be loaded once).
SEMIF_COLD_TIMEOUT = float(os.environ.get("SEMIF_COLD_TIMEOUT_S", "180.0"))

# Persistent SemIf server handle (same shape as _SERVER for Laya).
_SEMIF_SERVER = None  # dict with proc/stdout/stdin
_SEMIF_SERVER_LOCK = None
if _SERVER_LOCK is not None:
    try:
        import threading as _tid
        _SEMIF_SERVER_LOCK = _tid.Lock()
    except Exception:
        _SEMIF_SERVER_LOCK = None


class SemIfUnavailable(Exception):
    """Raised when the SemIf engine is not available -> caller fails open."""


def _semif_ssh_cmd() -> list[str]:
    """Base SSH argv that reaches the box key-only, no host check."""
    return [
        "ssh", "-i", SEMIF_SSH_KEY, "-p", str(SEMIF_PORT),
        "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{SEMIF_SSH_USER}@{SEMIF_HOST}",
    ]


def _start_semif_server() -> dict | None:
    """Spawn the persistent SemIf server on the box over a live SSH session.

    Returns a handle (proc/stdout/stdin) or None on spawn failure. Does NOT
    wait for a handshake here (the server only speaks when asked) — the caller
    confirms liveness via `_semif_ping`, which carries its own timeout. A down
    box / dead ssh / server crash => None or a dead ping => unavailable.
    """
    global _SEMIF_SERVER
    if _SEMIF_SERVER is not None and _SEMIF_SERVER.get("proc") is not None \
            and _SEMIF_SERVER["proc"].poll() is None:
        return _SEMIF_SERVER
    # Build a remote command that sets SEMIF_SYSPATH then runs the server.
    # The box's embedded python already has torch 2.10/transformers 5.17 in
    # semifpkg; we inject the sys.path shim via env so semif_phase1 resolves.
    remote = (
        f"set SEMIF_SYSPATH={SEMIF_SYSPATH}&& "
        f"\"{SEMIF_BOX_PY}\" \"{SEMIF_BOX_SERVER}\""
    )
    ssh = _semif_ssh_cmd()
    if len(ssh) and "ssh" in ssh[0]:
        ssh.append(remote)
    else:
        ssh = ssh + [remote]
    try:
        proc = subprocess.Popen(
            ssh,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
    except (OSError, ValueError):
        return None
    assert proc.stdout is not None
    _SEMIF_SERVER = {"proc": proc, "stdout": proc.stdout, "stdin": proc.stdin}
    return _SEMIF_SERVER


def _readline_timeout(stream, timeout: float):
    """Read one line from a text stream, enforced by a deadline.

    Uses a thread to bound file.readline() (no portable non-blocking read on
    pipes in text mode). Returns the stripped line or None on timeout/EOF.
    """
    import threading as _t
    import queue as _q
    result = _q.Queue()

    def _reader():
        try:
            line = stream.readline()
        except Exception:
            line = None
        result.put(line)

    thr = _t.Thread(target=_reader, daemon=True)
    thr.start()
    try:
        line = result.get(timeout=timeout)
    except _q.Empty:
        return None
    if not line:
        return None
    line = line.strip()
    return line if line else None


def _semif_ping(handle) -> dict | None:
    """Send a ping to the live SemIf server; return response dict or None."""
    stdin, stdout = handle.get("stdin"), handle.get("stdout")
    if stdin is None or stdout is None:
        return None
    try:
        stdin.write('{"cmd": "ping"}\n')
        stdin.flush()
        for _ in range(3):
            line = _readline_timeout(stdout, SEMIF_PROBE_TIMEOUT)
            if line is None:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except (BrokenPipeError, OSError, ValueError):
        return None
    return None


def _semif_route(handle, task, options, max_tokens=4096):
    """Send one route job to the SemIf server; return response dict or None."""
    stdin, stdout = handle.get("stdin"), handle.get("stdout")
    if stdin is None or stdout is None:
        return None
    req = {"cmd": "route", "id": "route-0", "task": task,
           "options": options, "max_tokens": max_tokens}
    try:
        stdin.write(json.dumps(req) + "\n")
        stdin.flush()
        for _ in range(5):
            line = _readline_timeout(stdout, SEMIF_ROUTE_TIMEOUT)
            if line is None:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except (BrokenPipeError, OSError, ValueError):
        return None
    return None


def semif_probe() -> dict:
    """Probe SemIf availability. Returns a status dict; never raises.

    Checks, in order:
      1. box reachable + server alive (start/handshake)   -> if not, unavailable
      2. model_warm? (a warmed model is required to route without a long stall)
      3. GPU not busy? (gpu_util/mem within thresholds)
    Any failure -> {"available": False, "reason": ...}.
    """
    lock = _SEMIF_SERVER_LOCK
    if lock is not None:
        lock.acquire()
    try:
        try:
            handle = _start_semif_server()
        except Exception as e:
            return {"available": False, "reason": "start: %s" % e}
        if handle is None:
            return {"available": False, "reason": "box/server unreachable"}
        pong = _semif_ping(handle)
        if not pong or pong.get("error"):
            return {"available": False,
                    "reason": "no ping: %s" % ((pong or {}).get("error") or "no response")}
        if not pong.get("model_warm", False):
            return {"available": False, "reason": "model not warmed"}
        util = pong.get("gpu_util")
        mem = pong.get("gpu_mem_used_mb")
        if util is not None and util > SEMIF_GPU_UTIL_MAX:
            return {"available": False,
                    "reason": "gpu busy: util=%d%%" % util}
        if mem is not None and mem > SEMIF_GPU_MEM_MAX_MB:
            return {"available": False,
                    "reason": "gpu busy: mem=%dMB" % mem}
        return {"available": True, "model_warm": True, "gpu_util": util, "gpu_mem_used_mb": mem}
    finally:
        if lock is not None:
            lock.release()


_SEMIF_ALLOWED_WIDTH = 200  # large option count guard (SemIf caps at 16)

# A recent cold-but-reachable probe: so we can background-warm asynchronously.
_BG_WARM_STARTED = False


def _semif_warm_sync() -> dict:
    """Tell the server to load the model now (BLOCKS up to COLD_TIMEOUT)."""
    handle = _start_semif_server()
    if handle is None:
        return {"warmed": False, "error": "box/server unreachable"}
    stdin, stdout = handle.get("stdin"), handle.get("stdout")
    if stdin is None or stdout is None:
        return {"warmed": False, "error": "no stream"}
    try:
        stdin.write('{"cmd": "warm"}\n')
        stdin.flush()
        for _ in range(3):
            line = _readline_timeout(stdout, SEMIF_COLD_TIMEOUT)
            if line is None:
                return {"warmed": False, "error": "no response"}
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except (BrokenPipeError, OSError, ValueError) as e:
        return {"warmed": False, "error": str(e)}
    return {"warmed": False, "error": "no response"}


def semif_warm(timeout: float | None = None) -> bool:
    """Synchronously warm the SemIf model (blocks until loaded or timeout).

    Returns True if the model is now warm. Non-destructive wrapper; used by the
    /skill-router-warm command and as a one-time pre-warm. Does NOT touch the
    interactive gate.
    """
    global _BG_WARM_STARTED
    old = SEMIF_COLD_TIMEOUT
    try:
        if timeout is not None:
            globals()["SEMIF_COLD_TIMEOUT"] = timeout
        res = _semif_warm_sync()
        return bool(res.get("warmed"))
    finally:
        globals()["SEMIF_COLD_TIMEOUT"] = old
        _BG_WARM_STARTED = False


def _background_warm():
    """Non-blocking warm: load the model so the NEXT turn can route."""
    global _BG_WARM_STARTED
    import threading
    if _BG_WARM_STARTED:
        return
    _BG_WARM_STARTED = True
    try:
        thr = threading.Thread(target=_semif_warm_sync, daemon=True)
        thr.start()
    except Exception:
        _BG_WARM_STARTED = False


def discover_semif_route(task: str, skills: list[dict], top_n: int = 2,
                         floor: float = 0.01) -> list[dict]:
    """Route `task` through SemIf against `skills` (already-narrowed pool).

    Returns [{skill, probability}] top-N above floor, ordered by P.
    SemIf returns per-option probabilities (NOT one-hot) so a floor is usable.
    Raises SemIfUnavailable if the engine can't be reached (caller fails open).
    """
    # SemIf caps at 16 options per decision; narrow to that (caller pre-filters).
    options = [{"id": s["name"], "description": s["description"]} for s in skills][:16]
    if len(skills) > 16:
        raise SemIfUnavailable("too many options for SemIf: %d" % len(skills))
    handle = _semif_server_or_raise()
    res = _semif_route(handle, task, options)
    if not res or res.get("error"):
        raise SemIfUnavailable("semif route failed: %s" % ((res or {}).get("error") or "no response"))
    probs = {oid: p for oid, p in zip(res.get("option_ids", []), res.get("probabilities", []))}
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    out = []
    for name, p in ranked:
        if p < floor:
            continue
        out.append({"skill": name, "probability": p})
        if len(out) >= top_n:
            break
    return out


def _semif_server_or_raise() -> dict:
    handle = _start_semif_server()
    if handle is None:
        raise SemIfUnavailable("box/server unreachable")
    return handle


def route_task_semif(
    task: str,
    *,
    skills_dir: str | None = None,
    top_n: int = 2,
    floor: float = 0.01,
) -> list[dict]:
    """SemIf-engine route with the AVAILABILITY GATE in front.

    Returns [] on any unavailability/failure (revert to normal session — no
    skill injection). NEVER raises into the hook (there it is caught anyway).
    """
    try:
        if not task or not task.strip():
            return []
        probe = semif_probe()
        if not probe.get("available"):
            # EDGE CASE: box reachable + GPU free but model cold — the turn
            # reverts to normal (no stall), but kick a NON-BLOCKING background
            # warm so the NEXT turn can route. Never blocks this turn.
            if probe.get("reason") == "model not warmed":
                try:
                    _background_warm()
                except Exception:
                    pass
            return []
        all_skills = discover_skills(skills_dir)
        if not all_skills:
            return []
        candidates = _prefilter(task, all_skills, top_k=PREFILTER_TOP_K)
        if len(candidates) > 16:
            candidates = candidates[:16]
        if not candidates:
            return []
        picks = discover_semif_route(task, candidates, top_n=top_n, floor=floor)
        name_to_skill = {s["name"]: s for s in candidates}
        result = []
        for p in picks:
            name = str(p.get("skill", "")).replace("...", "")
            if name not in name_to_skill:
                continue
            result.append({
                "name": name,
                "probability": p.get("probability"),
                "path": name_to_skill[name]["path"],
                "category": name_to_skill[name]["category"],
            })
        return result
    except Exception:
        return []


def semif_available() -> bool:
    """Quick public availability check used by status/CLI. Fail-open -> False."""
    try:
        return bool(semif_probe().get("available"))
    except Exception:
        return False


if __name__ == "__main__":
    # Manual test driver: route a quick task against the real index.
    t = sys.argv[1] if len(sys.argv) > 1 else "draft a 650 blog post about optical components earnings"
    import time
    engine = sys.argv[2] if len(sys.argv) > 2 else "semif"
    s = time.time()
    if engine == "semif":
        ans = route_task_semif(t)
    else:
        ans = route_task(t)
    print(f"task: {t!r}  ({len(ans)} routed in {time.time()-s:.2f}s)  [engine={engine}]")
    for a in ans:
        print(f"  {a['probability']:.2f}  {a['name']}  [{a['category']}]")
