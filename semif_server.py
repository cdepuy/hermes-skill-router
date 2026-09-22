"""SemIf routing server (runs ON the win11 RTX 3080 box, reached over SSH).

A persistent stdio JSON-line server mirroring the Laya-server pattern, but for
SemIf (Qwen3.5-4B, direct-logit readout). The Mac plugin launches this over a
LIVE SSH session (`ssh wav2lip@BOX "python semif_server.py"`); the Mac writes
one JSON request per line on stdin and reads one JSON response per line on
stdout. The model loads ONCE (cold ~103s) and stays resident for warm ~0.2s
routing.

Protocol (JSON lines, one request per line, one response per line):
  {"cmd": "ping"}                  -> {"ping": true, "model_warm": bool,
                                        "gpu_util": int, "gpu_mem_used_mb": int}
  {"cmd": "route", "task": str,
   "options": [{"id": str, "description": str}, ...],   # 2-16 options
   "max_tokens": int}              -> {"id": ..., "option_ids": [...],
                                        "probabilities": [...], ...}
Any error -> {"error": str}. The server NEVER auto-loads the model on ping;
load happens on the first route (or an explicit {"cmd":"warm"}).

GPU-busy handling: the plugin probes with PING first and compares gpu_util /
gpu_mem_used against thresholds BEFORE sending a route. A route while the GPU
is busy with another task is the caller's risk (SemIf's model is already in
VRAM, so a concurrent render would contend).

Standalone warmup: `python semif_server.py warm` loads the model and exits 0
on success (returns nonzero on failure) — used by the Mac to pre-warm outside
the request path.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time

REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL = os.environ.get("SEMIF_MODEL", "Qwen/Qwen3.5-4B")
# Allow the embeddable-python sys.path shim to be injected.
if os.environ.get("SEMIF_SYSPATH"):
    sys.path.insert(0, os.environ["SEMIF_SYSPATH"])

_model = None
_tokenizer = None
_metadata = None
_load_started = False


def _nvidia():
    """Return (util_pct:int, mem_used_mb:int) or (None, None) if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        util, mem = [int(x) for x in out.split(",")]
        return util, mem
    except Exception:
        return None, None


def _load():
    global _model, _tokenizer, _metadata, _load_started
    if _model is not None:
        return True
    if _load_started:
        return False  # a load is already in flight (or died); don't stack
    _load_started = True
    from semif_phase1.core import load_causal_model
    _model, _tokenizer, _metadata = load_causal_model(
        MODEL, REV, device="cuda", dtype="bfloat16")
    return True


def _handle(req: dict) -> dict:
    cmd = req.get("cmd")
    if cmd == "ping":
        util, mem = _nvidia()
        return {"ping": True, "model_warm": _model is not None,
                "gpu_util": util, "gpu_mem_used_mb": mem}
    if cmd == "warm":
        ok = _load()
        return {"warmed": ok, "model": _metadata.get("source") if _metadata else None}
    if cmd == "route":
        if not _load():
            return {"error": "model load failed or already in flight"}
        from semif_phase1.direct import score as direct_score
        from semif_phase1.core import validate_row
        options = req.get("options") or []
        row = {
            "id": req.get("id", "route-0"),
            "state": req.get("task", ""),
            "question": "Which Hermes skill best handles this user task?",
            "options": options,
        }
        try:
            validate_row(row)
        except ValueError as e:
            return {"error": "invalid row: %s" % e}
        t0 = time.perf_counter()
        res = direct_score(_model, _tokenizer, row, _metadata,
                           max_tokens=int(req.get("max_tokens", 4096)))
        res["_wall"] = round(time.perf_counter() - t0, 4)
        return res
    return {"error": "unknown cmd: %s" % cmd}


def main():
    # Warmup mode: load the model and exit (nonzero on failure). Used by the
    # Mac to pre-warm the model outside the interactive path.
    if len(sys.argv) > 1 and sys.argv[1] == "warm":
        ok = _load()
        print(json.dumps({"warmed": ok}))
        sys.exit(0 if ok else 1)
    # Interactive stdio server.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _handle(req)
        except Exception as e:  # noqa: BLE001 - fail-open: report and continue
            resp = {"error": "%s: %s" % (type(e).__name__, e)}
        try:
            sys.stdout.write(json.dumps(resp, allow_nan=False) + "\n")
            sys.stdout.flush()
        except Exception:
            break


if __name__ == "__main__":
    main()
