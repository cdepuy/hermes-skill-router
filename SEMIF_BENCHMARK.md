# SemIf (Qwen3.5-4B) vs Laya (MLX) — skill-routing benchmark

**Date:** 2026-09-21
**Machine:** win11 RTX 3080 (192.168.1.253). SemIf torch backend:
torch 2.10.0+cu126, transformers 5.17.0, torchvision 0.25.0+cu126,
model `Qwen/Qwen3.5-4B` rev `851bf6e8...`, direct-logit readout, BF16.
Isolated in `C:\wav2lip\semifpkg` (sys.path shim) — LatentSync's embedded
py311 (torch 2.5.1/transformers 4.48) untouched.

**Method:** same 9 real-task candidate pools the router feeds Laya
(correct skill present in the 4-12 option pool), top-1 accuracy. Input rows in
SemIf's `{id, state, question, options[]}` format; `direct_score` reads
option logits in one forward pass.

## Result

| Metric | Laya (MLX) | SemIf (Qwen3.5-4B) |
|---|---|---|
| Top-1 accuracy | 5/9 (56%) | **6/9 (67%)** |
| Avg non-top1 probability share | ~0 (one-hot) | **21% (usable spread)** |
| Warm wall-time / decision | ~0.1s | **0.21s** |
| Confidently-wrong @ P≈1.0 | 2 (whitepaper→pdf, ssh→remote-swap) | 0 |

## SemIf per-decision (top1, P, wall, in_tok)
- nas-zip-drain → smb-nas-mount-resiliency 99.8% (0.69s) ✅
- floor curve → skill-router-plugin 71.8% (0.13s) ✅
- compare SemanticRouter→Laya → system-one-model-evaluation 88.0% ❌ (skill-router 2nd 5.6%)
- new uses / Jev → system-one-model-evaluation 90.9% ❌ (jev-local 2nd 4.0%)
- blog post → blog-post-creation 51.8% (0.17s) ✅
- whitepaper → whitepaper-generation 84.6% ❌ (lenient-correct sibling of whitepaper-authoring)
- X announcement → x-post 77.6% (0.18s) ✅
- ssh failing → ssh-connection-fixer 44.6% (0.17s) ✅ [Laya picked remote-engine-swap @ 1.0]
- minecraft → minecraft-modpack-server 100.0% (0.12s) ✅

## Reading
SemIf is a **real improvement** over Laya for routing: higher top-1 (67 vs 56%),
fixes both of Laya's P=1.0 confidently-wrong picks, and—critically—returns a
usable probability distribution (not Laya's one-hot), so a confidence floor is
meaningful again. Same speed class (~0.2s warm on live GPU).

**But it is not a fix.** SemIf still misroutes the meta tasks
("compare router", "new Jev uses") toward `system-one-model-evaluation`
at 88-91% confidence, and on the full interactive set (including the 3/12
prefilter MISSES that remove the correct skill before the engine even runs)
end-to-end accuracy stays <70%. Both engines struggle on messy real prompts;
SemIf fails less often and less confidently.

## SemIf-on-3080 gotchas (verified)
- Reuse LatentSync's embedded `C:\wav2lip\py311\python.exe`; it has no venv.
- Isolate SemIf via `pip install --target C:\wav2lip\semifpkg` + a
  `sys.path.insert(0, ...)` shim. PYTHONPATH is ignored (embedded `_pth`).
- In `_pth` isolated mode, `sys.path.insert` at runtime still works.
- torch 2.10.0+cu126 (cu126 index, reachable direct; NOT cu128 — NGC DNS-blocked).
- transformers 5.17.0 pulls torchvision transitively; the box's base torchvision
  (built vs torch 2.5.1) breaks with `operator torchvision::nms does not exist`.
  FIX: install matching `torchvision==0.25.0+cu126` into `semifpkg` so it shadows base.
- Start-Process -WindowStyle Hidden silently dies when SSH closes → run python
  via a LIVE background SSH session from the Mac (terminal background=True).
- No admin needed: everything lands under C:\wav2lip and the user profile.
