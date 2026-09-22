"""Experiment: do cheap Option 2 (semantic prefilter) + Option 4 (no-skill gate)
improve skill_router? Compares the current lexical router against a semantic
(embedding) candidate selector and a confidence gate, on the REAL production
task set (actual telegram turns from the 6-day audit) plus a few eval tasks.

Ground truth per task: the ONE correct skill (strict). Measures:
  A) PREFILTER RECALL (stage 1): is the correct skill in the top-12 pool?
       - current lexical prefilter vs Qwen3-embedding cosine prefilter
  B) END-TO-END: does the current router pick the right skill at top-1?
  C) NO-SKILL GATE (Option 4): can an embedding-similarity threshold tell
       ambiguous "no good skill" turns apart from clear ones?
"""
from __future__ import annotations
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router

SKILLS = router.discover_skills()
NAME2SKILL = {s["name"]: s for s in SKILLS}

# (task, correct_skill) — STRICT ground truth
TASKS = [
    # --- REAL production telegram turns (from the 6-day audit) ---
    ("i'm still getting lots of nas-zip-drain failure messages. what should i do to fix it?",
     "smb-nas-mount-resiliency"),
    ("tell me - is this setup working on our local Hermes setup now?",
     "hermes-agent"),  # arguably none; strict best = hermes-agent
    ("make the curve and find optimal floor value for the router",
     "skill-router-plugin"),
    ("compare SemanticRouter to Laya router approach",
     "skill-router-plugin"),
    ("figure if any of these are new uses or Laya / Jev type approaches",
     "jev-local-system-one-engine"),
    ("is it a subset of what we have done already? is there anything new in it?",
     "x-article-idea-briefing"),
    ("getting back on this - is the router working? any commands to run?",
     "hermes-agent"),
    # --- eval-style clean tasks ---
    ("draft a 650 Group blog post about optical components earnings",
     "blog-post-creation"),
    ("write a formal 650 Group whitepaper as a PDF",
     "whitepaper-authoring"),
    ("post an announcement to X/Twitter as cdepuy",
     "x-post"),
    ("diagnose why ssh to the remote server is failing",
     "ssh-connection-fixer"),
    ("set up a modded minecraft server",
     "minecraft-modpack-server"),
]


def lexical_prefilter_recall(task):
    cands = router._prefilter(task, SKILLS, top_k=router.PREFILTER_TOP_K)
    names = [c["name"] for c in cands]
    return names


def embed_prefilter_recall(task, model, name_emb):
    """Top-12 most similar skills by Qwen3 embedding cosine similarity."""
    t_emb = model.encode([task])[0]
    import numpy as np
    scores = []
    for s in SKILLS:
        eb = name_emb[s["name"]]
        cos = float(np.dot(t_emb, eb) / (np.linalg.norm(t_emb)*np.linalg.norm(eb)+1e-9))
        scores.append((cos, s["name"]))
    scores.sort(key=lambda x: -x[0])
    return [n for _, n in scores[:router.PREFILTER_TOP_K]]


def main():
    # Precompute skill embeddings once (cheap Option 2 core)
    print("Loading Qwen3-Embedding-0.6B and embedding %d skills..." % len(SKILLS))
    t = time.time()
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    texts = [f"{s['name'].replace('-',' ')}. {s['description']}" for s in SKILLS]
    emb = model.encode(texts, convert_to_numpy=True)
    name_emb = {SKILLS[i]["name"]: emb[i] for i in range(len(SKILLS))}
    print("embedded in %.1fs" % (time.time()-t))

    lines = []
    lex_recall = emb_recall = 0
    for task, correct in TASKS:
        lnames = lexical_prefilter_recall(task)
        enames = embed_prefilter_recall(task, model, name_emb)
        l_in = correct in lnames
        e_in = correct in enames
        lex_recall += l_in
        emb_recall += e_in
        lines.append(f"task: {task[:55]}")
        lines.append(f"    correct={correct:<35} lex_in_pool={l_in}  emb_in_pool={e_in}")
        lines.append(f"    lex_top6  = {lnames[:6]}")
        lines.append(f"    emb_top6  = {enames[:6]}")
        lines.append("")
    n = len(TASKS)
    lines.append(f"=== PREFILTER RECALL (correct skill in top-{router.PREFILTER_TOP_K}) ===")
    lines.append(f"  lexical (current): {lex_recall}/{n}")
    lines.append(f"  embedding (Option2): {emb_recall}/{n}")
    text = "\n".join(lines)
    print(text)
    with open("/tmp/opt24_experiment.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
