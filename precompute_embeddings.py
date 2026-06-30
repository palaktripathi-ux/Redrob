"""
precompute_embeddings.py — the neural "meaning" upgrade (offline, one-time).

Why separate from rank.py:
  The contest's 5-minute ranking budget is too small to run a neural model over
  100,000 people on CPU. The rules explicitly allow PRE-computation outside that
  window. So we embed every candidate ONCE here, save the vectors to disk, and
  rank.py just reuses them (it only has to embed the single job description at
  rank time, which is instant).

What it produces:
  data/embeddings.npy   float32 matrix, one row per candidate (L2-normalised)
  data/embeddings_ids.json   the candidate_ids in the same row order
  data/embeddings_meta.json  which model was used

Run (needs internet the FIRST time, to download the model ~130 MB):
  python precompute_embeddings.py
After that the model is cached locally and everything runs offline.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import rank  # reuse open_candidates()

# all-MiniLM-L6-v2: 6-layer, 384-dim, much faster on CPU than 12-layer models,
# and strong enough for this short-text matching. Swap to BAAI/bge-small-en-v1.5
# if you have time/GPU and want a small quality bump.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ = 128            # cap tokens -> big CPU speedup; profiles fit fine
DATA = Path(__file__).parent / "data"
EMB_FILE = DATA / "embeddings.npy"
IDS_FILE = DATA / "embeddings_ids.json"
META_FILE = DATA / "embeddings_meta.json"


def embed_text(c):
    """A SHORT, focused text per candidate for embedding (speed + signal).
    Long role descriptions are trimmed so the model isn't drowned in boilerplate."""
    prof = c.get("profile", {})
    parts = [prof.get("headline", ""), prof.get("current_title", ""),
             (prof.get("summary", "") or "")[:400]]
    for job in c.get("career_history", [])[:3]:
        parts.append(job.get("title", ""))
        parts.append((job.get("description", "") or "")[:200])
    skills = [s.get("name", "") for s in c.get("skills", [])][:12]
    parts.append(", ".join(skills))
    return " ".join(p for p in parts if p)


def main():
    torch.set_num_threads(torch.get_num_threads())  # use all CPU cores
    print(f"Loading model {MODEL_NAME} (downloads once, then cached)...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ

    print("Loading candidates + building text...")
    ids, texts = [], []
    for c in rank.open_candidates(DATA / "candidates.jsonl"):
        ids.append(c["candidate_id"])
        texts.append(embed_text(c))
    print(f"  {len(texts):,} candidates")

    print("Embedding (one-time; outside the 5-min ranking budget)...")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,   # so cosine = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  done in {time.time()-t0:.0f}s, shape {emb.shape}")

    np.save(EMB_FILE, emb)
    IDS_FILE.write_text(json.dumps(ids))
    META_FILE.write_text(json.dumps({"model": MODEL_NAME, "dim": emb.shape[1]}))
    print(f"Saved -> {EMB_FILE.name}, {IDS_FILE.name}, {META_FILE.name}")


if __name__ == "__main__":
    main()
