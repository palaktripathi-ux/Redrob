"""
rank.py — the ranker.

What it does, in plain English:
  1. Reads every candidate.
  2. Builds a "fit score" (0-100) from several common-sense pieces.
  3. Multiplies that by an "availability" number (are they reachable / active?).
  4. Forces obviously-fake profiles (honeypots) to score 0.
  5. Sorts everyone, keeps the top 100, writes a short honest reason for each.
  6. Saves submission.csv in the exact required format.

Run:
  python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Everything here runs on CPU, offline, well under the 5-minute limit.
"""

import argparse
import gzip
import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# "Today" for recency math. The data's signals are dated up to ~mid 2026.
TODAY = date(2026, 6, 30)

# ---------------------------------------------------------------------------
# The job, expressed as a search query. The TF-IDF "meaning match" compares
# each résumé against this text. It captures what the JD actually wants.
# ---------------------------------------------------------------------------
JD_QUERY = """
senior ai engineer machine learning engineer applied scientist
embeddings based retrieval systems sentence transformers bge e5 openai embeddings
vector database hybrid search pinecone weaviate qdrant milvus faiss opensearch elasticsearch
ranking learning to rank recommendation system search information retrieval nlp
large language model llm fine tuning lora qlora peft rag re-ranking
evaluation framework ndcg mrr map ab testing offline online metrics
production deployment real users index refresh embedding drift retrieval quality
strong python code quality mlops model serving inference optimization
product company applied ml shipped end to end ranking search recommendation at scale
"""

# Skills / phrases that genuinely matter for THIS role.
RELEVANT_TERMS = [
    "embedding", "embeddings", "sentence-transformer", "sentence transformers",
    "bge", "e5", "retrieval", "vector search", "vector database", "vector db",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "bm25", "hybrid search", "semantic search",
    "ranking", "learning to rank", "ltr", "recommendation", "recommender",
    "search", "information retrieval", "nlp", "natural language",
    "llm", "large language model", "language model", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "rag", "transformer", "transformers",
    "pytorch", "tensorflow", "xgboost", "ndcg", "mrr",
    "machine learning", "deep learning", "mlops", "model serving",
    "inference", "python",
]

# Current-title scoring. Substring match on the lowercased title.
# High = clearly the right kind of role; 0 = clearly the wrong field.
STRONG_TITLES = {
    "machine learning": 1.0, "ml engineer": 1.0, "ai engineer": 1.0,
    "applied scientist": 1.0, "applied science": 1.0, "nlp": 1.0,
    "data scientist": 0.9, "research engineer": 0.85, "search engineer": 1.0,
    "ml ops": 0.85, "mlops": 0.85,
    "software engineer": 0.7, "backend engineer": 0.6, "back end": 0.6,
    "data engineer": 0.6, "full stack": 0.55, "fullstack": 0.55,
    "cloud engineer": 0.5, "devops": 0.45, "platform engineer": 0.55,
    "frontend": 0.35, "mobile developer": 0.3, "java developer": 0.4,
    ".net developer": 0.35, "developer": 0.5, "engineer": 0.45,
}
# Titles that are clearly the wrong field for this AI role.
WEAK_TITLES = [
    "hr manager", "human resource", "sales", "marketing", "accountant",
    "mechanical", "civil", "content writer", "graphic designer",
    "customer support", "operations manager", "business analyst",
    "project manager", "recruiter", "finance", "administrator",
]

# Career-long employment only at these = JD red flag ("services firms").
CONSULTING = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "ltimindtree", "mindtree", "mphasis",
    "igate", "syntel", "larsen", "ibm global",
]

# Locations the JD prefers (Pune/Noida) and nearby Indian metros.
TOP_LOCATIONS = ["pune", "noida", "delhi", "gurgaon", "gurugram", "hyderabad",
                 "bangalore", "bengaluru", "mumbai", "ncr"]


def open_candidates(path):
    f = gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") \
        else open(path, "r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_date(s):
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        return None


def candidate_text(c):
    """One big text blob per candidate, used for the TF-IDF meaning match."""
    prof = c.get("profile", {})
    parts = [prof.get("headline", ""), prof.get("summary", ""),
             prof.get("current_title", ""), prof.get("current_industry", "")]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for sk in c.get("skills", []):
        parts.append(sk.get("name", ""))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Individual scoring pieces. Each returns roughly 0..1; we weight them later.
# ---------------------------------------------------------------------------
PROF_WEIGHT = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}


def skills_score(c):
    """Reward REAL relevant skills: weighted by proficiency, endorsements,
    months of real use, and Redrob assessment scores. A long list of empty
    keywords scores low — this is the main defence against keyword stuffers."""
    sig = c.get("redrob_signals", {})
    assess = sig.get("skill_assessment_scores", {}) or {}
    total = 0.0
    for sk in c.get("skills", []):
        name = (sk.get("name") or "").lower()
        if not any(term in name for term in RELEVANT_TERMS):
            continue  # not a skill this job cares about
        prof = PROF_WEIGHT.get(sk.get("proficiency", "beginner"), 0.3)
        months = sk.get("duration_months", 0) or 0
        endorse = sk.get("endorsements", 0) or 0
        # trust grows with real usage time and peer endorsements, but saturates
        trust = (min(months, 48) / 48.0) * 0.6 + min(endorse, 30) / 30.0 * 0.4
        a = assess.get(sk.get("name", ""), None)
        assess_factor = (a / 100.0) if isinstance(a, (int, float)) else 0.6
        total += prof * (0.3 + 0.7 * trust) * (0.5 + 0.5 * assess_factor)
    # ~6 strong relevant skills ≈ full marks
    return min(total / 6.0, 1.0)


def title_score(c):
    """How well their current role matches the kind of job this is."""
    title = (c.get("profile", {}).get("current_title") or "").lower()
    for bad in WEAK_TITLES:
        if bad in title:
            return 0.0
    best = 0.0
    for key, val in STRONG_TITLES.items():
        if key in title:
            best = max(best, val)
    return best


def experience_score(c):
    """The JD wants 5-9 years (with some flex)."""
    y = c.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= y <= 9:
        return 1.0
    if 4 <= y < 5 or 9 < y <= 11:
        return 0.7
    if 3 <= y < 4 or 11 < y <= 13:
        return 0.4
    return 0.15


def company_score(c):
    """Bonus for product companies; penalty if the WHOLE career is services
    firms (a JD red flag)."""
    hist = c.get("career_history", [])
    if not hist:
        return 0.5
    consulting_hits = 0
    for job in hist:
        comp = (job.get("company") or "").lower()
        if any(x in comp for x in CONSULTING):
            consulting_hits += 1
    if consulting_hits == len(hist):
        return 0.1   # entire career at services firms
    cur_ind = (c.get("profile", {}).get("current_industry") or "").lower()
    if "it services" in cur_ind or "consulting" in cur_ind:
        return 0.5
    return 0.8


def location_score(c):
    prof = c.get("profile", {})
    loc = (prof.get("location") or "").lower()
    country = (prof.get("country") or "").lower()
    if any(x in loc for x in TOP_LOCATIONS):
        return 1.0
    if "india" in country:
        return 0.7
    # JD: outside India case-by-case, no visa sponsorship
    return 0.25


def availability_multiplier(c):
    """Can we actually hire/reach this person? Starts at 1.0, shrinks for
    red flags. A perfect-on-paper ghost ends up near zero."""
    sig = c.get("redrob_signals", {})
    m = 1.0

    rr = sig.get("recruiter_response_rate", 0.5)
    m *= 0.4 + 0.6 * max(0.0, min(rr, 1.0))          # 0% reply -> 0.4x

    last = parse_date(sig.get("last_active_date", "") or "")
    if last:
        days = (TODAY - last).days
        if days <= 30:      m *= 1.0
        elif days <= 90:    m *= 0.85
        elif days <= 180:   m *= 0.6
        else:               m *= 0.35                 # 6+ months gone
    if not sig.get("open_to_work_flag", False):
        m *= 0.7
    notice = sig.get("notice_period_days", 60) or 60
    if notice <= 30:    m *= 1.0
    elif notice <= 60:  m *= 0.95
    elif notice <= 90:  m *= 0.85
    else:               m *= 0.75
    icr = sig.get("interview_completion_rate", 0.7)
    m *= 0.7 + 0.3 * max(0.0, min(icr, 1.0))
    return m


def honeypot_flags(c):
    """Return a list of reasons this profile looks IMPOSSIBLE (fake).
    If non-empty, we force the score to 0 so it can't reach the top 100."""
    flags = []
    prof = c.get("profile", {})
    yoe = prof.get("years_of_experience", 0) or 0
    months_cap = yoe * 12 + 12  # can't have used a skill longer than you've worked

    # Skill used longer than the person's entire career.
    for sk in c.get("skills", []):
        if (sk.get("duration_months", 0) or 0) > months_cap and yoe > 0:
            flags.append(f"claims {sk.get('duration_months')}mo of "
                         f"{sk.get('name')} but only {yoe}y total experience")
            break

    # Several "expert/advanced" skills with 0 months of actual use.
    zero_use_expert = sum(
        1 for sk in c.get("skills", [])
        if sk.get("proficiency") in ("expert", "advanced")
        and (sk.get("duration_months", 0) or 0) == 0
    )
    if zero_use_expert >= 3:
        flags.append(f"{zero_use_expert} expert-level skills with 0 months used")

    # Tenure at one job longer than the whole career.
    for job in c.get("career_history", []):
        if (job.get("duration_months", 0) or 0) > months_cap and yoe > 0:
            flags.append("single job longer than total experience")
            break

    # Career total wildly exceeds stated experience (impossible overlap).
    total_months = sum(j.get("duration_months", 0) or 0
                       for j in c.get("career_history", []))
    if yoe > 0 and total_months > yoe * 12 * 1.6 + 24:
        flags.append("career durations add up to far more than stated experience")

    return flags


# Weights for the FIT score. These were TUNED with evaluate.py against an
# independent gold-relevance judge on a held-out validation split (composite
# 0.975 -> 0.996). `title` is kept at a floor of 10 as domain-knowledge
# insurance: the JD explicitly calls current-title the decisive anti-trap
# signal, even though our offline judge rewards it less.
WEIGHTS = {
    "semantic": 24.73,
    "skills":   31.74,
    "title":    10.00,
    "exp":      17.26,
    "company":   7.84,
    "location":  8.45,
}

# Softening exponent on the availability multiplier (tuned). <1 makes the
# availability penalty less extreme so a strong candidate with one weak
# behavioural signal isn't wiped out entirely.
AVAIL_ALPHA = 0.6


def build_reasoning(c, parts, final):
    """A short, honest, specific sentence built ONLY from real profile facts."""
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    title = prof.get("current_title", "professional")
    yoe = prof.get("years_of_experience", 0)
    loc = prof.get("location", "unknown location")

    # name the candidate's relevant skills (real ones)
    rel = [sk.get("name") for sk in c.get("skills", [])
           if any(t in (sk.get("name") or "").lower() for t in RELEVANT_TERMS)]
    rel = rel[:3]

    bits = [f"{title} with {yoe}y experience"]
    if rel:
        bits.append("relevant skills: " + ", ".join(rel))
    bits.append(f"based in {loc}")

    # add an honest concern where one exists
    concerns = []
    rr = sig.get("recruiter_response_rate", 1)
    if rr < 0.4:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    last = parse_date(sig.get("last_active_date", "") or "")
    if last and (TODAY - last).days > 120:
        concerns.append(f"last active {(TODAY-last).days} days ago")
    notice = sig.get("notice_period_days", 0)
    if notice and notice > 90:
        concerns.append(f"{notice}-day notice period")
    if parts["company"] <= 0.1:
        concerns.append("career mainly at services firms")
    if parts["title"] < 0.5 and parts["semantic"] > 0.3:
        concerns.append("current title is off-target but profile shows fit")

    sentence = "; ".join(bits) + "."
    if concerns:
        sentence += " Concern: " + ", ".join(concerns[:2]) + "."
    return sentence


def compute_semantic(candidates, texts):
    """The 'meaning match' to the job, as a 0..1 array (one per candidate).

    Prefers PRECOMPUTED NEURAL embeddings (data/embeddings.npy) when available
    — at rank time we only embed the single job description (instant) and take
    cosine against the cached candidate vectors. This stays offline + CPU and
    well within the time budget. If the embeddings/model aren't present, we
    transparently fall back to TF-IDF so the ranker always works.
    """
    emb_file = Path("./data/embeddings.npy")
    ids_file = Path("./data/embeddings_ids.json")
    meta_file = Path("./data/embeddings_meta.json")

    if emb_file.exists() and ids_file.exists():
        try:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")      # no network at rank time
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import numpy as _np
            from sentence_transformers import SentenceTransformer

            emb = _np.load(emb_file)
            ids = json.loads(ids_file.read_text())
            model_name = json.loads(meta_file.read_text())["model"] \
                if meta_file.exists() else "BAAI/bge-small-en-v1.5"
            print(f"  using NEURAL embeddings ({model_name}, {emb.shape})")

            # align cached rows to current candidate order
            pos = {cid: i for i, cid in enumerate(ids)}
            order = [pos[c["candidate_id"]] for c in candidates]
            emb = emb[order]

            model = SentenceTransformer(model_name)
            # bge-style models want a query instruction; MiniLM-style don't.
            if "bge" in model_name.lower():
                q = "Represent this sentence for searching relevant passages: " + JD_QUERY
            else:
                q = JD_QUERY
            jd = model.encode([q], normalize_embeddings=True,
                              convert_to_numpy=True)[0].astype(_np.float32)
            sims = emb @ jd                       # cosine (both L2-normalised)
            sims = (sims - sims.min())
            return sims / sims.max() if sims.max() > 0 else sims
        except Exception as e:
            print(f"  neural embeddings unavailable ({e}); falling back to TF-IDF")

    print("  using TF-IDF meaning match")
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2, stop_words="english")
    matrix = vec.fit_transform(texts)
    sims = linear_kernel(vec.transform([JD_QUERY]), matrix).ravel()
    return sims / sims.max() if sims.max() > 0 else sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

def rank_candidates(candidates, topk=100, semantic=None):
    """Score and rank a list of candidate dicts. Returns the top-k as a list of
    dicts: candidate_id, rank, score, reasoning. Shared by the CLI and the
    sandbox app so both use identical logic."""
    texts = [candidate_text(c) for c in candidates]
    if semantic is None:
        semantic = compute_semantic(candidates, texts)

    rows = []
    for i, c in enumerate(candidates):
        flags = honeypot_flags(c)
        parts = {
            "semantic": float(semantic[i]),
            "skills":   skills_score(c),
            "title":    title_score(c),
            "exp":      experience_score(c),
            "company":  company_score(c),
            "location": location_score(c),
        }
        fit = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)  # 0..100
        avail = availability_multiplier(c) ** AVAIL_ALPHA
        final = 0.0 if flags else fit * avail
        # Sort on the EXACT written score so the validator's tie-break rule
        # (equal score -> candidate_id ascending) is guaranteed to hold.
        score = round(final / 100.0, 6)
        rows.append((score, parts, flags, c))

    rows.sort(key=lambda r: (-r[0], r[3]["candidate_id"]))
    top = rows[:topk]

    out = []
    for rank_pos, (score, parts, flags, c) in enumerate(top, start=1):
        out.append({
            "candidate_id": c["candidate_id"],
            "rank": rank_pos,
            "score": score,
            "reasoning": build_reasoning(c, parts, score),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    print("Loading candidates...")
    candidates = list(open_candidates(args.candidates))
    print(f"  loaded {len(candidates):,}")

    print("Computing meaning match against the job + scoring...")
    ranked = rank_candidates(candidates, topk=args.topk)

    print(f"Top score: {ranked[0]['score']:.4f}  |  "
          f"#{len(ranked)} score: {ranked[-1]['score']:.4f}")

    import csv
    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in ranked:
            w.writerow([r["candidate_id"], r["rank"],
                        f"{r['score']:.6f}", r["reasoning"]])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
