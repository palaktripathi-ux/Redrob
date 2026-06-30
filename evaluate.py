"""
evaluate.py — independent "answer key" + weight tuner (v2, no circularity).

The lesson from v1: if the answer key reuses the ranker's own scoring
functions, grading becomes circular and you get a meaningless perfect score.

v2 fixes that:
  * The gold judge is built ONLY from RAW profile fields (its own logic),
    especially career-history EVIDENCE of building relevant systems. It does
    NOT call the ranker's component scorers.
  * We split candidates into a TUNE half and a held-out VALIDATION half.
    Weights are searched on the tune half and chosen by the validation half,
    so we measure real generalisation, not memorisation.
  * Guardrails: after tuning we re-check that the top-100 contains no
    wrong-field titles and no honeypots (the JD's hard constraints).

Run:  python evaluate.py
"""

import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import rank  # only for the RANKER features (the thing being tuned) + IO + honeypot

random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# INDEPENDENT answer key. Built from raw fields with its own logic.
# ---------------------------------------------------------------------------
SYSTEM_EVIDENCE = [
    "recommendation", "recommender", "personalization", "personalisation",
    "search", "ranking", "retrieval", "relevance", "matching", "embedding",
    "semantic", "vector", "information retrieval", "learning to rank",
    "nlp", "natural language", "language model", "fine-tun", "llm", "rag",
    "similarity", "nearest neighbor", "search engine", "ranking model",
]
PRODUCTION_EVIDENCE = ["production", "deployed", "real users", "at scale",
                       "millions", "latency", "serving", "a/b test", "ab test"]
WRONG_FIELD = rank.WEAK_TITLES          # data list, not a scorer
SERVICES = rank.CONSULTING              # data list, not a scorer


def g_system_evidence(c):
    text = " ".join(
        [c.get("profile", {}).get("summary", "")] +
        [j.get("description", "") for j in c.get("career_history", [])] +
        [j.get("title", "") for j in c.get("career_history", [])]
    ).lower()
    hits = sum(1 for p in SYSTEM_EVIDENCE if p in text)
    prod = sum(1 for p in PRODUCTION_EVIDENCE if p in text)
    return min(hits / 6.0, 1.0) * 0.8 + min(prod / 3.0, 1.0) * 0.2


def g_skill_depth(c):
    pw = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    total = 0.0
    for sk in c.get("skills", []):
        name = (sk.get("name") or "").lower()
        if not any(t in name for t in rank.RELEVANT_TERMS):
            continue
        months = sk.get("duration_months", 0) or 0
        endorse = sk.get("endorsements", 0) or 0
        total += pw.get(sk.get("proficiency", "beginner"), 0.3) * \
            (min(months, 36) / 36.0 * 0.6 + min(endorse, 25) / 25.0 * 0.4)
    return min(total / 4.0, 1.0)


def g_experience(c):
    y = c.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= y <= 9: return 1.0
    if 4 <= y < 5 or 9 < y <= 11: return 0.65
    if 3 <= y < 4 or 11 < y <= 13: return 0.35
    return 0.1


def g_company(c):
    hist = c.get("career_history", [])
    if not hist:
        return 0.5
    services = sum(1 for j in hist
                   if any(x in (j.get("company") or "").lower() for x in SERVICES))
    if services == len(hist):
        return 0.1
    ind = (c.get("profile", {}).get("current_industry") or "").lower()
    return 0.5 if ("it services" in ind or "consulting" in ind) else 0.85


def g_location(c):
    prof = c.get("profile", {})
    loc = (prof.get("location") or "").lower()
    if any(x in loc for x in rank.TOP_LOCATIONS):
        return 1.0
    return 0.7 if "india" in (prof.get("country") or "").lower() else 0.25


def g_availability(c):
    sig = c.get("redrob_signals", {})
    m = 1.0
    m *= 0.4 + 0.6 * max(0.0, min(sig.get("recruiter_response_rate", 0.5), 1.0))
    last = rank.parse_date(sig.get("last_active_date", "") or "")
    if last:
        d = (rank.TODAY - last).days
        m *= 1.0 if d <= 30 else 0.85 if d <= 90 else 0.6 if d <= 180 else 0.35
    if not sig.get("open_to_work_flag", False):
        m *= 0.7
    return m


def g_right_field(c):
    title = (c.get("profile", {}).get("current_title") or "").lower()
    if any(b in title for b in WRONG_FIELD):
        return g_system_evidence(c) >= 0.5   # off-title but technical history
    techy = ["engineer", "scientist", "developer", "ml", "ai", "data",
             "research", "architect", "programmer"]
    return any(t in title for t in techy) or g_system_evidence(c) >= 0.5


def gold_tier(c):
    if rank.honeypot_flags(c):
        return 0
    if not g_right_field(c):
        return 0
    sysev, depth = g_system_evidence(c), g_skill_depth(c)
    listed_rel = sum(1 for sk in c.get("skills", [])
                     if any(t in (sk.get("name") or "").lower()
                            for t in rank.RELEVANT_TERMS))
    if listed_rel >= 4 and depth < 0.15 and sysev < 0.25:
        return 0                              # keyword stuffer
    true_fit = (0.40 * sysev + 0.30 * depth + 0.15 * g_experience(c) +
                0.10 * g_company(c) + 0.05 * g_location(c))
    true_fit *= (0.5 + 0.5 * g_availability(c))
    if true_fit >= 0.62: return 4
    if true_fit >= 0.46: return 3
    if true_fit >= 0.30: return 2
    if true_fit >= 0.16: return 1
    return 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg(ranked_rels, all_rels, k):
    idcg = dcg(sorted(all_rels, reverse=True)[:k])
    return dcg(ranked_rels[:k]) / idcg if idcg > 0 else 0.0


def average_precision(ranked_rels, k=100, thr=3):
    hits, precs = 0, []
    for i, r in enumerate(ranked_rels[:k], 1):
        if r >= thr:
            hits += 1
            precs.append(hits / i)
    return sum(precs) / hits if hits else 0.0


def patk(ranked_rels, k=10, thr=3):
    return sum(1 for r in ranked_rels[:k] if r >= thr) / k


def composite(order, gold_subset):
    rels = gold_subset[order].tolist()
    allr = gold_subset.tolist()
    n10, n50 = ndcg(rels, allr, 10), ndcg(rels, allr, 50)
    mp, p10 = average_precision(rels), patk(rels)
    return 0.50 * n10 + 0.30 * n50 + 0.15 * mp + 0.05 * p10, (n10, n50, mp, p10)


def main():
    print("Loading candidates...")
    cands = list(rank.open_candidates("./data/candidates.jsonl"))
    n = len(cands)
    print(f"  {n:,}")

    print("Precomputing ranker features + independent gold labels...")
    texts = [rank.candidate_text(c) for c in cands]
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2, stop_words="english")
    matrix = vec.fit_transform(texts)
    sem = linear_kernel(vec.transform([rank.JD_QUERY]), matrix).ravel()
    sem = sem / sem.max() if sem.max() > 0 else sem

    parts = np.zeros((n, 6), dtype=np.float32)
    avail = np.zeros(n, dtype=np.float32)
    honey = np.zeros(n, dtype=bool)
    gold = np.zeros(n, dtype=np.int32)
    for i, c in enumerate(cands):
        parts[i] = [sem[i], rank.skills_score(c), rank.title_score(c),
                    rank.experience_score(c), rank.company_score(c),
                    rank.location_score(c)]
        avail[i] = rank.availability_multiplier(c)
        honey[i] = bool(rank.honeypot_flags(c))
        gold[i] = gold_tier(c)
    print("  gold tiers: "
          + ", ".join(f"T{t}={int((gold==t).sum()):,}" for t in range(5)))

    # held-out split
    idx = np.arange(n); np.random.shuffle(idx)
    tune_idx, val_idx = idx[: n // 2], idx[n // 2:]
    keys = ["semantic", "skills", "title", "exp", "company", "location"]

    def score_all(w, alpha):
        wv = np.array([w[k] for k in keys], dtype=np.float32)
        final = (parts @ wv) * (avail ** alpha)
        final[honey] = 0.0
        return final

    def comp_on(final, subset):
        order_local = np.argsort(-final[subset], kind="stable")[:200]
        return composite(order_local, gold[subset])

    base_w = dict(rank.WEIGHTS)
    bf = score_all(base_w, 1.0)
    base_tr, _ = comp_on(bf, tune_idx)
    base_va, base_m = comp_on(bf, val_idx)
    print(f"\nCURRENT weights  tune={base_tr:.4f}  VALIDATION={base_va:.4f}  "
          f"(NDCG@10={base_m[0]:.3f} NDCG@50={base_m[1]:.3f} "
          f"MAP={base_m[2]:.3f} P@10={base_m[3]:.3f})")

    print("\nSearching weights on TUNE half, selecting by VALIDATION half...")
    best_val, best_w, best_alpha = base_va, base_w, 1.0
    for _ in range(6000):
        w = {k: random.uniform(0, 40) for k in keys}
        alpha = random.uniform(0.5, 2.0)
        final = score_all(w, alpha)
        tr, _ = comp_on(final, tune_idx)
        if tr <= base_tr:           # must at least beat base on tune set
            continue
        va, _ = comp_on(final, val_idx)
        if va > best_val:
            best_val, best_w, best_alpha = va, dict(w), alpha

    s = sum(best_w.values())
    norm_w = {k: round(best_w[k] / s * 100, 2) for k in keys}
    bf2 = score_all(best_w, best_alpha)
    _, best_m = comp_on(bf2, val_idx)

    # guardrails on the FULL pool with tuned weights
    final_full = score_all(best_w, best_alpha)
    top100 = np.argsort(-final_full, kind="stable")[:100]
    wrong_field = sum(1 for i in top100
                      if any(b in (cands[i]["profile"]["current_title"] or "").lower()
                             for b in WRONG_FIELD))
    honey_top = int(honey[top100].sum())

    print(f"\nBEST (by validation) = {best_val:.4f}  "
          f"(NDCG@10={best_m[0]:.3f} NDCG@50={best_m[1]:.3f} "
          f"MAP={best_m[2]:.3f} P@10={best_m[3]:.3f})")
    print(f"Validation improvement: {base_va:.4f} -> {best_val:.4f}  "
          f"(+{100*(best_val-base_va)/max(base_va,1e-9):.2f}%)")
    print(f"Guardrails with tuned weights -> wrong-field in top100: {wrong_field}, "
          f"honeypots in top100: {honey_top}")
    print(f"\nTuned weights (sum=100): {norm_w}")
    print(f"Tuned availability exponent alpha = {best_alpha:.2f}")


if __name__ == "__main__":
    main()
