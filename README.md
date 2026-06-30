# Redrob Ranker — Intelligent Candidate Discovery & Ranking

Ranks the top-100 best-fit candidates from a 100,000-person pool for the
**Senior AI Engineer — Founding Team** job description, and writes a
submission CSV (`candidate_id,rank,score,reasoning`).

Runs on **CPU only, offline, in ~1 minute** — well inside the contest's
5-minute / 16 GB / no-network budget. No per-candidate LLM calls.

---

## TL;DR — reproduce the submission

```bash
# 1. install dependencies (CPU torch)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. put the data in place
#    data/candidates.jsonl   (from the hackathon bundle; .jsonl.gz also works)

# 3. (optional, recommended) precompute neural embeddings ONCE (offline-allowed)
python precompute_embeddings.py            # ~15 min one-time, CPU

# 4. produce the submission (this is the timed step; ~1 min)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 5. validate the format before uploading
python validate_submission.py submission.csv
```

If you skip step 3, `rank.py` automatically falls back to a TF-IDF meaning
match — it still produces a valid, strong submission with no extra setup.

---

## How it works (the method)

Every candidate gets one final score, and we keep the top 100:

```
FINAL = FIT(0..100)  x  AVAILABILITY(0..1)        # honeypots forced to 0
```

### FIT — how well they match the job (weights tuned, see below)
| Component | Weight | What it rewards |
|---|---:|---|
| `skills`   | 32 | **Real, used** relevant skills (proficiency x months x endorsements x assessment) — not keyword count |
| `semantic` | 25 | Meaning-match of their career story to the JD (neural embeddings, or TF-IDF fallback) |
| `exp`      | 17 | Experience in the JD's 5–9 year band |
| `title`    | 10 | Current role is the right kind (ML/AI/Search/Data...) — the decisive anti-trap signal |
| `location` |  8 | Pune/Noida and other Indian metros the JD prefers |
| `company`  |  8 | Product companies over career-long services firms |

### AVAILABILITY — can we actually hire them?
A multiplier (softened by `AVAIL_ALPHA`) that shrinks the score for low
recruiter-response-rate, long inactivity, not-open-to-work, and long notice
periods. A perfect-on-paper candidate who's unreachable is correctly buried.

### Honeypots — impossible profiles forced to 0
Sanity checks catch fabricated profiles (e.g. a skill used longer than the
person's whole career; multiple "expert" skills with 0 months of use). These
are zeroed so they can never reach the top 100 (>10% honeypots = disqualified).

### Defeating the dataset's traps
- **Keyword stuffers** → the `skills` score weights by real usage/endorsements, so empty buzzword lists score low.
- **Plain-language "hidden gems"** → the `semantic` match reads career *descriptions*, finding people who built search/ranking/recommendation systems without using the buzzwords.
- **Behavioural twins** → the multiplicative availability term down-weights the unreachable.
- **Honeypots** → consistency checks zero them out.

---

## Tuning & evaluation (`evaluate.py`)

The real ground truth is hidden, so we built our own evaluation framework:

1. An **independent gold-relevance judge** (tiers 0–4) built from *raw* fields
   — mainly career-history *evidence* of building relevant systems — which the
   ranker does **not** use directly, to avoid circular grading.
2. A **held-out validation split**: weights are searched on one half and
   selected on the other, so we measure generalisation, not memorisation.
3. Scored with the contest's real metrics
   (`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`).

Result: composite **0.975 → 0.996** on held-out data, with **0 honeypots** and
**0 wrong-field titles** in the top 100. `title` is kept at a floor of 10 as
domain-knowledge insurance (the JD calls it the decisive anti-trap signal).

```bash
python evaluate.py
```

---

## Files

| File | Purpose |
|---|---|
| `rank.py` | **The ranker.** Load → score → top-100 → reasoning → CSV. |
| `precompute_embeddings.py` | One-time offline neural embeddings (MiniLM). |
| `evaluate.py` | Independent answer-key + weight tuner (offline eval framework). |
| `explore.py` | Quick data exploration / sanity stats. |
| `app.py` | Sandbox demo (Gradio): rank a ≤100 sample in the browser. |
| `validate_submission.py` | Official format validator (from the bundle). |
| `submission_metadata.yaml` | Portal metadata mirror. |
| `requirements.txt` | Dependencies. |

---

## Sandbox / demo

`app.py` is a small Gradio app: upload a ≤100-candidate `.json`/`.jsonl`
sample and get a ranked table + downloadable CSV. Deploy free on
**HuggingFace Spaces** (SDK: Gradio) by pushing this repo, or run locally:

```bash
python app.py
```

## Compute constraints (met)
CPU-only · no network during ranking · ~1 min for 100k · < 16 GB RAM.
Neural embeddings are **pre**computed (allowed outside the 5-min window); the
timed ranking step only embeds the single job description.

## AI tools
Built with an AI coding assistant (GitHub Copilot CLI) for pair-programming and
review. All scoring logic and methodology were designed and verified by the
team; no candidate data was sent to any hosted LLM; the ranker makes no network
calls. See `submission_metadata.yaml`.
