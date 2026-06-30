# Redrob Ranker — Project Plan

## Goal
Read 100,000 résumés, output the top-100 best-fit candidates for the
"Senior AI Engineer" job, as a CSV (candidate_id,rank,score,reasoning).
Constraints: runs in <5 min, <16 GB RAM, CPU only, no internet during ranking.

## The scoring formula (plain English)
FINAL = FIT(0-100) x AVAILABILITY(0-1), with honeypots forced to 0.

FIT components:
- semantic/text match to the job (TF-IDF cosine)   ~35
- real skills (endorsements x duration x assessment) ~25
- right job title / role                             ~20
- right experience level (5-9 yrs)                   ~10
- product vs services company                         ~5
- location (Pune/Noida/India metros)                  ~5

AVAILABILITY multiplier (starts 1.0, shrinks for):
- low recruiter response rate
- not logged in recently
- not open to work
- long notice period

HONEYPOT checks force score to 0:
- experience > company age
- "expert" skills with 0 months used
- impossible career dates

## Build steps
1. [done] Explore the data (explore.py)
2. [in progress] Build rank.py: load -> score -> top 100 -> reasoning -> CSV
3. Validate with validate_submission.py
4. Tune weights against a small hand-labelled set
5. (optional upgrade) swap TF-IDF for neural sentence-embeddings
6. Repo README + metadata + sandbox

## Key facts
- Data: data/candidates.jsonl (100k rows, ~465 MB)
- Most candidates are non-tech noise (HR, Sales, Mechanical Eng).
- Validator tie-break rule: equal scores -> candidate_id ascending.
