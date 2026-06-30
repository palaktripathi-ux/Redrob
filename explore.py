"""
Step 1: Look at the data before we build anything.

This script reads the 100,000 résumés one line at a time (streaming, so it
never loads the whole 465 MB file into memory at once) and prints simple
summaries: how many people, what job titles are common, experience spread,
locations, and what the behavioural signals look like.

Run:  python explore.py
"""

import json
import gzip
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data" / "candidates.jsonl"


def open_candidates(path):
    """Open .jsonl or .jsonl.gz transparently and yield one candidate dict per line."""
    if str(path).endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    total = 0
    titles = Counter()
    countries = Counter()
    locations = Counter()
    years = []
    response_rates = []
    github_scores = []
    open_to_work = 0
    skills_per_candidate = []

    for c in open_candidates(DATA):
        total += 1
        prof = c.get("profile", {})
        titles[prof.get("current_title", "?")] += 1
        countries[prof.get("country", "?")] += 1
        locations[prof.get("location", "?")] += 1
        years.append(prof.get("years_of_experience", 0))

        sig = c.get("redrob_signals", {})
        response_rates.append(sig.get("recruiter_response_rate", 0))
        github_scores.append(sig.get("github_activity_score", -1))
        if sig.get("open_to_work_flag"):
            open_to_work += 1

        skills_per_candidate.append(len(c.get("skills", [])))

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0

    print(f"\n=== TOTAL CANDIDATES: {total:,} ===\n")

    print("--- Top 20 current job titles ---")
    for t, n in titles.most_common(20):
        print(f"  {n:6,}  {t}")

    print("\n--- Top 15 countries ---")
    for t, n in countries.most_common(15):
        print(f"  {n:6,}  {t}")

    print("\n--- Top 15 locations ---")
    for t, n in locations.most_common(15):
        print(f"  {n:6,}  {t}")

    print("\n--- Experience (years) ---")
    print(f"  min={min(years)}  max={max(years)}  avg={avg(years)}")
    in_band = sum(1 for y in years if 5 <= y <= 9)
    print(f"  in the JD's 5-9 year band: {in_band:,} ({round(100*in_band/total,1)}%)")

    print("\n--- Behavioural signals ---")
    print(f"  avg recruiter response rate: {avg(response_rates)}")
    print(f"  open_to_work = true:         {open_to_work:,} ({round(100*open_to_work/total,1)}%)")
    print(f"  avg skills listed:           {avg(skills_per_candidate)}")
    has_github = sum(1 for g in github_scores if g >= 0)
    print(f"  have a GitHub linked:        {has_github:,} ({round(100*has_github/total,1)}%)")


if __name__ == "__main__":
    main()
