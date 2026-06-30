"""
app.py — sandbox/demo (required by the hackathon, Section 10.5).

A tiny web app: upload a small candidate sample (<=100, .json array or .jsonl),
click Rank, and get a ranked table + a downloadable submission-style CSV.
Uses the SAME scoring logic as rank.py (imported), so the demo proves the real
ranker runs end-to-end on CPU within the budget.

Run locally:   python app.py
Deploy free:   push this repo to a HuggingFace Space (SDK: gradio).
"""

import csv
import io
import json

import gradio as gr

import rank


def _load(file_obj):
    text = open(file_obj.name, "r", encoding="utf-8").read().strip()
    # Accept either a JSON array or JSONL.
    if text.startswith("["):
        cands = json.loads(text)
    else:
        cands = [json.loads(l) for l in text.splitlines() if l.strip()]
    return cands


def run(file_obj, topk):
    if file_obj is None:
        return None, "Please upload a .json or .jsonl candidate sample."
    cands = _load(file_obj)
    if len(cands) > 100:
        cands = cands[:100]
    ranked = rank.rank_candidates(cands, topk=min(int(topk), len(cands)))

    # downloadable CSV
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in ranked:
        w.writerow([r["candidate_id"], r["rank"], f"{r['score']:.6f}", r["reasoning"]])
    out_path = "ranked_sample.csv"
    open(out_path, "w", encoding="utf-8", newline="").write(buf.getvalue())

    table = [[r["rank"], r["candidate_id"], f"{r['score']:.4f}", r["reasoning"]]
             for r in ranked]
    return out_path, table


with gr.Blocks(title="Redrob Ranker — Sandbox") as demo:
    gr.Markdown(
        "# Redrob Candidate Ranker — Sandbox\n"
        "Upload a small candidate sample (`.json` array or `.jsonl`, <=100 rows). "
        "The ranker scores them for the **Senior AI Engineer** JD and returns the "
        "top picks with reasoning. Runs on CPU, no network calls."
    )
    with gr.Row():
        f = gr.File(label="Candidate sample (.json / .jsonl)")
        k = gr.Slider(1, 100, value=20, step=1, label="How many to return")
    btn = gr.Button("Rank candidates", variant="primary")
    csv_out = gr.File(label="Download ranked CSV")
    tbl = gr.Dataframe(headers=["rank", "candidate_id", "score", "reasoning"],
                       label="Ranked candidates", wrap=True)
    btn.click(run, inputs=[f, k], outputs=[csv_out, tbl])


if __name__ == "__main__":
    demo.launch()
