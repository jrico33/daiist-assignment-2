"""
Assignment 2: Gradio dashboard for hard-hat compliance.
 
    uv run python main.py app
 
Loads ONLY what train.py already produced:
  weights/best.pt            the finetuned model
  artifacts/results.json     metrics, per-class numbers, chosen operating point
  artifacts/frame_scores.csv test-split frame scores, used to recompute the
                             confusion matrix and cost live as sliders move
Nothing is trained here.
"""
 
from __future__ import annotations
 
import csv
import json
from pathlib import Path
 
import gradio as gr
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
 
import costing  # noqa: E402
 
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ART = ROOT / "artifacts"
 
# --------------------------------------------------------------------------- #
# Load pipeline outputs
# --------------------------------------------------------------------------- #
RES = json.loads((ART / "results.json").read_text()) if (ART / "results.json").exists() else None
 
SCORES, LABELS, COCO_SCORES = np.zeros(0), np.zeros(0, dtype=int), np.zeros(0)
if (ART / "frame_scores.csv").exists():
    with open(ART / "frame_scores.csv") as f:
        rows = list(csv.DictReader(f))
    SCORES = np.array([float(r["score_finetuned"]) for r in rows])
    COCO_SCORES = np.array([float(r["score_coco_zero_shot"]) for r in rows])
    LABELS = np.array([int(r["label_has_head"]) for r in rows])
 
MODEL = None
if WEIGHTS.exists():
    from ultralytics import YOLO
 
    MODEL = YOLO(str(WEIGHTS))
 
POS = RES["positive_class"] if RES else "head"
CHOSEN = RES["chosen_single"]["t_high"] if RES else 0.5
BAND = (RES["band_chosen"]["t_low"], RES["band_chosen"]["t_high"]) if RES else (0.3, 0.6)
DEFAULT = dict(costing.ASSUMPTIONS, **(RES["assumptions"] if RES else {}))
 
 
def assumptions_from(base_rate, cost_fn, cost_fp, cost_review, frames):
    return {**DEFAULT, "base_rate": base_rate, "cost_fn": cost_fn, "cost_fp": cost_fp,
            "cost_tp": cost_fp, "cost_review": cost_review, "frames_per_day": frames}
 
 
# --------------------------------------------------------------------------- #
# Tab 1: upload an image
# --------------------------------------------------------------------------- #
def run_image(img, t_low, t_high):
    if MODEL is None:
        return None, "No weights found in weights/. Run `main.py train` first.", None
    if img is None:
        return None, "Upload an image.", None
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    r = MODEL.predict(img, conf=max(t_low, 0.01), verbose=False)[0]
    plotted = r.plot()[:, :, ::-1]  # BGR -> RGB
    names = r.names
    dets = [(names[int(c)], float(s)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    head = max([s for n, s in dets if n == POS], default=0.0)
    if head >= t_high:
        verdict = f"## ALERT: supervisor intervenes\nmax '{POS}' score {head:.2f} >= {t_high:.2f}"
    elif head >= t_low:
        verdict = f"## REVIEW: send to the safety desk\nmax '{POS}' score {head:.2f} between {t_low:.2f} and {t_high:.2f}"
    else:
        verdict = f"## PASS\nmax '{POS}' score {head:.2f} < {t_low:.2f}"
    table = pd.DataFrame(sorted(dets, key=lambda d: -d[1]), columns=["class", "confidence"]).round(3)
    return plotted, verdict, table
 
 
# --------------------------------------------------------------------------- #
# Tab 2: operating point
# --------------------------------------------------------------------------- #
def cost_plot(a, t_low, t_high):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if len(SCORES):
        ft = costing.single_threshold_sweep(SCORES, LABELS, a)
        co = costing.single_threshold_sweep(COCO_SCORES, LABELS, a)
        ax.plot(costing.THRESHOLDS, [r["cost_per_1000"] for r in ft], label="finetuned, model alone")
        ax.plot(costing.THRESHOLDS, [r["cost_per_1000"] for r in co], label="COCO zero-shot (person proxy)", alpha=0.6)
        best = min(ft, key=lambda r: r["cost_per_frame"])
        ax.scatter([best["t_high"]], [best["cost_per_1000"]], c="green", zorder=5,
                   label=f"minimum at these costs: t={best['t_high']:.2f}")
        chosen = costing.evaluate(SCORES, LABELS, CHOSEN, CHOSEN, a)
        ax.axvline(CHOSEN, c="red", lw=1)
        ax.scatter([CHOSEN], [chosen["cost_per_1000"]], c="red", zorder=6, marker="D",
                   label=f"chosen threshold (report) t={CHOSEN:.2f}")
    alts = costing.alternatives(a)
    ax.axhline(alts["do_nothing"]["cost_per_1000"], ls=":", c="grey", label="do nothing")
    ax.axhline(alts["human_checks_everything"]["cost_per_1000"], ls="--", c="grey", label="human checks everything")
    ax.axvspan(t_low, t_high, color="orange", alpha=0.2, label="current review band")
    ax.set_xlabel(f"threshold on max '{POS}' confidence")
    ax.set_ylabel("EUR per 1,000 frames")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
 
 
def update_policy(t_low, t_high, base_rate, cost_fn, cost_fp, cost_review, frames):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    a = assumptions_from(base_rate, cost_fn, cost_fp, cost_review, frames)
    if not len(SCORES):
        return "No artifacts/frame_scores.csv. Run `main.py train` first.", None, None, cost_plot(a, t_low, t_high)
    r = costing.evaluate(SCORES, LABELS, t_low, t_high, a)
    alone = costing.evaluate(SCORES, LABELS, CHOSEN, CHOSEN, a)
    alts = costing.alternatives(a)
 
    npos, nneg = r["pos_n"], r["neg_n"]
    cm = pd.DataFrame(
        {
            "PASS": [round(r["pos_pass"] * npos), round(r["neg_pass"] * nneg)],
            "REVIEW": [round(r["pos_review"] * npos), round(r["neg_review"] * nneg)],
            "ALERT": [round(r["pos_alert"] * npos), round(r["neg_alert"] * nneg)],
            "PASS per 1,000 prod.": [r["pos_pass"] * base_rate * 1000, r["neg_pass"] * (1 - base_rate) * 1000],
            "REVIEW per 1,000 prod.": [r["pos_review"] * base_rate * 1000, r["neg_review"] * (1 - base_rate) * 1000],
            "ALERT per 1,000 prod.": [r["pos_alert"] * base_rate * 1000, r["neg_alert"] * (1 - base_rate) * 1000],
        },
        index=[f"truth: violation ({npos} test frames)", f"truth: compliant ({nneg} test frames)"],
    ).round(1).reset_index(names="")
 
    capacity_ok = r["reviewers_needed"] <= a["reviewers_available"]
    summary = (
        f"### Expected cost: **EUR {r['cost_per_1000']:.0f} per 1,000 frames** (EUR {r['cost_per_day']:.0f}/day)\n"
        f"- Sent to human review: **{r['review_share']:.1%}** of frames = {r['reviews_per_day']:.0f}/day "
        f"= {r['review_hours_per_day']:.1f} h = **{r['reviewers_needed']:.2f} reviewers** "
        f"({'fits' if capacity_ok else 'EXCEEDS'} the {a['reviewers_available']} available)\n"
        f"- Missed violations: {r['missed_violations_per_day']:.1f}/day · false alerts: {r['false_alerts_per_day']:.1f}/day\n"
        f"- Base rate {base_rate:.0%}, FN EUR {cost_fn:.0f}, FP EUR {cost_fp:.2f}, review EUR {cost_review:.2f}, "
        f"{frames} frames/day"
    )
    compare = pd.DataFrame(
        [
            ["do nothing", alts["do_nothing"]["cost_per_1000"], alts["do_nothing"]["cost_per_day"], 0.0],
            ["human checks everything", alts["human_checks_everything"]["cost_per_1000"],
             alts["human_checks_everything"]["cost_per_day"], alts["human_checks_everything"]["reviewers_needed"]],
            [f"model alone (t={CHOSEN:.2f})", alone["cost_per_1000"], alone["cost_per_day"], 0.0],
            [f"model + review ({t_low:.2f}-{t_high:.2f})", r["cost_per_1000"], r["cost_per_day"], r["reviewers_needed"]],
        ],
        columns=["policy", "EUR / 1,000 frames", "EUR / day", "reviewers needed"],
    ).round(2)
    return summary, cm, compare, cost_plot(a, t_low, t_high)
 
 
# --------------------------------------------------------------------------- #
# Tab 3: per-class results
# --------------------------------------------------------------------------- #
def per_class_table():
    if not RES:
        return pd.DataFrame()
    rows = []
    for m, v in RES["models"].items():
        for c, d in v["detection"]["per_class"].items():
            rows.append([m, c, d["instances"], d["precision"], d["recall"], d["AP50"], d["AP50_95"]])
    return pd.DataFrame(rows, columns=["model", "class", "test instances", "precision", "recall", "AP50", "AP50-95"])
 
 
def overall_table():
    if not RES:
        return pd.DataFrame()
    rows = [[m, v["detection"]["overall"]["mAP50"], v["detection"]["overall"]["mAP50_95"], round(v["frame_auc"], 3),
             round(v["best_single"]["cost_per_1000"], 1)] for m, v in RES["models"].items()]
    return pd.DataFrame(rows, columns=["model", "mAP50", "mAP50-95", f"frame AUC ('{POS}' present)", "min EUR / 1,000"])
 
 
def class_cm_table():
    if not RES:
        return pd.DataFrame()
    cm = RES["models"]["finetuned"]["class_confusion"]
    df = pd.DataFrame(cm["matrix"], columns=cm["cols_pred"], index=[f"true {r}" for r in cm["rows_true"]])
    return df.reset_index(names="")
 
 
def dataset_table():
    if not RES:
        return pd.DataFrame()
    rows = []
    for s, v in RES["dataset"]["splits"].items():
        rows.append([s, v["images"]] + [v["instances"][n] for n in RES["names"]])
    return pd.DataFrame(rows, columns=["split", "images"] + [f"{n} instances" for n in RES["names"]])
 
 
# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Hard-hat compliance") as demo:
    gr.Markdown(
        "# Hard-hat compliance: pass / review / alert\n"
        f"Each camera frame gets a score = the model's highest confidence that it sees a **{POS} without a helmet**. "
        "Below the lower threshold it passes, above the upper one a supervisor is sent, in between a person reviews it."
        + (f"  \nDataset: `{RES['dataset_ref']['workspace']}/{RES['dataset_ref']['project']}/v{RES['dataset_ref']['version']}`"
           if RES else "  \n**No artifacts found: run `main.py train` first.**")
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=BAND[0], step=0.01, label="lower threshold (below = PASS)")
        t_high = gr.Slider(0, 1, value=BAND[1], step=0.01, label="upper threshold (above = ALERT)")
    with gr.Accordion("Cost assumptions (defaults = REPORT.md)", open=False):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.5, value=DEFAULT["base_rate"], step=0.005, label="base rate of violation frames")
            cost_fn = gr.Number(value=DEFAULT["cost_fn"], label="EUR per missed violation (FN)")
            cost_fp = gr.Number(value=round(DEFAULT["cost_fp"], 2), label="EUR per supervisor intervention (FP and TP)")
            cost_review = gr.Number(value=round(DEFAULT["cost_review"], 2), label="EUR per human review")
            frames = gr.Number(value=DEFAULT["frames_per_day"], label="frames per day", precision=0)
 
    with gr.Tab("Operating point"):
        summary = gr.Markdown()
        cm = gr.Dataframe(label="Confusion matrix on the test split (counts) and per 1,000 production frames")
        plot = gr.Plot(label="Cost against threshold")
        compare = gr.Dataframe(label="Against the alternatives")
 
    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="upload a site photo")
            img_out = gr.Image(label="predictions (boxes shown down to the lower threshold)")
        verdict = gr.Markdown()
        dets = gr.Dataframe(label="detections")
        gr.Button("Run").click(run_image, [img_in, t_low, t_high], [img_out, verdict, dets])
 
    with gr.Tab("Per-class results"):
        gr.Dataframe(value=overall_table(), label="Overall, test split")
        gr.Dataframe(value=per_class_table(),
                     label=f"Per class (precision/recall at box conf >= {RES['models']['finetuned']['detection']['box_conf_for_P_R'] if RES else 0.25}, IoU 0.5)")
        gr.Dataframe(value=class_cm_table(), label="Finetuned detection confusion matrix (rows = truth)")
        gr.Dataframe(value=dataset_table(), label="Dataset composition")
        if (ART / "ultralytics_confusion_matrix.png").exists():
            gr.Image(value=str(ART / "ultralytics_confusion_matrix.png"), label="Ultralytics confusion matrix (test)")
 
    inputs = [t_low, t_high, base_rate, cost_fn, cost_fp, cost_review, frames]
    outputs = [summary, cm, compare, plot]
    for comp in inputs:
        comp.change(update_policy, inputs, outputs)
    for comp in (t_low, t_high):
        comp.change(run_image, [img_in, t_low, t_high], [img_out, verdict, dets])
    demo.load(update_policy, inputs, outputs)
 
 
if __name__ == "__main__":
    demo.launch()