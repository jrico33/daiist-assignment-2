
"""
Assignment 2: training pipeline (hard-hat compliance on construction sites).
 
    uv run python main.py train
 
Start to finish, no manual steps:
  1. Download the Roboflow Universe dataset version named below.
  2. Describe it (instances per class, split sizes, object sizes).
  3. Finetune yolo11s from the COCO-pretrained checkpoint.
  4. Evaluate on the TEST split: finetuned model vs COCO zero-shot vs
     predict-nothing / majority-class.
  5. Turn scores into the frame-level decision, sweep the threshold, cost it,
     pick the single threshold and the review band.
  6. Save weights/best.pt and everything app.py needs into artifacts/.
 
Environment variables (all optional):
  EPOCHS=40        training epochs
  BATCH=16         batch size (-1 lets Ultralytics pick for your GPU)
  MODEL=yolo11s.pt base checkpoint (yolo11n.pt for a faster, weaker run)
  REUSE_WEIGHTS=1  skip training and evaluate the committed weights/best.pt
  LOCAL_DATASET=path  use an already-downloaded YOLO dataset folder
"""
 
from __future__ import annotations
 
import csv
import json
import os
import shutil
import time
from pathlib import Path
 
import numpy as np
import yaml
 
import costing
 
# --------------------------------------------------------------------------- #
# Dataset: Roboflow Universe, used as published (not forked).
# --------------------------------------------------------------------------- #
WORKSPACE = "computer-vision-filqz"
PROJECT = "hard-hat-detection-62rrp"
VERSION = 2
POSITIVE_CLASS = "head"  # a visible head with no helmet = violation
 
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
WEIGHTS_DIR = ROOT / "weights"
ART_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"
 
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = int(os.environ.get("BATCH", "16"))
IMGSZ = 640
BASE_MODEL = os.environ.get("MODEL", "yolo11s.pt")
SEED = 0
OPERATING_CONF = 0.25  # box confidence used for per-class precision/recall tables
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
# =========================================================================== #
# 1. Data
# =========================================================================== #
def download_dataset() -> Path:
    local = os.environ.get("LOCAL_DATASET")
    if local:
        print(f"Using local dataset at {local}")
        return Path(local).resolve()
 
    from roboflow import Roboflow
 
    from api_key import load_api_key
 
    target = DATA_DIR / f"{PROJECT}-{VERSION}"
    if (target / "data.yaml").exists():
        print(f"Dataset already downloaded at {target}")
        return target
    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    ds = version.download("yolov11", location=str(target), overwrite=True)
    return Path(ds.location).resolve()
 
 
def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
 
 
def label_for(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")
 
 
def _move(imgs: list[Path], ds: Path, split: str) -> None:
    (ds / split / "images").mkdir(parents=True, exist_ok=True)
    (ds / split / "labels").mkdir(parents=True, exist_ok=True)
    for img in imgs:
        lbl = label_for(img)
        shutil.move(str(img), ds / split / "images" / img.name)
        if lbl.exists():
            shutil.move(str(lbl), ds / split / "labels" / lbl.name)
 
 
def ensure_splits(ds: Path) -> str:
    """Make sure train / valid / test all exist, without ever touching an
    existing test split. Deterministic (sorted by file name), so anyone
    re-running gets the same split.
 
    - No valid split (this version ships 75/25 train/test): every 7th train
      image (~15%) moves to valid. Valid is only used for early stopping and
      picking best.pt, so the test split stays a clean hold-out.
    - No test split: every 2nd valid image moves to test.
    """
    notes = []
    if not images_in(ds / "valid" / "images"):
        _move(images_in(ds / "train" / "images")[::7], ds, "valid")
        notes.append("no valid split shipped; every 7th train image (~15%) moved to valid")
    if not images_in(ds / "test" / "images"):
        _move(images_in(ds / "valid" / "images")[1::2], ds, "test")
        notes.append("no test split shipped; every 2nd valid image moved to test")
    return "; ".join(notes) or "train/valid/test all shipped with the dataset version"
 
 
def write_data_yaml(ds: Path) -> tuple[Path, list[str]]:
    with open(ds / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    fixed = {
        "path": str(ds),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    out = ds / "data_fixed.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(fixed, f)
    return out, list(names)
 
 
def read_labels(img: Path) -> np.ndarray:
    """Ground truth as [cls, x1, y1, x2, y2] in normalised coordinates."""
    lbl = label_for(img)
    rows = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:
                x, y, w, h = vals
                rows.append([c, x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            else:  # polygon -> bounding box
                xs, ys = vals[0::2], vals[1::2]
                rows.append([c, min(xs), min(ys), max(xs), max(ys)])
    return np.array(rows, dtype=float).reshape(-1, 5)
 
 
def describe_dataset(ds: Path, names: list[str], split_note: str) -> dict:
    stats = {"split_note": split_note, "splits": {}}
    for split in ("train", "valid", "test"):
        imgs = images_in(ds / split / "images")
        inst = np.zeros(len(names), dtype=int)
        imgs_with = np.zeros(len(names), dtype=int)
        areas = {n: [] for n in names}
        empty = 0
        for img in imgs:
            gt = read_labels(img)
            if len(gt) == 0:
                empty += 1
            present = set()
            for c, x1, y1, x2, y2 in gt:
                c = int(c)
                inst[c] += 1
                present.add(c)
                areas[names[c]].append((x2 - x1) * (y2 - y1) * 100)
            for c in present:
                imgs_with[c] += 1
        size = {}
        for n, a in areas.items():
            a = np.array(a)
            size[n] = (
                {
                    "median_pct_of_image": round(float(np.median(a)), 3),
                    "p10_pct": round(float(np.percentile(a, 10)), 3),
                    "p90_pct": round(float(np.percentile(a, 90)), 3),
                    "share_tiny_lt_0.1pct": round(float(np.mean(a < 0.1)), 3),
                }
                if len(a)
                else {}
            )
        stats["splits"][split] = {
            "images": len(imgs),
            "images_without_boxes": empty,
            "instances": {n: int(inst[i]) for i, n in enumerate(names)},
            "images_containing": {n: int(imgs_with[i]) for i, n in enumerate(names)},
            "object_size": size,
        }
    test = stats["splits"]["test"]
    stats["test_positive_frame_share"] = round(
        test["images_containing"].get(POSITIVE_CLASS, 0) / max(test["images"], 1), 3
    )
    return stats
 
 
# =========================================================================== #
# 2. Training
# =========================================================================== #
def finetune(data_yaml: Path) -> tuple[Path, dict]:
    from ultralytics import YOLO
 
    WEIGHTS_DIR.mkdir(exist_ok=True)
    final = WEIGHTS_DIR / "best.pt"
    settings = {
        "base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
        "patience": 10, "seed": SEED, "optimizer": "auto (Ultralytics default)",
    }
    if os.environ.get("REUSE_WEIGHTS") == "1" and final.exists():
        print(f"REUSE_WEIGHTS=1: skipping training, using {final}")
        return final, {**settings, "note": "reused committed weights"}
 
    import torch
 
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    settings["device"] = str(torch.cuda.get_device_name(0) if device == 0 else device)
    print(f"Training {BASE_MODEL} for {EPOCHS} epochs on {settings['device']}")
    started = time.time()
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=10,
        seed=SEED, deterministic=True, device=device, project=str(RUNS_DIR),
        name="finetune", exist_ok=True, plots=True, verbose=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, final)
    settings["train_minutes"] = round((time.time() - started) / 60, 1)
    settings["run_dir"] = str(Path(model.trainer.save_dir).relative_to(ROOT))
    print(f"Saved {final}")
    return final, settings
 
 
# =========================================================================== #
# 3. Evaluation (one evaluator for all three models, so the comparison is fair)
# =========================================================================== #
def predict(model, imgs: list[Path], class_map: dict[int, int]) -> list[np.ndarray]:
    """Predictions per image as [cls, conf, x1, y1, x2, y2] (normalised),
    with model classes mapped to dataset classes. Unmapped classes dropped."""
    out = []
    for r in model.predict(source=[str(p) for p in imgs], conf=0.001, imgsz=IMGSZ,
                           stream=True, verbose=False, max_det=300):
        b = r.boxes
        if b is None or len(b) == 0:
            out.append(np.zeros((0, 6)))
            continue
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        xyxy = b.xyxyn.cpu().numpy()
        keep = np.array([c in class_map for c in cls], dtype=bool)
        mapped = np.array([class_map.get(c, -1) for c in cls])
        arr = np.column_stack([mapped, conf, xyxy])[keep]
        out.append(arr.reshape(-1, 6))
    return out
 
 
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a [N,4] and b [M,4] in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
 
 
IOU_THRS = np.linspace(0.5, 0.95, 10)
 
 
def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.flip(np.maximum.accumulate(np.flip(p)))
    x = np.linspace(0, 1, 101)  # COCO 101-point interpolation
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, r, p), x))
 
 
def detection_metrics(preds: list[np.ndarray], gts: list[np.ndarray], names: list[str]) -> dict:
    per_class = {}
    for c, name in enumerate(names):
        n_gt = int(sum((g[:, 0] == c).sum() for g in gts))
        confs, tps = [], []
        for p, g in zip(preds, gts):
            pc = p[p[:, 0] == c]
            gc = g[g[:, 0] == c]
            if len(pc) == 0:
                continue
            pc = pc[np.argsort(-pc[:, 1])]
            ious = iou(pc[:, 2:], gc[:, 1:])
            tp = np.zeros((len(pc), len(IOU_THRS)), dtype=bool)
            for k, thr in enumerate(IOU_THRS):
                taken = np.zeros(len(gc), dtype=bool)
                for i in range(len(pc)):
                    if len(gc) == 0:
                        break
                    cand = np.where(~taken & (ious[i] >= thr))[0]
                    if len(cand):
                        j = cand[np.argmax(ious[i, cand])]
                        taken[j] = True
                        tp[i, k] = True
            confs.append(pc[:, 1])
            tps.append(tp)
        if n_gt == 0:
            per_class[name] = {"instances": 0, "AP50": None, "AP50_95": None, "precision": None, "recall": None}
            continue
        if not confs:
            per_class[name] = {"instances": n_gt, "AP50": 0.0, "AP50_95": 0.0, "precision": 0.0, "recall": 0.0}
            continue
        conf = np.concatenate(confs)
        tp = np.concatenate(tps)
        order = np.argsort(-conf)
        conf, tp = conf[order], tp[order]
        aps = []
        for k in range(len(IOU_THRS)):
            ctp = np.cumsum(tp[:, k])
            cfp = np.cumsum(~tp[:, k])
            aps.append(average_precision(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1)))
        at = conf >= OPERATING_CONF
        tp50 = tp[at, 0].sum()
        per_class[name] = {
            "instances": n_gt,
            "AP50": round(aps[0], 4),
            "AP50_95": round(float(np.mean(aps)), 4),
            "precision": round(float(tp50 / max(at.sum(), 1)), 4),
            "recall": round(float(tp50 / n_gt), 4),
        }
    valid = [v for v in per_class.values() if v["AP50"] is not None]
    overall = {
        "mAP50": round(float(np.mean([v["AP50"] for v in valid])), 4) if valid else 0.0,
        "mAP50_95": round(float(np.mean([v["AP50_95"] for v in valid])), 4) if valid else 0.0,
    }
    return {"per_class": per_class, "overall": overall, "box_conf_for_P_R": OPERATING_CONF}
 
 
def class_confusion(preds: list[np.ndarray], gts: list[np.ndarray], names: list[str]) -> dict:
    """Detection confusion matrix at OPERATING_CONF and IoU 0.5.
    Rows = true class (+ 'background' for false boxes), cols = predicted class (+ 'missed')."""
    k = len(names)
    m = np.zeros((k + 1, k + 1), dtype=int)  # last row: background, last col: missed
    for p, g in zip(preds, gts):
        p = p[p[:, 1] >= OPERATING_CONF]
        ious = iou(g[:, 1:], p[:, 2:])
        used = np.zeros(len(p), dtype=bool)
        for i in range(len(g)):
            row = ious[i] if len(p) else np.array([])
            cand = np.where(~used & (row >= 0.5))[0] if len(p) else []
            if len(cand):
                j = cand[np.argmax(row[cand])]
                used[j] = True
                m[int(g[i, 0]), int(p[j, 0])] += 1
            else:
                m[int(g[i, 0]), k] += 1
        for j in np.where(~used)[0]:
            m[k, int(p[j, 0])] += 1
    labels = list(names)
    return {"rows_true": labels + ["background"], "cols_pred": labels + ["missed"], "matrix": m.tolist()}
 
 
def frame_scores(preds: list[np.ndarray], pos_id: int) -> np.ndarray:
    """Frame-level violation score = max confidence of a positive-class box."""
    return np.array([p[p[:, 0] == pos_id, 1].max() if np.any(p[:, 0] == pos_id) else 0.0 for p in preds])
 
 
def frame_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)
 
 
# =========================================================================== #
# 4. Costing
# =========================================================================== #
def sensitivity(scores, labels) -> dict:
    a0 = costing.ASSUMPTIONS
    out = {"cost_fn": [], "base_rate": []}
    for fn in [25, 50, 100, 150, 300, 600, 1000]:
        a = {**a0, "cost_fn": fn}
        s = costing.best_single_threshold(scores, labels, a)
        alt = costing.alternatives(a)
        out["cost_fn"].append({
            "cost_fn": fn, "best_threshold": s["t_high"], "model_alone_per_1000": round(s["cost_per_1000"], 1),
            "do_nothing_per_1000": round(alt["do_nothing"]["cost_per_1000"], 1),
            "human_all_per_1000": round(alt["human_checks_everything"]["cost_per_1000"], 1),
        })
    for pi in [0.01, 0.02, 0.05, 0.10, 0.20]:
        a = {**a0, "base_rate": pi}
        s = costing.best_single_threshold(scores, labels, a)
        out["base_rate"].append({
            "base_rate": pi, "best_threshold": s["t_high"], "model_alone_per_1000": round(s["cost_per_1000"], 1),
        })
    return out
 
 
def plot_cost_curve(sweeps: dict, chosen: dict, band: dict, alts: dict, path: Path) -> None:
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, sw in sweeps.items():
        ax.plot([r["t_high"] for r in sw], [r["cost_per_1000"] for r in sw], label=name)
    ax.axhline(alts["do_nothing"]["cost_per_1000"], ls=":", c="grey", label="do nothing")
    ax.axhline(alts["human_checks_everything"]["cost_per_1000"], ls="--", c="grey", label="human checks everything")
    ax.axvline(chosen["t_high"], c="red", lw=1)
    ax.scatter([chosen["t_high"]], [chosen["cost_per_1000"]], c="red", zorder=5,
               label=f"chosen t={chosen['t_high']:.2f} (EUR {chosen['cost_per_1000']:.0f}/1000)")
    ax.axvspan(band["t_low"], band["t_high"], color="orange", alpha=0.15,
               label=f"review band {band['t_low']:.2f}-{band['t_high']:.2f}")
    ax.set_xlabel("confidence threshold on max 'head' score")
    ax.set_ylabel("expected cost, EUR per 1,000 frames")
    ax.set_title("Cost against threshold (test split, re-weighted to production base rate)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
 
 
# =========================================================================== #
# 5. Main
# =========================================================================== #
def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"
 
 
def write_report_tables(res: dict, path: Path) -> None:
    """Markdown tables to paste into REPORT.md."""
    L = []
    ds = res["dataset"]
    L.append("### Dataset composition\n")
    L.append("| split | images | " + " | ".join(f"{n} inst." for n in res["names"]) + " | frames with " + POSITIVE_CLASS + " |")
    L.append("|---|---|" + "---|" * len(res["names"]) + "---|")
    for s, v in ds["splits"].items():
        L.append(f"| {s} | {v['images']} | " + " | ".join(str(v["instances"][n]) for n in res["names"])
                 + f" | {v['images_containing'].get(POSITIVE_CLASS, 0)} |")
    L.append("\n| class | median box area (% of image) | p10 | p90 | share < 0.1% |")
    L.append("|---|---|---|---|---|")
    for n, v in ds["splits"]["train"]["object_size"].items():
        if v:
            L.append(f"| {n} | {v['median_pct_of_image']} | {v['p10_pct']} | {v['p90_pct']} | {v['share_tiny_lt_0.1pct']} |")
    L.append(f"\nSplit note: {ds['split_note']}\n")
 
    L.append("### Models on the test split\n")
    L.append("| model | mAP50 | mAP50-95 | " + " | ".join(f"AP50 {n}" for n in res["names"]) + " | frame AUC | min cost / 1000 frames |")
    L.append("|---|---|---|" + "---|" * len(res["names"]) + "---|---|")
    for m, v in res["models"].items():
        d = v["detection"]
        L.append(f"| {m} | {d['overall']['mAP50']:.3f} | {d['overall']['mAP50_95']:.3f} | "
                 + " | ".join(fmt(d["per_class"][n]["AP50"]) for n in res["names"])
                 + f" | {v['frame_auc']:.3f} | EUR {v['best_single']['cost_per_1000']:.0f} |")
 
    L.append(f"\n### Finetuned per-class (box conf >= {OPERATING_CONF}, IoU 0.5)\n")
    L.append("| class | instances | precision | recall | AP50 | AP50-95 |")
    L.append("|---|---|---|---|---|---|")
    for n, v in res["models"]["finetuned"]["detection"]["per_class"].items():
        L.append(f"| {n} | {v['instances']} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['AP50'])} | {fmt(v['AP50_95'])} |")
    cm = res["models"]["finetuned"]["class_confusion"]
    L.append("\nDetection confusion matrix (rows = truth, cols = prediction):\n")
    L.append("| | " + " | ".join(cm["cols_pred"]) + " |")
    L.append("|---|" + "---|" * len(cm["cols_pred"]))
    for r, row in zip(cm["rows_true"], cm["matrix"]):
        L.append(f"| **{r}** | " + " | ".join(str(x) for x in row) + " |")
 
    c, b, bc = res["chosen_single"], res["band_unconstrained"], res["band_chosen"]
    alt = res["alternatives"]
    a = res["assumptions"]
    L.append("\n### Operating point and policies (per day = "
             f"{a['frames_per_day']} frames, base rate {a['base_rate']:.0%})\n")
    L.append("| policy | thresholds | EUR / 1000 | EUR / day | review share | reviewers needed | missed violations/day | false alerts/day |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append(f"| do nothing | - | {alt['do_nothing']['cost_per_1000']:.0f} | {alt['do_nothing']['cost_per_day']:.0f} | 0% | 0 | {a['base_rate']*a['frames_per_day']:.0f} | 0 |")
    L.append(f"| human checks everything | - | {alt['human_checks_everything']['cost_per_1000']:.0f} | {alt['human_checks_everything']['cost_per_day']:.0f} | 100% | {alt['human_checks_everything']['reviewers_needed']:.2f} | 0 | 0 |")
    for label, r in (("model alone", c), ("model + review (unconstrained)", b), (f"model + review (<= {a['reviewers_available']} reviewer)", bc)):
        L.append(f"| {label} | {r['t_low']:.2f} / {r['t_high']:.2f} | {r['cost_per_1000']:.0f} | {r['cost_per_day']:.0f} | "
                 f"{r['review_share']:.1%} | {r['reviewers_needed']:.2f} | {r['missed_violations_per_day']:.1f} | {r['false_alerts_per_day']:.1f} |")
 
    L.append("\n### Sensitivity: cost of a missed violation\n")
    L.append("| cost_fn | best threshold | model alone /1000 | do nothing /1000 | human all /1000 |")
    L.append("|---|---|---|---|---|")
    for r in res["sensitivity"]["cost_fn"]:
        L.append(f"| {r['cost_fn']} | {r['best_threshold']:.2f} | {r['model_alone_per_1000']} | {r['do_nothing_per_1000']} | {r['human_all_per_1000']} |")
    L.append("\n### Sensitivity: base rate\n")
    L.append("| base rate | best threshold | model alone /1000 |")
    L.append("|---|---|---|")
    for r in res["sensitivity"]["base_rate"]:
        L.append(f"| {r['base_rate']:.0%} | {r['best_threshold']:.2f} | {r['model_alone_per_1000']} |")
    path.write_text("\n".join(L) + "\n")
 
 
def main() -> None:
    from ultralytics import YOLO
 
    ART_DIR.mkdir(exist_ok=True)
 
    # 1. Data
    ds = download_dataset()
    split_note = ensure_splits(ds)
    data_yaml, names = write_data_yaml(ds)
    if POSITIVE_CLASS not in names:
        raise SystemExit(f"Positive class '{POSITIVE_CLASS}' not in dataset classes {names}")
    pos_id = names.index(POSITIVE_CLASS)
    stats = describe_dataset(ds, names, split_note)
    print(json.dumps(stats, indent=2))
 
    # 2. Finetune
    weights, train_settings = finetune(data_yaml)
 
    # 3. Evaluate all models on the same test images
    test_imgs = images_in(ds / "test" / "images")
    gts = [read_labels(p) for p in test_imgs]
    labels = np.array([int(np.any(g[:, 0] == pos_id)) for g in gts])
 
    finetuned = YOLO(str(weights))
    ft_preds = predict(finetuned, test_imgs, {i: i for i in range(len(names))})
 
    coco = YOLO(BASE_MODEL)  # same architecture, COCO weights, zero-shot
    coco_map = {0: names.index("person")} if "person" in names else {}
    coco_preds = predict(coco, test_imgs, coco_map)
 
    nothing_preds = [np.zeros((0, 6)) for _ in test_imgs]
 
    # Frame-level scores. COCO has no 'head' class: its best available proxy is
    # "flag the frame if it sees a person", so its score is max person conf.
    person_id = coco_map.get(0, -1)
    majority = int(labels.mean() >= 0.5)
    frame = {
        "finetuned": frame_scores(ft_preds, pos_id),
        "coco_zero_shot": frame_scores(coco_preds, person_id),
        "predict_nothing": np.zeros(len(test_imgs)),
        "majority_class": np.full(len(test_imgs), float(majority)),
    }
    det = {
        "finetuned": ft_preds, "coco_zero_shot": coco_preds,
        "predict_nothing": nothing_preds, "majority_class": nothing_preds,
    }
 
    models = {}
    sweeps = {}
    for name in frame:
        sw = costing.single_threshold_sweep(frame[name], labels)
        sweeps[name] = sw
        models[name] = {
            "detection": detection_metrics(det[name], gts, names),
            "frame_auc": frame_auc(frame[name], labels),
            "best_single": min(sw, key=lambda r: r["cost_per_frame"]),
        }
    models["finetuned"]["class_confusion"] = class_confusion(ft_preds, gts, names)
    models["coco_zero_shot"]["class_confusion"] = class_confusion(coco_preds, gts, names)
 
    # Official Ultralytics numbers for the finetuned model, as a cross-check.
    try:
        v = finetuned.val(data=str(data_yaml), split="test", imgsz=IMGSZ, plots=True,
                          project=str(RUNS_DIR), name="test_eval", exist_ok=True, verbose=False)
        models["finetuned"]["ultralytics_test"] = {
            "mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
            "per_class_AP50": {names[i]: round(float(x), 4) for i, x in zip(v.box.ap_class_index, v.box.ap50)},
        }
        cm_png = Path(v.save_dir) / "confusion_matrix.png"
        if cm_png.exists():
            shutil.copy(cm_png, ART_DIR / "ultralytics_confusion_matrix.png")
    except Exception as e:  # noqa: BLE001 - cross-check only
        print(f"Ultralytics val cross-check skipped: {e}")
 
    # 4. Costing on the finetuned model
    s, y = frame["finetuned"], labels
    chosen = models["finetuned"]["best_single"]
    band_free = costing.best_band(s, y)
    band_cap = costing.best_band(s, y, max_reviewers=costing.ASSUMPTIONS["reviewers_available"])
    alts = costing.alternatives()
    plot_cost_curve({k: sweeps[k] for k in ("finetuned", "coco_zero_shot")}, chosen, band_cap, alts,
                    ART_DIR / "cost_curve.png")
 
    # 5. Save
    with open(ART_DIR / "frame_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "label_has_head", "score_finetuned", "score_coco_zero_shot"])
        for p, l, a1, a2 in zip(test_imgs, labels, frame["finetuned"], frame["coco_zero_shot"]):
            w.writerow([p.name, int(l), f"{a1:.5f}", f"{a2:.5f}"])
    with open(ART_DIR / "cost_sweep.csv", "w", newline="") as f:
        keys = ["t_high", "cost_per_1000", "review_share", "pos_pass", "pos_alert", "neg_alert"]
        w = csv.writer(f)
        w.writerow(["model"] + keys)
        for m, sw in sweeps.items():
            for r in sw:
                w.writerow([m] + [round(r[k], 5) for k in keys])
 
    results = {
        "dataset_ref": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                        "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"},
        "names": names, "positive_class": POSITIVE_CLASS,
        "dataset": stats, "train_settings": train_settings,
        "models": models,
        "chosen_single": chosen, "band_unconstrained": band_free, "band_chosen": band_cap,
        "alternatives": alts, "assumptions": costing.ASSUMPTIONS,
        "sensitivity": sensitivity(s, y),
        "test_frames": int(len(labels)), "test_positive_frames": int(labels.sum()),
    }
    with open(ART_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    write_report_tables(results, ART_DIR / "report_tables.md")
    print((ART_DIR / "report_tables.md").read_text())
    print("Done. Commit weights/best.pt and artifacts/.")
 
 
if __name__ == "__main__":
    main()