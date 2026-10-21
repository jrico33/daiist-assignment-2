
"""
Business assumptions and cost maths, shared by train.py and app.py.
 
Everything that turns a model score into euros lives here, so the training
pipeline and the dashboard can never disagree about what a mistake costs.
 
The decision unit is ONE CAMERA FRAME of a construction site.
  - Positive class  = the frame shows at least one worker WITHOUT a hard hat
                      (at least one "head" box in the ground truth).
  - Model score     = the highest "head" confidence the model gives in that frame.
  - Three outcomes  = score <  t_low            -> PASS   (nothing happens)
                      t_low <= score < t_high   -> REVIEW (a person looks at the frame)
                      score >= t_high           -> ALERT  (supervisor goes to the worker)
"""
 
from __future__ import annotations
 
import numpy as np
 
# --------------------------------------------------------------------------- #
# Assumptions. Every number is defended in REPORT.md. Change them here only.
# --------------------------------------------------------------------------- #
SUPERVISOR_EUR_PER_HOUR = 45.0   # loaded cost of a site supervisor in Spain
INTERVENTION_MINUTES = 10.0      # walk to the worker, stop work, check, walk back
REVIEWER_EUR_PER_HOUR = 30.0     # loaded cost of a remote safety-desk operator
REVIEW_SECONDS = 20.0            # look at one flagged frame and decide
 
ASSUMPTIONS = {
    # Cost of each outcome, in euros, per frame.
    "cost_fn": 150.0,  # missed worker without helmet (expected fine + injury cost)
    "cost_fp": SUPERVISOR_EUR_PER_HOUR * INTERVENTION_MINUTES / 60,  # 7.50
    "cost_tp": SUPERVISOR_EUR_PER_HOUR * INTERVENTION_MINUTES / 60,  # 7.50, same walk
    "cost_tn": 0.0,
    "cost_review": REVIEWER_EUR_PER_HOUR * REVIEW_SECONDS / 3600,    # 0.17
    # How often the positive class happens in production (share of frames).
    "base_rate": 0.05,
    # Volume: 10 cameras x 1 frame every 2 minutes x 10-hour shift.
    "frames_per_day": 3000,
    "review_seconds": REVIEW_SECONDS,
    "reviewer_productive_hours": 6.5,  # of an 8h shift, after breaks/admin
    "reviewers_available": 1,
}
 
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)
 
 
def outcome_rates(scores: np.ndarray, labels: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of positives / negatives that land in PASS, REVIEW, ALERT."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    out = {}
    for name, mask in (("pos", labels == 1), ("neg", labels == 0)):
        s = scores[mask]
        n = max(len(s), 1)
        out[f"{name}_alert"] = float(np.sum(s >= t_high)) / n
        out[f"{name}_review"] = float(np.sum((s >= t_low) & (s < t_high))) / n
        out[f"{name}_pass"] = float(np.sum(s < t_low)) / n
        out[f"{name}_n"] = int(mask.sum())
    return out
 
 
def expected_cost(rates: dict, a: dict = ASSUMPTIONS) -> dict:
    """Expected euros per frame at the assumed base rate.
 
    The test set's own positive share is NOT used: rates are measured within
    each true class on the test set, then re-weighted by the production base
    rate. Assumption: the reviewer always gets it right, and a reviewed
    positive then triggers the same intervention as an alert.
    """
    pi = a["base_rate"]
    pos = (
        rates["pos_pass"] * a["cost_fn"]
        + rates["pos_review"] * (a["cost_review"] + a["cost_tp"])
        + rates["pos_alert"] * a["cost_tp"]
    )
    neg = (
        rates["neg_pass"] * a["cost_tn"]
        + rates["neg_review"] * (a["cost_review"] + a["cost_tn"])
        + rates["neg_alert"] * a["cost_fp"]
    )
    per_frame = pi * pos + (1 - pi) * neg
    review_share = pi * rates["pos_review"] + (1 - pi) * rates["neg_review"]
    reviews_per_day = review_share * a["frames_per_day"]
    review_hours = reviews_per_day * a["review_seconds"] / 3600
    return {
        "cost_per_frame": per_frame,
        "cost_per_1000": per_frame * 1000,
        "cost_per_day": per_frame * a["frames_per_day"],
        "review_share": review_share,
        "reviews_per_day": reviews_per_day,
        "review_hours_per_day": review_hours,
        "reviewers_needed": review_hours / a["reviewer_productive_hours"],
        "missed_violations_per_day": pi * rates["pos_pass"] * a["frames_per_day"],
        "false_alerts_per_day": (1 - pi) * rates["neg_alert"] * a["frames_per_day"],
    }
 
 
def evaluate(scores, labels, t_low, t_high, a: dict = ASSUMPTIONS) -> dict:
    rates = outcome_rates(scores, labels, t_low, t_high)
    return {**rates, **expected_cost(rates, a), "t_low": float(min(t_low, t_high)), "t_high": float(max(t_low, t_high))}
 
 
def single_threshold_sweep(scores, labels, a: dict = ASSUMPTIONS) -> list[dict]:
    """Model decides alone: one threshold, no review band (t_low == t_high)."""
    return [evaluate(scores, labels, t, t, a) for t in THRESHOLDS]
 
 
def best_single_threshold(scores, labels, a: dict = ASSUMPTIONS) -> dict:
    sweep = single_threshold_sweep(scores, labels, a)
    return min(sweep, key=lambda r: r["cost_per_frame"])
 
 
def best_band(scores, labels, a: dict = ASSUMPTIONS, max_reviewers: float | None = None) -> dict:
    """Grid-search (t_low, t_high) for the cheapest three-way policy.
 
    If max_reviewers is given, only bands whose review load fits that many
    people are allowed.
    """
    best = None
    for i, lo in enumerate(THRESHOLDS):
        for hi in THRESHOLDS[i:]:
            r = evaluate(scores, labels, lo, hi, a)
            if max_reviewers is not None and r["reviewers_needed"] > max_reviewers + 1e-9:
                continue
            if best is None or r["cost_per_frame"] < best["cost_per_frame"] - 1e-12:
                best = r
    return best
 
 
def alternatives(a: dict = ASSUMPTIONS) -> dict:
    """The three policies the model has to beat, per frame and per day."""
    pi, n = a["base_rate"], a["frames_per_day"]
    nothing = pi * a["cost_fn"] + (1 - pi) * a["cost_tn"]
    human_all = a["cost_review"] + pi * a["cost_tp"] + (1 - pi) * a["cost_tn"]
    human_hours = n * a["review_seconds"] / 3600
    return {
        "do_nothing": {"cost_per_1000": nothing * 1000, "cost_per_day": nothing * n, "reviewers_needed": 0.0},
        "human_checks_everything": {
            "cost_per_1000": human_all * 1000,
            "cost_per_day": human_all * n,
            "reviewers_needed": human_hours / a["reviewer_productive_hours"],
        },
    }
 