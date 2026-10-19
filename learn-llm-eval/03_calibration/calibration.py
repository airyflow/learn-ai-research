"""Lesson 3: when the model says "90% sure", is it right 90% of the time? (calibration)

Run:   python 03_calibration/calibration.py     (3 seconds)

IN PLAIN WORDS
Your extractor writes a confidence (0-1) next to every value. A confidence is CALIBRATED if,
among all answers stated at ~0.9, about 90% are actually right.

Part A: a reliability table -- group answers by stated confidence, compare with accuracy.
Part B: ECE (expected calibration error) -- the average gap, weighted by how many answers
        fall in each group. One number; 0 is perfect.
Part C: two confidence signals compared -- the model's SELF-REPORTED confidence vs SEED
        AGREEMENT (lesson 2). Which separates right from wrong answers better?

The lesson: a model grading its own answers is a proxy, and often an overconfident one.
Measure it before you trust it -- the same rule as "graders must not trust the agent".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import extractions, ground_truth, with_truth  # noqa: E402

BINS = [0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def reliability(conf, correct, bins=BINS):
    t = pd.DataFrame({"conf": conf, "correct": correct})
    t["bin"] = pd.cut(t["conf"], bins, include_lowest=True)
    return t.groupby("bin", observed=True).agg(stated=("conf", "mean"), actual=("correct", "mean"),
                                               n=("correct", "size"))


def ece(conf, correct, bins=BINS):
    r = reliability(conf, correct, bins)
    return float((r["n"] / r["n"].sum() * (r["stated"] - r["actual"]).abs()).sum())


if __name__ == "__main__":
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=("A",), seeds=range(5)))

    print("Part A: reliability table, model A (all 5 seeds)")
    print(reliability(df["confidence"], df["is_correct"]).round(3).to_string())

    print(f"\nPart B: ECE = {ece(df['confidence'], df['is_correct']):.3f}"
          "   (average gap between stated and actual, 0 = perfect)")

    print("\nPart C: which signal separates right from wrong? (AUC: 0.5 = useless, 1 = perfect)")
    key = ["report_id", "field"]
    agree = df.groupby(key)["value"].transform(lambda s: s.astype(str).map(s.astype(str).value_counts()) / len(s))
    print(f"  self-reported confidence: AUC = {roc_auc_score(df['is_correct'], df['confidence']):.3f}")
    print(f"  seed agreement:           AUC = {roc_auc_score(df['is_correct'], agree):.3f}")

# EXERCISE 1: In Part A, find the group where the model is most overconfident. By how much?
# EXERCISE 2: ECE depends on the bins. Try BINS = np.linspace(0, 1, 11). Does ECE change? Always
#   report the binning with an ECE number.
# EXERCISE 3: Part C uses AUC to compare two signals. Which would you use to decide which
#   answers a human should check -- and could you combine them? (Try their average.)
# EXERCISE 4 (real code): your project computes ECE with seed agreement as the confidence:
#   analysis/extra_model_metrics.py line 89 (ece) and line 69 (risk_coverage). The model's own
#   confidence is read at analysis/compute_confidence_metrics.py line 99.
