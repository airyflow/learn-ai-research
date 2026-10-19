"""Lesson 7: do extraction errors matter downstream? (error propagation)

Run:   python 07_error_propagation/error_propagation.py     (about 30 seconds)

IN PLAIN WORDS
Your extracted variables feed a model (in your project: disease-stage classification). A field
that is 65% accurate sounds bad -- but does it change the downstream result? Measure it:

  ground-truth features  -> model -> AUC        (the best you could do with perfect extraction)
  LLM-extracted features -> model -> AUC        (what you actually get)
  the gap = the cost of extraction errors

Part A: AUC with true features vs extracted features (one seed) vs majority vote of 5 seeds,
        on the SAME cross-validation folds, so the comparison is paired.
Part B: which field's errors hurt most? Start from the true features and swap in the LLM's
        values for ONE field at a time.
Small data (160 reports) makes AUC noisy -- read differences under ~0.01 as "no clear effect".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import FIELDS, extractions, ground_truth  # noqa: E402


def to_features(table):
    """One row per report -> numbers: each field as a value (missing -> median) plus a
    'was missing' indicator; the categorical Lewy type as 0/1 columns."""
    out = {}
    for f in FIELDS:
        col = table[f.name]
        if f.kind == "category":
            for level in f.levels[1:]:
                out[f"{f.name}={level}"] = (col == level).astype(float)
        else:
            x = pd.to_numeric(col, errors="coerce")
            out[f"{f.name}_missing"] = x.isna().astype(float)
            out[f.name] = x.fillna(x.median())
    return pd.DataFrame(out).values


def cv_aucs(X, y, folds):
    aucs = []
    for tr, te in folds:
        m = LogisticRegression(max_iter=2000, C=0.5).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)


def table_from(ext, gt, how):
    """Turn extractions back into one row per report: one seed, or the majority vote."""
    if how == "seed0":
        e = ext[ext.seed == 0]
        wide = e.pivot(index="report_id", columns="field", values="value")
    else:
        vote = ext.groupby(["report_id", "field"])["value"].agg(
            lambda s: s[s.astype(str) == s.astype(str).value_counts().index[0]].iloc[0])
        wide = vote.unstack()
    return wide.reindex(gt["report_id"]).reset_index(drop=True)


if __name__ == "__main__":
    gt = ground_truth()
    y = gt["dementia"].values
    ext = extractions(gt, models=("A",), seeds=range(5))
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(gt, y))

    X_true = to_features(gt)
    results = {"true features": cv_aucs(X_true, y, folds),
               "LLM, one seed": cv_aucs(to_features(table_from(ext, gt, "seed0")), y, folds),
               "LLM, 5-seed vote": cv_aucs(to_features(table_from(ext, gt, "vote")), y, folds)}
    print("Part A: downstream AUC (dementia), 25 paired CV folds")
    for name, a in results.items():
        gap = (a - results["true features"]).mean()
        print(f"  {name:17s} AUC {a.mean():.3f}   vs true: {gap:+.3f}")

    print("\nPart B: swap in the LLM's (seed 0) values for ONE field at a time")
    llm = table_from(ext, gt, "seed0")
    rows = []
    for f in FIELDS:
        mixed = gt.copy()
        mixed[f.name] = llm[f.name].values
        a = cv_aucs(to_features(mixed), y, folds)
        acc = np.mean([(t == p) or (pd.isna(t) and pd.isna(p)) for t, p in zip(gt[f.name], llm[f.name])])
        rows.append((f.name, acc, (a - results["true features"]).mean()))
    for name, acc, gap in sorted(rows, key=lambda r: r[2]):
        print(f"  {name:22s} extraction accuracy {acc:.2f}   AUC change {gap:+.4f}")
    print("  A field can be badly extracted and still not matter (if the model barely uses it),")
    print("  or be well extracted and matter a lot. Accuracy alone doesn't tell you which.")
    print("  Positive changes (errors that seem to 'help') are noise: with 160 reports, about")
    print("  +/-0.01 is within chance. Only the larger drops are worth believing.")

# EXERCISE 1: Part A -- does the 5-seed vote recover most of the gap? Is that worth 5x compute?
# EXERCISE 2: Part B -- find a field with LOW accuracy but almost no AUC change. Why? (Look at
#   how dementia is generated in common/fake_reports.py: which fields matter?)
# EXERCISE 3: The AUC differences are small and the folds overlap. How would you put a CI on
#   "vote vs true"? (Lesson 4: a paired bootstrap over reports with a fixed held-out split.)
# EXERCISE 4 (real code): analysis/downstream_auc_by_model.py line 62 does Part A per LLM on your
#   real data; analysis/rederived_analysis.py lines 152-176 put a bootstrap CI on an AUC gain
#   with a single held-out split, and refuse to run with uncommitted changes (lines 153-156).
