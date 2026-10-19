"""Lesson 5: how to fool yourself -- many tests, and picking the folds you like.

Run:   python 05_multiple_testing/cherry_picking.py     (about 30 seconds)

IN PLAIN WORDS
Two classic ways a real analysis "finds" an effect that isn't there. Both use PURE NOISE here,
so any "finding" is false by construction.

Part A: MULTIPLE TESTING. Test 100 random candidate variables against the outcome at p < 0.05
        and about 5 will "pass" by luck. The Benjamini-Hochberg FDR correction asks a stricter
        question ("what share of my passes are false?") and removes them.

Part B: CHERRY-PICKED FOLDS. Cross-validate a model with and without a useless extra variable
        over 50 folds. Averaged over ALL folds, the gain is about zero. Averaged over only the
        folds where the extra variable helped ("positive folds"), there is a gain -- always,
        by construction. That is exactly the bug that inflated a collaborator's AUCs in your
        NeuroPathology work, which you found and which is documented in that repo.

Part C: the NULL CONTROL. Shuffle the outcome labels and run the same pipeline. Any "gain" that
        survives shuffled labels is produced by the pipeline, not by biology.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import ground_truth  # noqa: E402


def base_features(gt):
    """Three AD staging variables, missing values filled with the column median."""
    X = gt[["braak_stage", "thal_phase", "cerad_score"]].astype(float)
    return X.fillna(X.median()).values


def fold_deltas(X, extra, y, seed=0):
    """AUC(base + extra) - AUC(base) on each of 50 folds (5-fold CV repeated 10 times)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    Xe = np.column_stack([X, extra])
    deltas = []
    for tr, te in cv.split(X, y):
        auc = []
        for M in (X, Xe):
            m = LogisticRegression(max_iter=1000).fit(M[tr], y[tr])
            auc.append(roc_auc_score(y[te], m.predict_proba(M[te])[:, 1]))
        deltas.append(auc[1] - auc[0])
    return np.array(deltas)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = ground_truth()
    y = gt["dementia"].values

    print("Part A: 100 random 'candidate variables' vs dementia (none is real)")
    cands = rng.normal(size=(len(y), 100))
    p = np.array([stats.ttest_ind(c[y == 1], c[y == 0]).pvalue for c in cands.T])
    print(f"  p < 0.05 without correction: {(p < 0.05).sum()} variables 'significant' (about 5 expected by luck)")
    q = stats.false_discovery_control(p)                     # Benjamini-Hochberg adjusted p
    print(f"  after BH-FDR (q < 0.05):      {(q < 0.05).sum()} variables")

    print("\nPart B: a useless extra variable, 50 CV folds")
    X = base_features(gt)
    noise = rng.normal(size=len(y))
    d = fold_deltas(X, noise, y)
    print(f"  mean AUC gain over ALL 50 folds:           {d.mean():+.4f}")
    print(f"  mean AUC gain over 'positive folds' only:  {d[d > 0].mean():+.4f}   ({(d > 0).sum()} of 50 folds)")
    print("  The second number is positive for ANY variable, even pure noise.")

    print("\nPart C: null control -- shuffle the outcome, same pipeline")
    y_shuf = rng.permutation(y)
    d0 = fold_deltas(X, noise, y_shuf)
    print(f"  ALL folds, shuffled labels:            {d0.mean():+.4f}")
    print(f"  positive folds only, shuffled labels:  {d0[d0 > 0].mean():+.4f}   <- a 'gain' with no signal at all")

# EXERCISE 1: Part A -- run with 4 different rng seeds. How many false passes do you get each
#   time without correction? With FDR? (Expect ~5 vs ~0.)
# EXERCISE 2: Part B -- replace `noise` with gt["gvd_severity"] (filled with its median). Is the
#   ALL-folds gain different from zero? How would you decide? (Lesson 4: CI over folds is
#   tricky because folds overlap -- a held-out set or a permutation test is cleaner.)
# EXERCISE 3: Write two sentences you could say in an interview: what the positive-fold bug is,
#   and how a null control would have caught it.
# EXERCISE 4 (real code):
#   - BUGFIXES_AND_CORRECTIONS.md lines 121-143 document the cherry-pick and the fix.
#   - analysis/rederived_analysis.py line 101 (scan) screens candidates with BH-FDR (line 111),
#     and line 195 reruns feature selection INSIDE every training fold.
#   - analysis/holdout_selection_validation.py line 137 runs a permuted-label null (Part C).
