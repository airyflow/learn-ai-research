"""Lesson 4: how sure are you of that number? Bootstrap confidence intervals, done right.

Run:   python 04_bootstrap/bootstrap.py     (about 15 seconds)

IN PLAIN WORDS
"Model A is 84% accurate" -- give or take how much? The BOOTSTRAP answers by pretending your
data is the whole world: resample it with replacement many times, recompute the number each
time, and see how much it moves. The middle 95% of those numbers is a 95% CI.

The one decision that matters: WHAT do you resample? Resample the independent unit. Here that
is the REPORT, not the cell. Ten fields from one report share its length, its writing style,
its difficulty -- they are not ten independent tries.

Part A: CI from resampling cells (too narrow) vs resampling reports (honest).
Part B: comparing two models -- PAIRED (same reports, take the difference) vs unpaired.
Part C: how the CI shrinks as you label more reports.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import extractions, ground_truth, with_truth  # noqa: E402

B = 2000


def ci(values):
    return np.percentile(values, [2.5, 97.5])


def boot_cells(correct, rng):
    correct = np.asarray(correct, float)
    return [rng.choice(correct, len(correct)).mean() for _ in range(B)]


def boot_reports(per_report_correct, per_report_n, rng):
    """Resample whole reports; accuracy = total correct / total cells in the resample."""
    k = len(per_report_correct)
    out = []
    for _ in range(B):
        idx = rng.integers(0, k, k)
        out.append(per_report_correct[idx].sum() / per_report_n[idx].sum())
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=("A", "B"), seeds=[0]))
    a, b = df[df.model == "A"], df[df.model == "B"]

    print("Part A: 95% CI for model A's accuracy (seed 0)")
    lo, hi = ci(boot_cells(a["is_correct"], rng))
    print(f"  resample CELLS   (1600 'independent' cells): [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")
    per = a.groupby("report_id")["is_correct"].agg(["sum", "size"])
    lo, hi = ci(boot_reports(per["sum"].values, per["size"].values, rng))
    print(f"  resample REPORTS (160 independent reports):   [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")
    print("  The cell CI is narrower: it pretends you have more independent data than you do.")
    print("  Here the gap is modest because fake reports share little difficulty; in real reports")
    print("  (one bad scan or unusual template affects many fields) it is usually much bigger.\n")

    print("Part B: is A better than B?")
    acc_a = a.groupby("report_id")["is_correct"].mean()
    acc_b = b.groupby("report_id")["is_correct"].mean().reindex(acc_a.index)
    diff = (acc_a - acc_b).values
    paired = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    unpaired = [acc_a.values[rng.integers(0, 160, 160)].mean() - acc_b.values[rng.integers(0, 160, 160)].mean()
                for _ in range(B)]
    print(f"  difference A - B: {diff.mean():+.3f}")
    lo, hi = ci(paired)
    print(f"  PAIRED   (same reports, per-report difference): [{lo:+.3f}, {hi:+.3f}]  width {hi - lo:.3f}")
    lo, hi = ci(unpaired)
    print(f"  UNPAIRED (resample A and B separately):        [{lo:+.3f}, {hi:+.3f}]  width {hi - lo:.3f}")
    print("  Pairing removes 'some reports are just hard for everyone', so the CI is narrower.\n")

    print("Part C: CI width vs number of labeled reports (model A, report-level bootstrap)")
    for n in (20, 40, 80, 160):
        ids = per.index[:n]
        lo, hi = ci(boot_reports(per.loc[ids, "sum"].values, per.loc[ids, "size"].values, rng))
        print(f"  {n:4d} reports: width {hi - lo:.3f}")
    print("  4x the reports -> about half the width (width shrinks like 1/sqrt(n)).")

# EXERCISE 1: In Part A, how much narrower is the cell-level CI? If a paper reported the
#   cell-level one, what would a careful reviewer say?
# EXERCISE 2: Part B -- does the paired CI exclude 0? Then "A is better" is supported. Write the
#   sentence you'd put in a paper, with the number and the CI.
# EXERCISE 3: Using Part C, roughly how many reports would you need to label to get a CI width
#   of 0.02? Is that realistic for neuropathology ground truth?
# EXERCISE 4 (real code): analysis/build_llm_bootstrap_metrics.py -- read the docstring (lines
#   1-22): the report is the resampling unit, B=2000. Lines 135-144 reuse each resample for all
#   3 models, so the differences are PAIRED. You already did Part B in your project.
