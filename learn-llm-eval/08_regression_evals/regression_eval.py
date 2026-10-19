"""Lesson 8: regression evals -- catch a change that makes the pipeline worse.

Run:   python 08_regression_evals/regression_eval.py     (about 5 seconds)

IN PLAIN WORDS
You will keep changing the pipeline: a new prompt, a new model, a schema tweak. A REGRESSION
EVAL is an automatic check you run after every change: "is it still at least as good?" It
needs three things:
  1. a FROZEN GOLD SET    a fixed set of reports with trusted labels, never edited casually
  2. a METRIC             e.g. accuracy on the gold set
  3. a THRESHOLD          how big a drop counts as a failure

The trap is the threshold. LLM runs wobble between seeds (lesson 2). If your threshold is
smaller than that wobble, the test fails on changes that did nothing -- and people learn to
ignore it.

Part A: the NOISE FLOOR -- how much does accuracy wobble on the gold set when NOTHING changed?
Part B: a naive test ("fail if accuracy drops at all") -- false alarms on an unchanged pipeline.
Part C: a better test -- average over seeds, threshold above the noise floor. It passes the
        unchanged pipeline and fails the real regression (model B).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import extractions, ground_truth, with_truth  # noqa: E402

GOLD = [f"R{i:03d}" for i in range(0, 160, 4)]       # a frozen gold set: 40 reports


def gold_accuracy(model, seeds):
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=(model,), seeds=seeds))
    df = df[df["report_id"].isin(GOLD)]
    return df.groupby("seed")["is_correct"].mean()


if __name__ == "__main__":
    base = gold_accuracy("A", range(5))                # the accepted pipeline, 5 seeds
    print("Part A: noise floor -- model A on the gold set, 5 seeds, NOTHING changed")
    print(f"  accuracy per seed: {base.round(3).tolist()}")
    print(f"  spread (max - min): {base.max() - base.min():.3f}   std: {base.std():.3f}\n")

    accepted = 1                                       # suppose the run you once accepted was seed 1
    print(f"Part B: naive test -- baseline = the run you accepted (seed {accepted}); FAIL if lower at all")
    for seed in (0, 2, 3, 4):
        new = base[seed]
        verdict = "FAIL" if new < base[accepted] else "pass"
        print(f"  unchanged pipeline, seed {seed}: {new:.3f} vs {base[accepted]:.3f} -> {verdict}")
    print("  Any FAIL above is a false alarm: the pipeline did not change.\n")

    print("Part C: better test -- 5-seed mean, FAIL if it drops by more than 3x the seed std")
    threshold = 3 * base.std()
    candidates = {"unchanged (model A, new seeds)": gold_accuracy("A", range(5, 10)),
                  "real regression (model B)": gold_accuracy("B", range(5))}
    for name, acc in candidates.items():
        drop = base.mean() - acc.mean()
        verdict = "FAIL" if drop > threshold else "pass"
        print(f"  {name:31s} mean {acc.mean():.3f}  drop {drop:+.3f}  (threshold {threshold:.3f}) -> {verdict}")

    ok = (base.mean() - candidates["real regression (model B)"].mean()) > threshold
    assert ok, "the regression test failed to catch model B"
    print("\nself-check passed: the test catches model B and passes the unchanged pipeline")

# EXERCISE 1: Part B -- how many false alarms out of 4? Now set accepted = 0 and run again: no
#   alarms at all. So whether the naive test cries wolf depends on which run you happened to
#   accept -- luck. Would you trust a test like that?
# EXERCISE 2: Change the threshold to 1x and 6x the std. At 1x, does the unchanged pipeline
#   ever fail? At 6x, does model B still get caught? Pick a multiplier and justify it.
# EXERCISE 3: The test checks overall accuracy. Model B could get worse on ONE variable while
#   the total barely moves. Add a per-variable check for your 3 most important variables
#   (lesson 7 tells you which matter downstream).
# EXERCISE 4 (build, week 5 of your plan): your NeuroPathology test/ folder is empty. Make
#   test/test_regression.py: freeze 10 reports' ground truth, load the latest 5-seed outputs,
#   and assert accuracy >= baseline - 3*std. Run it after every prompt or schema change.
