"""Lesson 6: can an LLM grade the extractions? Validate the judge before you trust it.

Run:   python 06_llm_judge/judge_agreement.py     (about 10 seconds)

IN PLAIN WORDS
Checking every extraction by hand is slow. An LLM-AS-JUDGE reads the report and the extracted
value and says "supported" or "not supported". But the judge is itself an LLM -- it makes
mistakes, and they may be BIASED (e.g. more lenient on long reports). So before using it you
measure it against human labels, exactly like any other model.

Here the "human" labels are the true correctness (from the fake ground truth) and the judge is
simulated with two realistic flaws: it is too lenient with wrong answers, and more lenient
still on long reports.

Part A: how often the judge agrees with humans -- raw agreement and Cohen's kappa
        (agreement corrected for luck). With a report-level bootstrap CI (lesson 4).
Part B: the judge as an ERROR DETECTOR: of the real errors, how many does it catch (recall)?
        Of what it flags, how much is really wrong (precision)?
Part C: BIAS -- is the judge worse on long reports?
Part D: how many human labels do you need before the kappa CI is narrow enough?
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import extractions, ground_truth, with_truth  # noqa: E402


def simulated_judge(df, gt, rng):
    """Returns 1 = 'supported', 0 = 'not supported'. Lenient, and more so on long reports."""
    long = df["report_id"].map(gt.set_index("report_id")["length_words"] > gt["length_words"].median())
    p_supported = np.where(df["is_correct"], 0.93, 0.30 + 0.25 * long)
    p_supported = np.where(df["outcome"] == "hallucinated", 0.20 + 0.25 * long, p_supported)
    return (rng.random(len(df)) < p_supported).astype(int)


def kappa_ci(df, rng, B=1000):
    ids = df["report_id"].unique()
    groups = {r: g for r, g in df.groupby("report_id")}
    ks = []
    for _ in range(B):
        sample = [groups[r] for r in rng.choice(ids, len(ids))]
        h = np.concatenate([g["human"].values for g in sample])
        j = np.concatenate([g["judge"].values for g in sample])
        ks.append(cohen_kappa_score(h, j))
    return np.percentile(ks, [2.5, 97.5])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=("A",), seeds=[0]))
    df["human"] = df["is_correct"].astype(int)          # 1 = the extraction is right
    df["judge"] = simulated_judge(df, gt, rng)

    print("Part A: judge vs human labels (1600 extractions)")
    agree = (df["human"] == df["judge"]).mean()
    k = cohen_kappa_score(df["human"], df["judge"])
    lo, hi = kappa_ci(df, rng)
    print(f"  raw agreement: {agree:.3f}")
    print(f"  Cohen's kappa: {k:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print("  Raw agreement looks great because most extractions are correct -- a judge that always")
    print(f"  said 'supported' would agree {df['human'].mean():.3f} of the time. Kappa corrects for that.\n")

    print("Part B: the judge as an error detector (positive class = 'this extraction is WRONG')")
    err_h, err_j = 1 - df["human"], 1 - df["judge"]
    print(f"  recall:    {recall_score(err_h, err_j):.3f}   (share of real errors the judge catches)")
    print(f"  precision: {precision_score(err_h, err_j):.3f}   (share of judge flags that are real errors)\n")

    print("Part C: is the judge worse on long reports?")
    long = df["report_id"].map(gt.set_index("report_id")["length_words"] > gt["length_words"].median())
    for name, mask in (("short reports", ~long), ("long reports", long)):
        s = df[mask]
        print(f"  {name:13s}: catches {recall_score(1 - s['human'], 1 - s['judge']):.3f} of errors")
    print("  A biased judge makes long reports look better than they are.\n")

    print("Part D: kappa CI width vs number of human-labeled reports")
    ids = df["report_id"].unique()
    for n in (20, 40, 80, 160):
        sub = df[df["report_id"].isin(ids[:n])]
        lo, hi = kappa_ci(sub, rng, B=500)
        print(f"  {n:4d} reports labeled: kappa CI width {hi - lo:.3f}")

# EXERCISE 1: Part A -- explain to a non-statistician, in two sentences, why kappa is lower
#   than raw agreement here.
# EXERCISE 2: Part B -- the judge catches only part of the errors. If you used it to REPORT your
#   extractor's accuracy, would the number be too high or too low? (Too high: missed errors
#   count as correct.) This is "the grader trusts the agent" from learn-agent-rl lesson 7.
# EXERCISE 3: Part C shows a length bias. Your real reports vary a lot in length. Which other
#   report properties might bias a judge? (Form version, OCR quality, section order...)
# EXERCISE 4 (build, week 6 of your plan): your NeuroPathology repo has no LLM judge yet. Design
#   one: input = (report excerpt, variable, extracted value, quoted evidence); output = JSON
#   {"supported": true/false, "reason": ...}. Label 40 reports by hand (or with a pathologist),
#   then compute Parts A-C on real data. The evidence quotes are already in field_annotations.
