"""Lesson 2: run it again -- consistency across seeds.

Run:   python 02_consistency/consistency.py     (3 seconds)

IN PLAIN WORDS
An LLM can give different answers to the same report on different runs (seeds). That matters:
  - one run's accuracy is partly luck
  - if 5 runs disagree, at least some of them are wrong -- a free warning sign, no labels needed

Part A: how often do all 5 seeds agree? Which variables wobble most?
Part B: one run vs MAJORITY VOTE of 5 runs -- does voting help?
Part C: does agreement predict correctness? (If yes, agreement is a useful confidence signal.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import extractions, ground_truth, outcome, with_truth  # noqa: E402


def vote(values):
    """Most common answer across seeds (NaN counts as the answer "not documented")."""
    counts = values.astype(str).value_counts()
    top = counts.index[0]
    return values[values.astype(str) == top].iloc[0], counts.iloc[0] / len(values)


if __name__ == "__main__":
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=("A",), seeds=range(5)))

    rows = []
    for (rid, field), g in df.groupby(["report_id", "field"]):
        value, agreement = vote(g["value"])
        rows.append({"report_id": rid, "field": field, "true": g["true"].iloc[0],
                     "vote": value, "agreement": agreement})
    import pandas as pd
    votes = pd.DataFrame(rows)
    votes["vote_correct"] = [outcome(t, v) in ("correct", "correct_missing")
                             for t, v in zip(votes["true"], votes["vote"])]

    print("Part A: share of items where all 5 seeds agree, per variable")
    print(votes.groupby("field")["agreement"].apply(lambda a: (a == 1).mean()).sort_values().round(2).to_string())

    print("\nPart B: one run vs majority vote")
    one_run = df.groupby("seed")["is_correct"].mean()
    print(f"  single run accuracy, seeds 0-4: {one_run.round(3).tolist()}")
    print(f"  majority vote of 5:             {votes['vote_correct'].mean():.3f}")

    print("\nPart C: accuracy of the vote, by how many seeds agreed")
    table = votes.groupby("agreement")["vote_correct"].agg(["mean", "size"])
    table.index = [f"{int(round(a * 5))}/5 agree" for a in table.index]
    print(table.round(3).to_string())
    print("  -> when the seeds disagree, the answer is much more likely to be wrong.")

# EXERCISE 1: Part B -- is the vote better than the BEST single seed? Better than the average
#   seed? Why can voting never fix an item that is wrong on all 5 seeds?
# EXERCISE 2: Using Part C, write a rule: "send to a human reviewer if fewer than k/5 agree".
#   Pick k. How many items would a reviewer see, and what share of all errors would they catch?
# EXERCISE 3 (real code): analysis/analyze_seeds.py line 68 (consistency) and
#   analysis/compute_confidence_metrics.py lines 111 and 135 do Parts A-B on your real runs.
#   Find where ties between seeds are handled. What does this script do with a 2-2-1 tie?
