"""Lesson 8, step 4 of 5: is the difference real? Confidence intervals.

Run:   python 08_agent_evals/step4_confidence_intervals.py      (a few seconds)

IN PLAIN WORDS
With 20 campaigns per agent, a difference of 0.02 in mean recall could easily be luck. A
CONFIDENCE INTERVAL (CI) says how much a number might move if you re-ran with new campaigns.

The BOOTSTRAP is the simplest way to get one:
  1. resample the campaigns WITH replacement (some twice, some not at all)
  2. recompute the metric
  3. repeat 2,000 times; the middle 95% of the results is the 95% CI

PAIRED comparison: both agents ran on the SAME (library, seed) pairs. So compare them
campaign by campaign (careful minus sloppy on the same seed) and bootstrap that difference.
Pairing removes the luck of the seed, which is usually the biggest source of noise. It is the
same idea as GRPO's groups (lesson 5) and your adaptive_acq paired-by-seed tables.

Rule of thumb: if the 95% CI of the difference includes 0, you can't claim a difference.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import attempts, group_campaigns, load_rounds  # noqa: E402


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% CI of the mean of `values`, by resampling with replacement."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def per_campaign(campaigns, agent, metric):
    """{(regime, seed): value} for one agent."""
    return {(k[1], k[2]): metric(rounds) for k, rounds in campaigns.items() if k[0] == agent}


def final_recall(rounds):
    return rounds[-1]["recall"]


def first_try_valid(rounds):
    tried = [r for r in rounds if attempts(r)]
    return np.mean([attempts(r)[0] == "accepted" for r in tried])


if __name__ == "__main__":
    campaigns = group_campaigns(load_rounds())
    for label, metric in (("final recall", final_recall), ("first-try valid rate", first_try_valid)):
        a = per_campaign(campaigns, "careful", metric)
        b = per_campaign(campaigns, "sloppy", metric)
        pairs = sorted(set(a) & set(b))
        diffs = [a[p] - b[p] for p in pairs]
        print(f"{label}:")
        for name, vals in (("careful", [a[p] for p in pairs]), ("sloppy", [b[p] for p in pairs])):
            lo, hi = bootstrap_ci(vals)
            print(f"  {name:8s} mean {np.mean(vals):.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        lo, hi = bootstrap_ci(diffs)
        verdict = "real difference" if lo > 0 or hi < 0 else "can't tell apart (CI includes 0)"
        print(f"  careful - sloppy, paired over {len(pairs)} campaigns: {np.mean(diffs):+.3f}"
              f"   95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}\n")

# EXERCISE 1: Which metric shows a real difference, and which doesn't? Put it in one sentence
#   you could say in an interview: "The sloppy agent's process is measurably worse, but ..."
# EXERCISE 2: Compute the UNPAIRED CI of the difference: bootstrap careful and sloppy separately
#   and subtract. Is it wider or narrower than the paired one? Why?
# EXERCISE 3: Halve the data (use only seeds 10000-10004 by filtering `pairs`). What happens to
#   the CI widths? Roughly how many campaigns would you need to detect a 0.01 recall difference?
