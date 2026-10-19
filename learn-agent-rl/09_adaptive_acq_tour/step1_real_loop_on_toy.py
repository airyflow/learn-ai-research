"""Lesson 9, step 1: run YOUR real adaptive_acq code on the course's toy library.

Run:   python 09_adaptive_acq_tour/step1_real_loop_on_toy.py      (about 10 seconds)

IN PLAIN WORDS
Lessons 0-8 used a small copy of your project. This step proves the copy is faithful: it plugs
the toy library and surrogate from lesson 0 into the REAL `adaptive_acq.Campaign` loop and the
REAL controllers from al-molecular (FixedBeta, Schedule, BatchSplitThompson). Same library, same
seeds, your production code.

If the numbers look like lesson 0's table (on hidden, greedy far behind a little exploring),
then everything you learned in the course applies to the real code line for line.

What you need to know to read it:
  Campaign needs two functions from you:
    fit_predict(labeled_idx, y_hb, cand_idx) -> (mu, sigma) for the candidates
    score_fn(idx)                            -> docking-style scores (LOWER = better)
  The toy library's y is HIGHER = better, so score_fn returns -y. That sign flip is the same
  one run_experiment.py does for real docking scores.
"""
import sys
from pathlib import Path

import numpy as np

COURSE = Path(__file__).resolve().parent.parent
PROJECT = Path("/N/slate/mengjing/repos/al-molecular")
sys.path.insert(0, str(COURSE))
sys.path.insert(0, str(PROJECT))

from adaptive_acq.campaign import Campaign  # noqa: E402   <- your real code
from adaptive_acq.controllers import BatchSplitThompson, FixedBeta, Schedule  # noqa: E402
from common.toy_campaign import RFFSurrogate, make_library, top_k  # noqa: E402   <- the course's toy

N, INIT, BATCH, ROUNDS, TOPK = 10_000, 100, 50, 6, 100          # same sizes as lesson 0


def make_campaign(regime, seed, lib):
    X, y, surr = lib
    true_top = np.zeros(N, dtype=bool)
    true_top[top_k(y, TOPK)] = True
    init = np.random.default_rng(seed).choice(N, INIT, replace=False)

    def fit_predict(labeled_idx, y_hb, cand_idx):
        mu, sigma = surr.fit_predict(labeled_idx, y_hb)          # over the whole library...
        return mu[cand_idx], sigma[cand_idx]                    # ...Campaign wants candidates only

    def score_fn(idx):
        return -y[idx]                                          # adaptive_acq: lower = better

    return Campaign(n_pool=N, fit_predict=fit_predict, score_fn=score_fn, init_idx=init,
                    batch_size=BATCH, n_rounds=ROUNDS, k_obs=TOPK, true_top_mask=true_top)


CONTROLLERS = {
    "FixedBeta(0)  greedy": lambda s: FixedBeta(0.0),
    "FixedBeta(0.25)": lambda s: FixedBeta(0.25),
    "Schedule explore->exploit": lambda s: Schedule("explore_then_exploit", beta_hi=1.0, beta_lo=0.0),
    "BatchSplitThompson": lambda s: BatchSplitThompson(arms=(0.0, 0.25, 0.5, 1.0, 2.0), seed=s),
}

if __name__ == "__main__":
    seeds = range(8)
    print(f"{'controller (from al-molecular)':28s} {'smooth':>8s} {'hidden':>8s}")
    libs = {r: (lambda X_y: (*X_y, RFFSurrogate(X_y[0], seed=0)))(make_library(r, N, seed=0))
            for r in ("smooth", "hidden")}
    for name, make in CONTROLLERS.items():
        cells = []
        for regime in ("smooth", "hidden"):
            recalls = []
            for s in seeds:
                history = make_campaign(regime, s, libs[regime]).run(make(s))
                recalls.append(history[-1]["topk_recall"])
            cells.append(np.mean(recalls))
        print(f"{name:28s} {cells[0]:8.3f} {cells[1]:8.3f}")
    print("\nCompare with lesson 0: smooth greedy ~0.99, hidden greedy ~0.64, hidden beta=0.25 ~0.93.")

# EXERCISE 1: Compare with lesson 0. On 'hidden' the order matches (greedy ~0.66 is far behind
#   beta=0.25 ~0.91). On 'smooth', beta=0.25 slightly BEATS greedy here (0.998 vs 0.984), while
#   lesson 0 had greedy ahead. Why? (This run uses ONE library per regime; lesson 0 averages
#   over 4. Small gaps on a single library are noise -- lesson 8 step 4 tells you how to check.)
# EXERCISE 2: BatchSplitThompson lands between greedy and beta=0.25 on 'hidden' (~0.88). Its
#   reward is this round's hits, so it leans greedy (lesson 1 Part C) -- but read min_frac in
#   controllers.py: every arm keeps at least 5% of each batch. How does that floor rescue it?
# EXERCISE 3: Add a print of history[0]["context"] for one campaign. Those 10 numbers are the
#   same FEATURE_NAMES as common/toy_campaign.py -- find them in adaptive_acq/state.py.
