"""Lesson 7, step 3 of 4: rewards vs graders.

Run:   python 07_environments/step3_grader.py      (about 15 seconds)

IN PLAIN WORDS
Two different things score an agent, and mixing them up is the most common mistake:

  REWARD   given at every step, INSIDE the environment, and seen by the agent.
           Here: the fraction of this batch that were hits. It has to be something a real
           campaign could compute, so it's only a hint of how things are going.

  GRADER   run once at the END, OUTSIDE the agent's reach, using the ground truth.
           Here: final recall of the true top 1%. It is what you actually care about.

A grader that checks the answer against ground truth automatically -- no human, no LLM
opinion -- is called a VERIFIER, and the rewards it produces are VERIFIABLE REWARDS. That's
the "VR" in RLVR, a 2026 hiring keyword. Your simulator is a verifier: it knows the true top 1%.

This script runs 3 policies on both libraries and prints, for each: the total reward (what the
agent was told) and the grade (what really happened). Look for the rows where they disagree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ToyCampaign  # noqa: E402

PASS_MARK = 0.80     # a campaign "passes" if it finds at least 80% of the true top 1%


def grade(env):
    """The grader. Called AFTER the episode. Uses the ground truth (env.true_top),
    which the agent never sees. Returns (score, passed)."""
    found = env.true_top[env.labeled_idx].sum()      # true top-1% molecules we labeled
    score = float(found / env.true_top.sum())
    return score, score >= PASS_MARK


def run(env, beta, seed):
    env.reset(seed=seed)
    total_reward, done = 0.0, False
    while not done:
        _, reward, done, _ = env.step(beta)
        total_reward += reward
    return total_reward, grade(env)


if __name__ == "__main__":
    seeds = range(8)
    print(f"{'library':8s} {'policy':10s} {'total reward':>13s} {'grade':>6s} {'pass rate':>10s}")
    for regime in ("smooth", "hidden"):
        env = ToyCampaign(regime)
        for name, beta in (("greedy", 0.0), ("beta=0.25", 0.25), ("beta=1", 1.0)):
            results = [run(env, beta, s) for s in seeds]
            reward = np.mean([r for r, _ in results])
            score = np.mean([g[0] for _, g in results])
            passed = np.mean([g[1] for _, g in results])
            print(f"{regime:8s} {name:10s} {reward:13.2f} {score:6.2f} {passed:10.0%}")

# EXERCISE 1: On 'hidden', which policy has the most reward, and which has the best grade?
#   Write the two answers side by side. That gap is the subject of step 4.
# EXERCISE 2: Change PASS_MARK to 0.95. Which pass rates change? A pass/fail threshold turns a
#   score into "task success" -- the main number in most agent evals (lesson 8).
# EXERCISE 3: grade() reads env.true_top. Could a real screening campaign run this grader?
#   (No: nobody knows the true top 1% in advance.) So where can verifiable rewards exist?
#   (Simulators, retrospective benchmarks, unit tests for code, proof checkers for maths.)
