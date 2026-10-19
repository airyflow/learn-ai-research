"""Lesson 8, step 5 of 5: does the agent's stated reason match what it did?

Run:   python 08_agent_evals/step5_reason_check.py      (instant)

IN PLAIN WORDS
An agent explains each decision. If the reason says "exploit" but the action explores, either
the reasoning is wrong or the explanation is -- and you can't trust the logs for auditing.
Checking text like this needs a GRADER FOR TEXT. Three options, from cheapest to richest:

  1. keyword rules     fast, free, transparent, but brittle      <- this step
  2. LLM-as-judge      flexible, but it can be wrong or biased too
  3. human labels      the ground truth, but slow and expensive

Whatever grader you use, check it against human labels on a sample first. That is your
NeuroPathology plan for week 6 (judge vs pathologist agreement), applied to agents.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import load_rounds  # noqa: E402

EXPLOIT_WORDS = ("exploit", "greedy", "no exploration")
EXPLORE_WORDS = ("explore", "uncertain", "map the space")


def reason_mismatch(decision):
    """Rule-based check. Returns a short label if reason and action disagree, else None."""
    reason, beta = decision["reason"].lower(), decision["beta"]
    says_exploit = any(w in reason for w in EXPLOIT_WORDS)
    says_explore = any(w in reason for w in EXPLORE_WORDS)
    if says_exploit and not says_explore and beta > 0:
        return "says exploit, explores"
    if says_explore and not says_exploit and beta == 0:
        return "says explore, exploits"
    return None


if __name__ == "__main__":
    rows = load_rounds()
    for agent in sorted({r["agent"] for r in rows}):
        mine = [r for r in rows if r["agent"] == agent and "FALLBACK" not in r["decision"]["reason"]]
        flags = [(r, reason_mismatch(r["decision"])) for r in mine]
        flagged = [(r, f) for r, f in flags if f]
        print(f"{agent:8s}: {len(flagged)} of {len(mine)} decisions flagged ({len(flagged) / len(mine):.0%})")
        for r, f in flagged[:3]:
            d = r["decision"]
            print(f"   e.g. round {r['round']}, beta={d['beta']}: \"{d['reason']}\"  -> {f}")
    print("\nNext: is the RULE right? Hand-label some decisions and compare (exercise 1).")

# EXERCISE 1: Hand-label 20 decisions (10 per agent) as match / mismatch, reading the reason and
#   the beta yourself. Compare with reason_mismatch(): how many did the rule get right? Compute
#   its precision (of the flagged ones, how many were really mismatches) and recall (of the real
#   mismatches, how many were flagged). This is "validate your grader" in miniature.
# EXERCISE 2: The first flagged example is a FALSE POSITIVE: a repair message ("Previous
#   action rejected (... strategy 'greedy' requires beta ...); corrected.") contains the word
#   'greedy' only because it quotes an error. The agent didn't say "exploit" at all. Improve
#   the rule so it ignores quoted errors, then re-check your hand labels. Did fixing one case
#   break another? (This is what grader iteration looks like.)
# EXERCISE 3 (GPU, free): write an LLM-as-judge version with the local vLLM model: send the
#   reason and the beta, ask "match or mismatch?" as JSON. Measure its agreement with your 20
#   hand labels. Is it better than the keyword rule?
