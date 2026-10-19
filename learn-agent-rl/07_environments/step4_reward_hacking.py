"""Lesson 7, step 4 of 4: reward hacking, and how to build graders that resist it.

Run:   python 07_environments/step4_reward_hacking.py     (about 15 seconds)

IN PLAIN WORDS
REWARD HACKING: an agent gets a high score on the thing you MEASURED without doing the thing
you WANTED. It isn't cheating on purpose -- an optimizer simply finds whatever raises the
number. Job postings ask for "graders that resist adversarial optimization" and "reward
hacking defenses"; this step shows two classic ways it happens.

PART A -- the proxy is not the goal.
  Two proxy rewards a campaign could easily compute:
    hits      fraction of the batch above the hit cutoff (the env's reward)
    model_mu  how good the SURROGATE THINKS the batch is (mean predicted score)
  Greedy maximizes both by construction -- and on 'hidden' it has the worst real grade.
  "model_mu" is the worst kind of proxy: the model grading its own choices. (Same trap as an
  LLM grading its own answers.)

PART B -- the grader trusts the agent.
  A sloppy grader reads the score the agent REPORTS about itself. A cheating policy reports
  1.0 and "passes". A strict grader recomputes the score from data the agent can't touch.
  For LLM agents this is "the agent says it fixed the bug" vs "run the tests yourself".
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ToyCampaign  # noqa: E402

PASS_MARK = 0.80


def strict_grade(env):
    """Recomputes recall from the ground truth. The agent can't influence this."""
    return float(env.true_top[env.labeled_idx].sum() / env.true_top.sum())


def run_with_proxies(env, beta, seed):
    """One campaign. Returns the two proxy rewards and the real grade."""
    env.reset(seed=seed)
    hits, model_mu, done = [], [], False
    while not done:
        picked = env._pick(beta)                      # the batch UCB will choose this round
        model_mu.append(float(env.mu[picked].mean()))  # what the MODEL thinks of it
        _, reward, done, _ = env.step(beta)
        hits.append(reward)
    return np.mean(hits), np.mean(model_mu), strict_grade(env)


def part_a():
    print("PART A: the proxy is not the goal ('hidden' library, 8 seeds)")
    print(f"  {'policy':10s} {'proxy: hits':>12s} {'proxy: model_mu':>16s} {'real grade':>11s}")
    env = ToyCampaign("hidden")
    for name, beta in (("greedy", 0.0), ("beta=0.25", 0.25), ("beta=0.5", 0.5)):
        rows = [run_with_proxies(env, beta, s) for s in range(8)]
        h, m, g = (np.mean([r[i] for r in rows]) for i in range(3))
        print(f"  {name:10s} {h:12.3f} {m:16.3f} {g:11.3f}")
    print("  -> greedy wins BOTH proxies and loses the real grade. Optimizing a proxy picks greedy.")


def part_b():
    print("\nPART B: a grader that trusts the agent")
    env = ToyCampaign("hidden")

    def honest_agent(seed):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(0.25)
        return {"claimed_recall": strict_grade(env)}        # reports the truth

    def cheating_agent(seed):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _ = env.step(0.0)                   # lazy: greedy
        return {"claimed_recall": 1.0}                      # ...and claims perfection

    def sloppy_grader(report):
        return report["claimed_recall"]                     # believes the agent

    print(f"  {'agent':8s} {'sloppy grader says':>19s} {'strict grader says':>19s}")
    for name, agent in (("honest", honest_agent), ("cheating", cheating_agent)):
        report = agent(seed=0)
        sloppy = sloppy_grader(report)
        strict = strict_grade(env)                          # recomputed from env, not from report
        print(f"  {name:8s} {sloppy:12.2f} {'PASS' if sloppy >= PASS_MARK else 'fail':>6s} "
              f"{strict:12.2f} {'PASS' if strict >= PASS_MARK else 'fail':>6s}")
    print("  -> the sloppy grader passes the cheater. Rule: a grader computes the score itself,")
    print("     from data the agent cannot write to.")


if __name__ == "__main__":
    part_a()
    part_b()

# EXERCISE 1: In Part A, why does greedy ALWAYS maximize model_mu, on any library? (Hint: greedy
#   = pick the highest mu. You don't need to run anything.)
# EXERCISE 2: Design a third proxy that does NOT favour greedy -- e.g. "how many of the batch were
#   in regions the model had never sampled". Add it to run_with_proxies(). Does it agree with the
#   real grade? Is it also hackable? (Almost every proxy is, by some policy.)
# EXERCISE 3: Write down 3 places in your own projects where a grader might trust the thing it
#   grades. Example from NeuroPathology: using the extractor's own confidence field as accuracy.
# EXERCISE 4: Connect to lessons 1 and 4: the bandit and LinUCB with --reward hit were "reward
#   hacking" all along. Rewrite your lesson-1 notes using the words proxy, grader and hacking.
#   That rewritten paragraph is an interview answer.
