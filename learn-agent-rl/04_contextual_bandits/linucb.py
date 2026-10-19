"""Contextual bandits: LinUCB chooses beta from the campaign's context vector.

IN PLAIN WORDS: a bandit that looks at the situation (the 10 features) before choosing beta,
using one linear regression per beta. Trained on this round's hits it copies greedy; trained on
"what final recall would this lead to?" it starts exploring -- but needs more data to win.
Core exercises: 1 and 2. Walkthrough: LESSONS.md, Lesson 4.

Run:   python 04_contextual_bandits/linucb.py                   (~1 min; reward = hit rate)
       python 04_contextual_bandits/linucb.py --reward rollout  (~5 min; reward = final recall
                                                                  of finishing the campaign)

Module 01's bandit ignored the state: one hit-rate estimate per arm. A CONTEXTUAL bandit
learns a reward model per arm as a function of the context x_t (env.observe()):

    E[r | x, a] = theta_a . x                          (one linear model per arm)
    a_t = argmax_a  theta_a . x_t + alpha * sqrt(x_t^T A_a^-1 x_t)
                    \\__ estimate __/   \\__ uncertainty of that estimate (cf. UCB acquisition) __/
    A_a = I + sum x x^T,  b_a = sum r x,  theta_a = A_a^-1 b_a     (ridge regression per arm)

Li et al. 2010 (news recommendation) introduced it; adaptive_acq/controllers.py:LinUCB is the
same algorithm. As there, we train it OFFLINE in the simulator with full-information
(counterfactual) feedback -- for each round, the reward of EVERY arm (env.peek / rollouts) --
because one real campaign has only 6 pulls, far too few to fit 5 x 10 parameters.

Task: regime="mixed" -- each campaign is secretly 'smooth' or 'hidden'. The best fixed beta
differs between them (module 02), so a controller that could tell them apart from x_t could
beat every fixed beta. Can LinUCB?

Two rewards, and the difference between them is the lesson:
  hit      the observable per-round hit rate. Myopic (gamma = 0, module 03).
  rollout  play arm a this round, then finish the campaign with a default beta; reward =
           final recall. This is a Monte-Carlo estimate of Q^pi(x, a) (pi = default policy),
           so the bandit becomes one step of POLICY ITERATION. Sim-only: needs the truth.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ARMS, FEATURE_DIM, FEATURE_NAMES, ToyCampaign  # noqa: E402


class LinUCB:
    def __init__(self, n_arms, d=FEATURE_DIM, alpha=0.5, ridge=1.0):
        self.alpha = alpha
        self.A = np.stack([ridge * np.eye(d) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, d))

    def theta(self):
        return np.stack([np.linalg.solve(A, b) for A, b in zip(self.A, self.b)])

    def scores(self, x, explore=True):
        th = self.theta()
        mean = th @ x
        if not explore:
            return mean
        bonus = np.array([np.sqrt(x @ np.linalg.solve(A, x)) for A in self.A])
        return mean + self.alpha * bonus

    def select(self, x, explore=True):
        return int(np.argmax(self.scores(x, explore)))

    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


def arm_rewards(env, kind, default_beta=0.25):
    """Reward of every arm at the current state, without advancing the campaign."""
    if kind == "hit":
        return np.array([env.peek(b) for b in ARMS])
    out = []
    snap = env.snapshot()
    for b in ARMS:
        env.restore(snap)
        _, _, done, info = env.step(b)
        while not done:
            _, _, done, info = env.step(default_beta)
        out.append(info["recall"])
    env.restore(snap)
    return np.array(out)


def train(env, kind, n_episodes, behaviour_beta=0.25):
    """Full-information training: every arm gets its reward each round. The campaign
    itself moves forward with a fixed behaviour beta (the states we learn about are
    the states that policy visits -- a real limitation; see EXERCISE 3)."""
    bandit = LinUCB(len(ARMS))
    for ep in range(n_episodes):
        x, done = env.reset(seed=ep), False
        while not done:
            for a, r in enumerate(arm_rewards(env, kind)):
                bandit.update(a, x, r)
            x, _, done, _ = env.step(behaviour_beta)
    return bandit


def evaluate(env, policy, seeds):
    """policy(x) -> beta. Returns {regime: [final recalls]} on held-out seeds."""
    out = {"smooth": [], "hidden": []}
    for s in seeds:
        x, done = env.reset(seed=s), False
        while not done:
            x, _, done, info = env.step(policy(x))
        out[info["regime"]].append(info["recall"])
    return out


def fmt(res):
    allr = res["smooth"] + res["hidden"]
    return (f"smooth {np.mean(res['smooth']):.3f} (n={len(res['smooth'])})  "
            f"hidden {np.mean(res['hidden']):.3f} (n={len(res['hidden'])})  all {np.mean(allr):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=["hit", "rollout"], default="hit")
    p.add_argument("--train-episodes", type=int, default=None)
    p.add_argument("--test-episodes", type=int, default=30)
    args = p.parse_args()
    n_train = args.train_episodes or (60 if args.reward == "hit" else 30)

    env = ToyCampaign("mixed")
    test = range(10_000, 10_000 + args.test_episodes)      # held-out seeds: never trained on

    print("Fixed-beta baselines on held-out campaigns (final recall):")
    for b in ARMS:
        print(f"  beta={b:4.2f}   {fmt(evaluate(env, lambda x, b=b: b, test))}")

    t0 = time.perf_counter()
    bandit = train(env, args.reward, n_train)
    print(f"\nLinUCB trained on {n_train} campaigns, reward='{args.reward}' "
          f"({time.perf_counter() - t0:.0f}s)")
    chosen = []

    def policy(x):
        a = bandit.select(x, explore=False)                 # deployment: no exploration bonus
        chosen.append(ARMS[a])
        return ARMS[a]

    print(f"  LinUCB      {fmt(evaluate(env, policy, test))}")
    print(f"  betas chosen on test campaigns: {dict(zip(*np.unique(chosen, return_counts=True)))}")
    th = bandit.theta()
    print("\n  theta per arm (rows) x feature (cols): which features move which arm's predicted reward")
    print("           " + " ".join(f"{n[:8]:>8s}" for n in FEATURE_NAMES))
    for b, row in zip(ARMS, th):
        print(f"  b={b:4.2f}   " + " ".join(f"{v:8.3f}" for v in row))

# EXERCISE 1: With --reward hit, expect LinUCB to choose beta=0 almost always, i.e. to match
#   greedy, and to lose on 'hidden'. Use the theta table to explain why: for every context,
#   which arm has the highest predicted HIT RATE? (Module 01 Part C, now with features.)
# EXERCISE 2: Run --reward rollout. Does LinUCB now explore? Does it pick different betas on
#   'smooth' vs 'hidden' campaigns? Add a print of info["regime"] next to each choice to check.
#   If it can't separate them, which feature would you ADD to the context so that it could?
# EXERCISE 3: train() moves the campaign forward with behaviour_beta=0.25, so LinUCB only sees
#   states that policy reaches. Train with behaviour_beta=0 instead. What happens on 'hidden'?
#   (This is "distribution shift" -- the reason train_linucb.py and GRPO collect data with the
#   policy being learned, not a fixed one.)
# EXERCISE 4: Read adaptive_acq/train_linucb.py and controllers.py:LinUCB. Map: env.peek ->
#   Campaign(counterfactual=True); arm_rewards(kind="hit") -> the `cf` dict in campaign.step.
#   Which of this module's two rewards does adaptive_acq use? What does that predict?
