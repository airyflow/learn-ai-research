"""Multi-armed bandits: epsilon-greedy, UCB1 and Thompson sampling.

IN PLAIN WORDS: 5 slot machines (one per beta), unknown payoffs. Pull the best-looking one too
often and you may miss a better one; try them all too often and you waste pulls. Three classic
ways to balance that, and (Part C) why a bandit picks greedy on a problem where greedy is wrong.
Core exercises: 1 and 3. Walkthrough: LESSONS.md, Lesson 1.

Run:   python 01_bandits/bandits.py            (about 2 min on CPU)

The setting: you have K "arms" (here: K values of UCB's beta). Each pull of
an arm returns a random reward (here: 1 if the acquired molecule is a hit,
else 0). You don't know the arms' hit rates. Every pull spent on a bad arm is
lost; every arm you never try might have been the best one. That tension is
EXPLORATION vs EXPLOITATION, and it is the same tension UCB acquisition
resolves one level down (molecules instead of betas).

  Part A: the three classic algorithms on a Bernoulli bandit; regret.
  Part B: the same, but rewards arrive in batches (as in a screening round).
  Part C: a bandit over beta on the toy AL campaign -- and why it learns the
          wrong thing (the hook into module 03).

Maths you need (Lattimore & Szepesvari, "Bandit Algorithms", ch. 1, 7, 36):
  regret(T) = T * mu_best - sum_t mu_{a_t}          (expected reward lost to not knowing)
  UCB1:      a_t = argmax_a  mean_a + c * sqrt(log t / N_a)
  Thompson:  theta_a ~ Beta(1 + hits_a, 1 + misses_a);  a_t = argmax_a theta_a
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `common` imports
from common.toy_campaign import ARMS, ToyCampaign  # noqa: E402


# ── the three algorithms ──────────────────────────────────────────────────────
# All share one interface: select() -> arm index, update(arm, reward).
# This is the same interface as adaptive_acq/controllers.py's plan()/update().

class EpsilonGreedy:
    def __init__(self, k, eps=0.1, seed=0):
        self.eps, self.rng = eps, np.random.default_rng(seed)
        self.n, self.s = np.zeros(k), np.zeros(k)        # pulls and reward sums per arm

    def select(self):
        if self.rng.random() < self.eps or self.n.min() == 0:
            return int(self.rng.integers(len(self.n)))
        return int(np.argmax(self.s / self.n))

    def update(self, a, r):
        self.n[a] += 1
        self.s[a] += r


class UCB1:
    def __init__(self, k, c=np.sqrt(2), seed=0):
        self.c = c
        self.n, self.s, self.t = np.zeros(k), np.zeros(k), 0

    def select(self):
        self.t += 1
        if self.n.min() == 0:                             # pull every arm once first
            return int(np.argmin(self.n))
        bonus = self.c * np.sqrt(np.log(self.t) / self.n)  # "optimism in the face of uncertainty"
        return int(np.argmax(self.s / self.n + bonus))

    def update(self, a, r):
        self.n[a] += 1
        self.s[a] += r


class Thompson:
    """Beta-Bernoulli Thompson sampling: keep a posterior over each arm's hit
    rate, draw one sample per arm, play the arm whose sample is largest. Arms
    are tried in proportion to the probability that they are the best."""

    def __init__(self, k, seed=0):
        self.a, self.b = np.ones(k), np.ones(k)           # Beta(1,1) = uniform prior
        self.rng = np.random.default_rng(seed)

    def select(self):
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def update(self, a, r):
        self.a[a] += r
        self.b[a] += 1 - r


ALGOS = {"eps-greedy": EpsilonGreedy, "UCB1 (c=1.41)": UCB1,
         "UCB1 (c=0.5)": lambda k, seed: UCB1(k, c=0.5), "Thompson": Thompson}


# ── Part A: Bernoulli bandit ─────────────────────────────────────────────────

def run_bandit(algo_cls, means, T, seed, batch=1):
    """Returns cumulative regret over T pulls. With batch > 1 the algorithm
    must choose `batch` arms before it sees any of their rewards."""
    rng = np.random.default_rng(seed)
    algo = algo_cls(len(means), seed=seed)
    regret, t = np.zeros(T), 0
    while t < T:
        arms = [algo.select() for _ in range(min(batch, T - t))]
        rewards = rng.random(len(arms)) < means[arms]
        for a, r in zip(arms, rewards):
            algo.update(a, float(r))
            regret[t] = means.max() - means[a]
            t += 1
    return np.cumsum(regret)


def part_a(batch=1, T=30000, runs=20):
    # Hit rates for 5 beta arms. Close together, like real acquisition functions.
    means = np.array([0.30, 0.34, 0.36, 0.31, 0.22])
    print(f"\nPart {'A' if batch == 1 else 'B'}: arm hit rates {means}, best = arm {means.argmax()}, "
          f"T={T}, batch={batch}, {runs} runs")
    checkpoints = (100, 1000, 3000, 10000, T)
    print(f"{'algorithm':14s}" + "".join(f"  regret@{c:<6d}" for c in checkpoints))
    for name, cls in ALGOS.items():
        R = np.mean([run_bandit(cls, means, T, s, batch) for s in range(runs)], axis=0)
        print(f"{name:14s}" + "".join(f"  {R[c - 1]:12.1f}" for c in checkpoints), flush=True)
    print("(random play would have regret", round(T * (means.max() - means.mean()), 1), "at T)")


# ── Part C: a bandit over beta on the toy AL campaign ─────────────────────────

def part_c(regime="hidden", n_campaigns=40):
    """Each campaign round is one pull: the bandit picks beta for the whole
    batch; reward = that round's hit rate. The bandit keeps learning ACROSS
    campaigns (one campaign has only 6 rounds -- too few to learn anything)."""
    env = ToyCampaign(regime)
    algo = Thompson(len(ARMS), seed=0)
    recalls, picks = [], np.zeros(len(ARMS), dtype=int)
    for ep in range(n_campaigns):
        env.reset(seed=ep)
        done = False
        while not done:
            a = algo.select()
            _, r, done, info = env.step(ARMS[a])
            # Thompson expects a 0/1 reward; a hit RATE is fine as a fractional
            # Bernoulli observation (the posterior update stays valid in expectation).
            algo.update(a, r)
            picks[a] += 1
        recalls.append(info["recall"])
    print(f"\nPart C: Thompson over beta={ARMS} on the '{regime}' toy campaign, {n_campaigns} campaigns")
    print("  arm pulls:", dict(zip(ARMS, picks.tolist())))
    print(f"  posterior mean hit rate per arm: {np.round(algo.a / (algo.a + algo.b), 3)}")
    print(f"  final recall, last 10 campaigns: {np.mean(recalls[-10:]):.3f}")
    print("  compare (from common/toy_campaign.py): fixed beta=0 -> ~0.64, fixed beta=0.25 -> ~0.93")


if __name__ == "__main__":
    part_a(batch=1)
    part_a(batch=50)
    part_c("hidden")

# EXERCISE 1: Part A. Expect (roughly): at T=3000 eps-greedy looks BEST; by T=30000 it is worse
#   than Thompson, and its regret keeps growing linearly while Thompson's flattens. Why linear?
#   (Hint: what fraction of pulls is random even after it knows the best arm?) Lesson: compare
#   algorithms at the horizon you actually have -- a campaign has 6 rounds, not 30000.
#   Try eps=0.01 and a decaying eps_t = min(1, 5/t^0.5).
# EXERCISE 1b: UCB1 with the textbook c=sqrt(2) is the worst here; c=0.5 is the best. The
#   theory constant makes the regret BOUND hold for any rewards in [0,1]; it is loose for these
#   arms. Same story as beta in UCB acquisition: the "right" exploration weight is empirical.
# EXERCISE 2: Part B (batch=50) is what screening looks like: 50 decisions before any feedback.
#   Which algorithm suffers most and why? Thompson is naturally "randomized" -- a batch of 50
#   Thompson draws already spreads over arms. This is the idea behind adaptive_acq's
#   BatchSplitThompson. Read controllers.py:BatchSplitThompson.plan() and explain `min_frac`.
# EXERCISE 3: Part C. The bandit learns that beta=0 (greedy) has the highest hit rate, and it is
#   RIGHT about that -- yet greedy has the worst final recall on 'hidden'. Write down, in one
#   sentence, what the bandit's reward fails to measure. Module 03 formalizes this.
# EXERCISE 4: Re-run Part C with regime="smooth". Does the bandit now do the right thing? Why is
#   "the bandit is fine on ENHITS" (adaptive_acq README, Stage 1) consistent with this?
# EXERCISE 5: Add a `discount` (gamma < 1) to Thompson: before each update multiply a-1 and b-1
#   by gamma. When does forgetting help? (Hint: is the best beta the same in round 1 and round 6?)
