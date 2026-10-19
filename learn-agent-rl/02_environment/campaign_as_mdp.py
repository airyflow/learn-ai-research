"""An AL campaign IS a sequential decision problem. This module makes that concrete.

IN PLAIN WORDS: names the parts of the loop (state, action, reward, episode, policy) and shows
the most important table in the course: on 'hidden', the policy with the most hits per round
has the WORST final recall. Core exercises: 1 and 3. Walkthrough: LESSONS.md, Lesson 2.

Run:   python 02_environment/campaign_as_mdp.py           (about 1 min)
       python 02_environment/campaign_as_mdp.py --trace   (print one episode step by step)

Vocabulary (Sutton & Barto, ch. 3), mapped onto common/toy_campaign.py:

  state s_t        everything that determines what happens next: the labeled set, the
                   surrogate's mu and sigma over the pool, the round number. HUGE.
  observation o_t  what the controller actually sees: env.observe() -> 10 numbers.
                   o_t is a lossy summary of s_t, so strictly this is a POMDP. Every
                   practical controller (adaptive_acq included) just treats o_t as the state.
  action a_t       beta for this round (from ARMS). The controller never picks molecules;
                   UCB does, given beta. "Constrained action space" is a design choice.
  transition       label the batch, retrain the surrogate -> s_{t+1}. Stochastic only
                   through the initial sample here; real docking is noisier.
  reward r_t       observable: fraction of the batch above the frozen hit cutoff.
  return G         sum (or mean) of rewards over the episode -- OR a trajectory-level score
                   such as final recall, which a simulator can compute and a real screen can't.
  policy pi(a|o)   the controller. Fixed beta, a schedule, a bandit, LinUCB, an MLP, an LLM.
  episode          one campaign: reset() -> n_rounds steps -> done.

The point of this script: the best ACTION depends on the round, and the best POLICY depends on
the task ("regime"). If neither were true, a fixed beta would be optimal and you wouldn't need
any of modules 03-06.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ARMS, FEATURE_NAMES, ToyCampaign  # noqa: E402


def rollout(env, policy, seed):
    """policy: function (obs, round) -> beta. Returns per-round rewards, recalls, betas."""
    obs = env.reset(seed)
    out = {"reward": [], "recall": [], "beta": []}
    done, t = False, 0
    while not done:
        beta = policy(obs, t)
        obs, r, done, info = env.step(beta)
        out["reward"].append(r)
        out["recall"].append(info["recall"])
        out["beta"].append(beta)
        t += 1
    return out


POLICIES = {
    "greedy (beta=0)": lambda o, t: 0.0,
    "fixed beta=0.25": lambda o, t: 0.25,
    "fixed beta=1": lambda o, t: 1.0,
    "explore->exploit (1,1,0,0,0,0)": lambda o, t: 1.0 if t < 2 else 0.0,
    "exploit->explore (0,0,0,0,1,1)": lambda o, t: 1.0 if t >= 4 else 0.0,
}


def compare(regime, seeds=10):
    env = ToyCampaign(regime)
    print(f"\n=== regime: {regime}  (mean over {seeds} seeds) ===")
    print(f"{'policy':32s} {'sum of rewards':>15s} {'final recall':>13s}   recall by round")
    for name, pol in POLICIES.items():
        runs = [rollout(env, pol, s) for s in range(seeds)]
        G = np.mean([sum(r["reward"]) for r in runs])
        rec = np.mean([r["recall"] for r in runs], axis=0)
        print(f"{name:32s} {G:15.2f} {rec[-1]:13.3f}   {np.round(rec, 2)}")


def trace(regime="hidden", seed=0, beta=0.25):
    env = ToyCampaign(regime)
    obs = env.reset(seed)
    print(f"\nOne '{regime}' episode at beta={beta}. The controller sees ONLY the obs vector:")
    print("   " + " ".join(f"{n[:10]:>10s}" for n in FEATURE_NAMES))
    done = False
    while not done:
        print("o: " + " ".join(f"{v:10.3f}" for v in obs))
        obs, r, done, info = env.step(beta)
        print(f"   -> a={beta}  r={r:.2f}  (hidden from controller: recall={info['recall']:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    if args.trace:
        trace()
    else:
        compare("smooth")
        compare("hidden")

# EXERCISE 1: Fill in a table in your notes: for each regime, which policy maximizes the SUM OF
#   REWARDS and which maximizes FINAL RECALL? On 'hidden' they disagree. That disagreement is the
#   whole reason RL (module 03, 05) exists on top of bandits (module 01, 04).
# EXERCISE 2: Run --trace. Which observation features look different on 'hidden' vs 'smooth'
#   in round 0? (Try trace("smooth").) A contextual controller can only adapt to the regime if
#   some feature reveals it. If none do in round 0, what does that imply for round-0 decisions?
# EXERCISE 3: Write your own policy: "explore while headroom is large, then greedy" using
#   obs[FEATURE_NAMES.index("headroom")]. Add it to POLICIES. Does it beat the best fixed beta
#   on BOTH regimes? (That is the bar every controller in adaptive_acq has to clear.)
# EXERCISE 4: The env's reward uses a cutoff estimated from the 100 random initial molecules.
#   Why not reward recall directly? (Read the docstring of adaptive_acq/state.py.)
