"""Lesson 7, step 1 of 4: what an environment is -- reset() and step().

Run:   python 07_environments/step1_reset_and_step.py      (about 3 seconds)

IN PLAIN WORDS
Every RL environment -- the toy campaign here, Gymnasium's games, the "RL environments" that
OpenAI, Anthropic and Scale hire whole teams to build -- offers the same two calls:

    obs = env.reset(seed)                       start a new episode, get the first observation
    obs, reward, done, info = env.step(action)  do one action, see what happened

That's the whole contract. An agent only ever talks to the world through these two calls.

This script does two things:
  1. plays one campaign with a RANDOM beta each round and prints every step
  2. plays the same campaign again with the same seed and checks it is IDENTICAL

Why check 2? An environment must be reproducible. If the same seed and the same actions can
give different results, you can't debug a bad run, you can't compare two agents fairly, and
you can't trust a training curve. "Deterministic given the seed" is rule number one for
environment builders.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ARMS, ToyCampaign  # noqa: E402


def play(env, seed, verbose=False):
    """One episode with a random policy. Returns the list of (action, reward) pairs."""
    rng = np.random.default_rng(seed)          # the POLICY's randomness, also seeded
    obs = env.reset(seed=seed)                 # call 1: reset
    steps, done = [], False
    while not done:
        beta = float(rng.choice(ARMS))         # random policy: any beta
        obs, reward, done, info = env.step(beta)   # call 2: step
        steps.append((beta, reward))
        if verbose:
            print(f"  round {info['round']}:  action beta={beta:<5}  reward={reward:.2f}  "
                  f"done={done}   (info, for us not the agent: recall={info['recall']:.2f})")
    return steps


if __name__ == "__main__":
    env = ToyCampaign("hidden")

    print("1) One episode with a random policy, seed 0:")
    first = play(env, seed=0, verbose=True)

    print("\n2) Same seed again. Is it identical?")
    second = play(env, seed=0)
    print("  identical:", first == second)
    assert first == second, "the environment is not deterministic!"

    print("\n3) A different seed gives a different episode:")
    other = play(env, seed=1)
    print("  identical to seed 0:", first == other)

# EXERCISE 1: Read ToyCampaign.reset() and ToyCampaign.step() in common/toy_campaign.py. Find
#   where each of the four outputs of step() (obs, reward, done, info) is computed.
# EXERCISE 2: What does the agent see, and what is only in `info`? Why must recall stay out of
#   the observation? (Hint: a real screen doesn't know the true top 1%.)
# EXERCISE 3: Break determinism on purpose: in play(), replace np.random.default_rng(seed) with
#   np.random.default_rng() (no seed). Run again. Which check fails? Put it back.
