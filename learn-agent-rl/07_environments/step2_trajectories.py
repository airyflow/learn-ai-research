"""Lesson 7, step 2 of 4: record trajectories.

Run:   python 07_environments/step2_trajectories.py       (about 10 seconds)

IN PLAIN WORDS
A TRAJECTORY is the full record of one episode: at every step, what the agent saw, what it
did, and what it got. Trajectories are the raw material for almost everything else:
  - evaluation   (lesson 8 reads them to measure an agent)
  - training     (GRPO in lesson 5 learns from them)
  - debugging    (what was the agent looking at when it made that bad choice?)

Job postings call this "trajectory capture" or "rollout logging".

This script plays a few campaigns with three simple policies, writes every step as one line of
JSON (a "JSONL" file), then reads one trajectory back and prints it as a table.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ARMS, FEATURE_NAMES, ToyCampaign  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "trajectories.jsonl"

POLICIES = {
    "greedy": lambda obs, rng: 0.0,
    "beta_0.25": lambda obs, rng: 0.25,
    "random": lambda obs, rng: float(rng.choice(ARMS)),
}


def record(env, policy_name, seed):
    """Play one episode and return its trajectory: a list of step records."""
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    obs, done, steps = env.reset(seed=seed), False, []
    while not done:
        action = policy(obs, rng)
        next_obs, reward, done, info = env.step(action)
        steps.append({
            "policy": policy_name, "regime": env.regime, "seed": seed, "t": info["round"],
            "obs": {n: round(float(v), 3) for n, v in zip(FEATURE_NAMES, obs)},  # what it SAW
            "action": action,                                                     # what it DID
            "reward": round(reward, 3),                                           # what it GOT
            "done": done,
        })
        obs = next_obs
    return steps


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for regime in ("smooth", "hidden"):
            env = ToyCampaign(regime)
            for name in POLICIES:
                for seed in range(3):
                    for step in record(env, name, seed):
                        f.write(json.dumps(step) + "\n")
                        n += 1
    print(f"wrote {n} steps to {OUT}")

    # Read it back: one trajectory = all lines with the same (policy, regime, seed).
    rows = [json.loads(line) for line in open(OUT)]
    one = [r for r in rows if r["policy"] == "random" and r["regime"] == "hidden" and r["seed"] == 0]
    print("\nThe 'random' policy on 'hidden', seed 0:")
    print(f"  {'t':>2}  {'headroom':>8}  {'last_reward':>11}  {'action':>6}  {'reward':>6}")
    for r in one:
        print(f"  {r['t']:>2}  {r['obs']['headroom']:>8}  {r['obs']['last_reward']:>11}  "
              f"{r['action']:>6}  {r['reward']:>6}")

# EXERCISE 1: Open 07_environments/out/trajectories.jsonl in VS Code. One line = one step.
#   How many lines does one trajectory have, and why?
# EXERCISE 2: Using `rows`, compute the average reward per policy and regime (a dict and a loop
#   are enough). Which policy has the highest average reward on 'hidden'? Keep that answer
#   for step 4.
# EXERCISE 3: Notice there is no "recall" in these records. Why might you deliberately keep the
#   ground truth out of the trajectory file? (Step 3 answers this.)
