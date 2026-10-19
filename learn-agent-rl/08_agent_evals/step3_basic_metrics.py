"""Lesson 8, step 3 of 5: turn logs into numbers.

Run:   python 08_agent_evals/step3_basic_metrics.py                         (instant)
       python 08_agent_evals/step3_basic_metrics.py 06_agents/decisions_vllm.jsonl   (real logs)

IN PLAIN WORDS
An agent eval measures two different things. Keep them apart:

  OUTCOME   did the task succeed?      task success rate (final recall >= 0.8), mean recall
  PROCESS   did it behave reliably?    first-try valid rate, repair rate, fallback rate,
                                       no-tool-call rate, calls per decision, time, tokens

An agent can have a perfect process and poor outcomes (it reliably makes bad choices), or a
messy process and fine outcomes (validation and repair are covering for it). You need both
columns to know which.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import (DEFAULT_LOG, attempts, group_campaigns, had_no_tool_reply, is_fallback,  # noqa: E402
                     load_rounds, unknown_tool_calls)

SUCCESS = 0.80


def metrics(rows):
    campaigns = group_campaigns(rows)
    final = [rounds[-1]["recall"] for rounds in campaigns.values()]
    tried = [r for r in rows if attempts(r)]
    first_ok = [attempts(r)[0] == "accepted" for r in tried]
    needed_repair = [r for r in rows if attempts(r) and attempts(r)[0] != "accepted"]
    repaired = [not is_fallback(r) for r in needed_repair]
    return {
        "campaigns": len(campaigns),
        "task success": np.mean([f >= SUCCESS for f in final]),
        "mean recall": np.mean(final),
        "first-try valid": np.mean(first_ok) if first_ok else float("nan"),
        "repair success": np.mean(repaired) if repaired else float("nan"),
        "fallback rate": np.mean([is_fallback(r) for r in rows]),
        "no-tool replies": np.mean([had_no_tool_reply(r) for r in rows]),
        "unknown-tool calls": np.mean([unknown_tool_calls(r) > 0 for r in rows]),
        "calls/decision": np.mean([r.get("model_calls", np.nan) for r in rows]),
        "sec/decision": np.mean([r.get("seconds", np.nan) for r in rows]),
        "tokens/decision": np.mean([r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in rows]),
    }


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:12d}"
    return f"{'n/a':>12s}" if np.isnan(v) else f"{v:12.3f}"    # n/a: nothing to measure


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    rows = load_rounds(path)
    agents = sorted({r["agent"] for r in rows})
    table = {a: metrics([r for r in rows if r["agent"] == a]) for a in agents}

    names = list(next(iter(table.values())).keys())
    print(f"{'metric':18s}" + "".join(f"{a:>12s}" for a in agents))
    for i, name in enumerate(names):
        if i == 3:
            print("-" * (18 + 12 * len(agents)) + "   (above: outcome, below: process)")
        vals = "".join(fmt(table[a][name]) for a in agents)
        print(f"{name:18s}{vals}")
    print("\nMock agents use no tokens; the time and token rows matter for real backends.")

# EXERCISE 1: The sloppy agent's outcome is close to the careful one's. Which process metrics
#   show it is less reliable? Which layer from lesson 6 (schema, guard, repair, fallback) is
#   hiding its mistakes from the outcome numbers?
# EXERCISE 2: Why is "repair success" n/a for the careful agent? Is n/a the same as 0 or 1?
#   (A metric with no cases to measure must say so, not pretend to be a number.)
# EXERCISE 2b: The first version of "no-tool replies" in this file had a bug: it said 1.000 for
#   BOTH agents, because the mock says a sentence before calling tools. Look at
#   had_no_tool_reply() in evalkit.py -- what does it check instead? Metrics have bugs too:
#   sanity-check every new metric on an agent where you know the answer.
# EXERCISE 2c (a real failure): run this on 06_agents/decisions_vllm_run1_failed.jsonl, the first
#   local-model run (Oct 12). Mean recall looks fine (0.78, same as greedy). Which row shows that
#   the agent never made a single decision? Read one round with step 2's code to see why.
# EXERCISE 3 (after the GPU run): run this on 06_agents/decisions_vllm.jsonl. Now time and
#   tokens are real. Compute cost per SUCCESSFUL campaign: total tokens / number of successes.
