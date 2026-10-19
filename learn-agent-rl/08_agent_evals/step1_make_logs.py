"""Lesson 8, step 1 of 5: make some agent logs to evaluate.

Run:   python 08_agent_evals/step1_make_logs.py       (about 15 seconds)

IN PLAIN WORDS
To practise evaluating agents we need agents to evaluate. We use two mock agents from lesson 6:

  careful   the normal MockLLM: always uses the tools, always gives a valid action
  sloppy    the same rules, but it misbehaves at random, the way real LLMs sometimes do:
              - sometimes it answers in text and forgets to call any tool
              - sometimes it sends an invalid action (strategy 'greedy' with beta > 0)
              - sometimes its stated reason contradicts its action
              ("Exploit now" while choosing beta = 0.5)

Each agent runs 10 campaigns on each library. Every round is one line in
08_agent_evals/out/agent_logs.jsonl, in the same format as lesson 6's decisions.jsonl. So
everything in steps 2-5 also works on REAL logs, e.g. from the local model run.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.loader import load_lesson  # noqa: E402
from common.toy_campaign import ToyCampaign  # noqa: E402

agent = load_lesson("06_agents/agent_controller.py")
OUT = Path(__file__).resolve().parent / "out" / "agent_logs.jsonl"


class SloppyMock(agent.MockLLM):
    """MockLLM plus random mistakes. Seeded, so the logs are reproducible."""

    def __init__(self, seed):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def _act(self, strategy, beta, reason):
        r = self.rng.random()
        if r < 0.10:                                   # forgets to call a tool
            return agent._block(stop_reason="end_turn",
                                usage=agent._block(input_tokens=0, output_tokens=0),
                                content=[agent._block(type="text", text="I think we should explore a bit.")])
        if r < 0.25:                                   # invalid: greedy must have beta = 0
            return super()._act("greedy", beta if beta > 0 else 0.5, reason)
        if r < 0.40 and beta > 0:                      # reason contradicts the action
            return super()._act(strategy, beta, "Exploit now: the model is good enough.")
        return super()._act(strategy, beta, reason)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    seeds = range(10_000, 10_010)
    with open(OUT, "w") as log:
        for regime in ("smooth", "hidden"):
            env = ToyCampaign(regime)
            for s in seeds:
                agent.run_agent_campaign(env, agent.MockLLM(), s, log=log, agent_name="careful")
                agent.run_agent_campaign(env, SloppyMock(seed=s), s, log=log, agent_name="sloppy")
    n = sum(1 for _ in open(OUT))
    print(f"wrote {n} rounds (2 agents x 2 libraries x 10 campaigns x 6 rounds) to {OUT}")

# EXERCISE 1: Open out/agent_logs.jsonl. Find one 'sloppy' round whose "trace" contains a
#   rejected set_acquisition. What did the agent try, and what did it do next?
# EXERCISE 2: Change the 0.10 / 0.25 / 0.40 thresholds to make the sloppy agent worse. Re-run
#   steps 3 and 4 later and watch the numbers move. (Good eval metrics should move!)
