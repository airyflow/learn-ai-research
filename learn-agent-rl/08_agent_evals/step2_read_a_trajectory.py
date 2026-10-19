"""Lesson 8, step 2 of 5: read one trajectory like a detective.

Run:   python 08_agent_evals/step2_read_a_trajectory.py               (instant)
       python 08_agent_evals/step2_read_a_trajectory.py careful 10000 hidden

IN PLAIN WORDS
Before computing any metric, READ some trajectories. Every good eval starts with someone
looking at raw runs and writing down what goes wrong. Those notes become the failure
categories you count in step 3.

This prints one campaign round by round: every tool call, every rejected action, the final
decision and its stated reason.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import group_campaigns, load_rounds  # noqa: E402

if __name__ == "__main__":
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "sloppy"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    regime = sys.argv[3] if len(sys.argv) > 3 else "hidden"

    campaigns = group_campaigns(load_rounds())
    rounds = campaigns[(agent_name, regime, seed)]
    print(f"agent={agent_name}  library={regime}  seed={seed}\n")
    for r in rounds:
        print(f"round {r['round']}:")
        for t in r["trace"]:
            if "text" in t:
                print(f"   says:     {t['text']}")
            elif t.get("tool") == "set_acquisition":
                ok = t["result"] == "accepted"
                print(f"   acts:     beta={t['input'].get('beta')}  strategy={t['input'].get('strategy')}"
                      f"  -> {'accepted' if ok else 'REJECTED: ' + t['result'][:60]}")
            elif "tool" in t:
                print(f"   looks:    {t['tool']}")
            elif t.get("fallback"):
                print("   FALLBACK to greedy (no valid action in time)")
        d = r["decision"]
        print(f"   decision: beta={d['beta']}   reason: \"{d['reason']}\"")
        print(f"   result:   hit rate {r['hit_rate']:.2f}   (recall so far {r['recall']:.2f})\n")

# EXERCISE 1: Read the sloppy agent's campaign. Make a list of every kind of thing that went
#   wrong. Give each kind a short name (e.g. "no tool call", "invalid action", "reason mismatch").
#   That list is your FAILURE TAXONOMY -- step 3 counts it.
# EXERCISE 2: Now read a 'careful' campaign (args: careful 10000 hidden). Anything wrong there?
#   Is "nothing went wrong" the same as "it made good decisions"? (No -- step 3 measures both.)
