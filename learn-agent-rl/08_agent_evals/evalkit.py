"""Small helpers shared by lesson 8's steps. Read this file once; it's short.

    rows      = load_rounds(path)    one dict per round (one JSONL line)
    campaigns = group_campaigns(rows) {(agent, regime, seed): [rounds in order]}
"""
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent / "out" / "agent_logs.jsonl"


def load_rounds(path=DEFAULT_LOG):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:                       # older logs (lesson 6 before this lesson) had no "agent"
        r.setdefault("agent", "agent")
    return rows


def group_campaigns(rows):
    campaigns = defaultdict(list)
    for r in rows:
        campaigns[(r["agent"], r["regime"], r["seed"])].append(r)
    for rounds in campaigns.values():
        rounds.sort(key=lambda r: r["round"])
    return dict(campaigns)


def attempts(round_row):
    """The set_acquisition attempts in one round, in order: a list of 'accepted' or the error."""
    return [t["result"] for t in round_row["trace"] if t.get("tool") == "set_acquisition"]


def had_no_tool_reply(round_row):
    """True if the model answered in plain text at least once instead of calling a tool."""
    return any(t.get("no_tool_call") or str(t.get("text", "")).startswith("(unparseable")
               for t in round_row["trace"])


def unknown_tool_calls(round_row):
    """How many tool calls in this round used a name that doesn't exist."""
    return sum(1 for t in round_row["trace"] if t.get("unknown"))


def is_fallback(round_row):
    return any(t.get("fallback") for t in round_row["trace"])
