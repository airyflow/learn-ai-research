"""An LLM agent as the campaign controller: tools, constrained actions, validation, cost.

IN PLAIN WORDS: an LLM picks beta by calling tools (look at the state, look at the history, set
the action). Every action is checked twice -- is it well-formed (Pydantic), is it allowed now
(PolicyGuard) -- and a rejected action goes back to the LLM to fix. First read: sections 1, 2,
3 and 5; skip 4 (backends). Core exercises: 1 and 3a. Walkthrough: LESSONS.md, Lesson 6.

Run:   python 06_agents/agent_controller.py --selftest                 (validation checks, instant)
       python 06_agents/agent_controller.py --backend mock --trace     (one campaign, every tool call)
       python 06_agents/agent_controller.py --backend mock             (agent vs baselines, ~30 s)
       sbatch 06_agents/run_vllm_agent.slurm                          (local gpt-oss-20b via vLLM on
           one H100: free, uses your allocation; same setup as NeuroPathology)
       python 06_agents/agent_controller.py --backend claude --seeds 3 (real Claude; needs
           `pip install anthropic` and ANTHROPIC_API_KEY or `ant auth login`; costs money --
           the script prints tokens and $ at the end)

What makes this an AGENT and not "an LLM that says UCB":
    observe (tools) -> reason -> act (a tool call) -> the action is VALIDATED -> real
    computation (one AL round) -> new evidence -> next decision.

The design rules, each visible in the code below:
  1. The agent never picks molecules. Its only action is set_acquisition(strategy, beta, reason);
     UCB picks the molecules. (Same as every controller in modules 01-05 and adaptive_acq.)
  2. Two validation layers. SCHEMA (Pydantic): is the action well-formed? POLICY (PolicyGuard):
     is it allowed here? e.g. no exploring in the last round (module 03, exercise 1).
  3. A rejected action is returned to the model as an error tool_result, and it gets to fix it --
     the same self-correcting retry loop as NeuroPathology's extraction pipeline. After
     MAX_TURNS it falls back to greedy, and the fallback is logged, never silent.
  4. Every decision is logged with its inputs and stated reason (JSONL), so runs can be audited:
     does the reason match the action? Does the action match the state?
  5. Cost is part of the result. An LLM controller that matches LinUCB at $2 per campaign is
     a worse controller than LinUCB.

The manual loop below is deliberate: you should see every message. In production you'd likely
use the SDK's tool runner (client.beta.messages.tool_runner), which runs this loop for you.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ARMS, FEATURE_NAMES, ToyCampaign  # noqa: E402

BETA_MAX = max(ARMS)
MAX_TURNS = 5
MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}      # claude-opus-5-5, USD


# ── 1. the action schema (layer 1: is it well-formed?) ───────────────────────

class Decision(BaseModel):
    strategy: Literal["greedy", "ucb"]
    beta: float = Field(ge=0.0, le=BETA_MAX)
    reason: str = Field(min_length=10, max_length=400)

    @model_validator(mode="after")
    def strategy_matches_beta(self):
        if self.strategy == "greedy" and self.beta != 0.0:
            raise ValueError("strategy 'greedy' requires beta = 0")
        if self.strategy == "ucb" and self.beta == 0.0:
            raise ValueError("strategy 'ucb' requires beta > 0 (use 'greedy' for beta = 0)")
        return self


# ── 2. domain constraints (layer 2: is it allowed HERE?) ─────────────────────

class PolicyGuard:
    """Rules that depend on the campaign state, kept out of the schema on purpose: the schema
    says what an action IS, the guard says what this deployment ALLOWS. In a real system this
    is where budget limits, GPU quotas and safety rules live."""

    def __init__(self, last_round_beta_cap=0.25):
        self.cap = last_round_beta_cap

    def check(self, d: Decision, rounds_left: int):
        if rounds_left == 1 and d.beta > self.cap:
            return (f"beta={d.beta} rejected: this is the final round, so exploration cannot pay "
                    f"off later. Use beta <= {self.cap}.")
        return None


# ── 3. the tools ─────────────────────────────────────────────────────────────

def _schema(props, required):
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {"name": "get_campaign_state", "strict": True,
     "description": "Current campaign state: round number, rounds left, and the controller "
                    "features computed from the surrogate's predictions and the last batch.",
     "input_schema": _schema({}, [])},
    {"name": "get_round_history", "strict": True,
     "description": "Previous rounds of this campaign: the beta used and the hit rate obtained.",
     "input_schema": _schema({}, [])},
    {"name": "set_acquisition", "strict": True,
     "description": "Choose how THIS round's batch is acquired and end your turn. "
                    "strategy 'greedy' (beta must be 0) or 'ucb' (0 < beta <= 2, normalized: "
                    "beta=1 means an average uncertainty bonus is worth one std of predicted "
                    "score). Give a one-sentence reason grounded in the state.",
     "input_schema": _schema({"strategy": {"type": "string", "enum": ["greedy", "ucb"]},
                              "beta": {"type": "number"},
                              "reason": {"type": "string"}},
                             ["strategy", "beta", "reason"])},
]

FEATURE_HELP = {
    "progress": "round / total rounds",
    "last_reward": "fraction of last batch that were hits",
    "last_spearman": "rank correlation of predicted vs observed scores on last batch (model accuracy)",
    "last_sigma_err_corr": "rank correlation of uncertainty vs actual error on last batch (is sigma informative?)",
    "headroom": "how large uncertainty is relative to the spread of predictions, 0..1",
    "ucb_greedy_jaccard": "overlap between greedy's batch and beta=2's batch",
    "improvement": "recent improvement of the best molecules found, -1..1",
    "last_beta": "previous beta / 2",
}

SYSTEM = f"""You control the exploration level of an active-learning campaign that screens a \
molecule library with a surrogate model. Each round, a batch of molecules is chosen by UCB = \
mu + beta * sigma and sent to the oracle. Your goal is the highest FINAL recall of the \
library's top 1% after the last round. Exploring (beta > 0) usually lowers this round's hit \
rate but can improve the surrogate for later rounds; greedy (beta = 0) maximizes this round.

Each round: inspect the state with the tools, then call set_acquisition exactly once. If it \
returns an error, read it and call set_acquisition again with a corrected action.

State features: {json.dumps(FEATURE_HELP)}"""


class CampaignTools:
    """The agent's only window onto the environment. It sees what env.observe() sees -- never
    recall, never the regime, never the molecules."""

    def __init__(self, env, history):
        self.env, self.history = env, history

    def get_campaign_state(self):
        obs = self.env.observe()
        feats = {n: round(float(v), 3) for n, v in zip(FEATURE_NAMES, obs) if n not in ("bias", "is_first")}
        return {"round": self.env.rnd + 1, "rounds_total": self.env.n_rounds,
                "rounds_left": self.env.n_rounds - self.env.rnd, "features": feats}

    def get_round_history(self):
        return {"rounds": self.history}


# ── 4. the LLM backends: same interface, so the loop doesn't care ────────────

def _block(**kw):
    return SimpleNamespace(**kw)


class MockLLM:
    """A scripted stand-in that speaks the same message protocol as the API: it emits
    tool_use blocks, reads tool_result blocks, and fixes its action after an error. Use it to
    test the plumbing offline and for free. Its 'reasoning' is a hand-written rule close to
    what GRPO learned in module 05 (explore early, then moderate, greedy at the end)."""

    def __init__(self, show_repair=False):
        self.show_repair, self.n = show_repair, 0

    def create(self, system, messages, tools):
        self.n += 1
        last = messages[-1]["content"]
        results = [b for b in last if isinstance(b, dict) and b.get("type") == "tool_result"] \
            if isinstance(last, list) else []
        usage = _block(input_tokens=0, output_tokens=0)
        if not results:                                     # turn 1: look before acting (parallel calls)
            return _block(stop_reason="tool_use", usage=usage, content=[
                _block(type="text", text="Let me look at the campaign first."),
                _block(type="tool_use", id=f"m{self.n}a", name="get_campaign_state", input={}),
                _block(type="tool_use", id=f"m{self.n}b", name="get_round_history", input={})])
        state = next((json.loads(r["content"]) for r in results if '"features"' in r["content"]), None)
        if state is None:                                   # last action was rejected: repair it
            err = results[0]["content"]
            beta = 0.25 if "final round" in err else min(BETA_MAX, 1.0)
            return self._act("ucb", beta, f"Previous action rejected ({err[:60]}...); corrected.")
        f, left = state["features"], state["rounds_left"]
        if self.show_repair and state["round"] == 1:
            return self._act("ucb", 3.0, "Explore aggressively at the start.")  # invalid: beta > 2
        if left == 1:
            return self._act("greedy", 0.0, "Final round: nothing left to learn, so exploit.")
        if state["round"] == 1:
            return self._act("ucb", 1.0, "First round, no feedback yet: explore to map the space.")
        if f["last_sigma_err_corr"] > 0.3:
            return self._act("ucb", 0.5, f"sigma tracks error (corr {f['last_sigma_err_corr']}), "
                                         "so uncertainty is worth paying for.")
        return self._act("ucb", 0.25, "Uncertainty is weakly informative; explore lightly.")

    def _act(self, strategy, beta, reason):
        return _block(stop_reason="tool_use", usage=_block(input_tokens=0, output_tokens=0),
                      content=[_block(type="tool_use", id=f"m{self.n}", name="set_acquisition",
                                      input={"strategy": strategy, "beta": beta, "reason": reason})])


class ClaudeLLM:
    def __init__(self, model=MODEL, effort="low"):
        import anthropic                                    # only needed for this backend
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()                 # reads ANTHROPIC_API_KEY or `ant auth login`
        self.model, self.effort = model, effort

    def create(self, system, messages, tools):
        # Refusal fallbacks: if a safety classifier declines, the API re-runs the request on a
        # fallback model inside the same call ("default" routes by refusal category).
        return self.client.beta.messages.create(
            model=self.model, max_tokens=16000, system=system, tools=tools, messages=messages,
            output_config={"effort": self.effort},          # one small decision: low effort is plenty
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )


class VLLMLLM:
    """A local open-weight model behind vLLM's OpenAI-compatible server (the same setup as
    NeuroPathology: gpt-oss-20b on one H100, see run_vllm_agent.slurm). Costs no API money.

    mode="native" (default): the tools are sent in the request's `tools` field and the model
        answers with real tool calls. Needs the server started with
        `--enable-auto-tool-choice --tool-call-parser openai` (run_vllm_agent.slurm does).
    mode="json": no tool support needed. The tools are described in the prompt and the model
        answers with {"calls": [{"tool": ..., "input": {...}}]} as plain JSON text.

    Why "native" is the default -- a lesson from the first GPU run (job 11915188, Oct 12): in
    json mode gpt-oss kept trying to make NATIVE tool calls anyway (the prompt talks about
    tools), which the server, started without a tool parser, returned as EMPTY replies (248 of
    248) or crashed on (14 HarmonyErrors). Every decision fell back to greedy. The outcome
    numbers looked fine (it matched greedy); only the fallback rate (1.000) gave it away.

    Either way, this adapter turns the reply into the same tool_use blocks the other backends
    return, so decide_round() can't tell the difference."""

    def __init__(self, url, model="openai/gpt-oss-20b", reasoning_effort="low", mode="native"):
        from openai import OpenAI                           # the client vLLM's server speaks
        self.client = OpenAI(base_url=f"{url.rstrip('/')}/v1", api_key="local")
        self.model, self.effort, self.mode, self.n = model, reasoning_effort, mode, 0

    # ── json mode: tools described in text ──
    @staticmethod
    def protocol(tools):
        lines = [f'- {t["name"]}: {t["description"]} Input schema: {json.dumps(t["input_schema"]["properties"])}'
                 for t in tools]
        return ("\n\nACTIONS. You can request these actions:\n" + "\n".join(lines) +
                '\n\nReply with ONLY one JSON object and no other text:\n'
                '{"calls": [{"tool": "<action name>", "input": {<arguments>}}]}\n'
                "You may put several calls in the list. You will get their results back.")

    def _to_openai_json(self, system, messages, tools):
        out, names = [{"role": "system", "content": system + self.protocol(tools)}], {}
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                out.append({"role": m["role"], "content": c})
            elif m["role"] == "assistant":                  # our blocks -> the JSON the model wrote
                calls = []
                for b in c:
                    if b.type == "tool_use":
                        names[b.id] = b.name
                        calls.append({"tool": b.name, "input": b.input})
                text = " ".join(b.text for b in c if b.type == "text")
                out.append({"role": "assistant", "content": json.dumps({"calls": calls}) if calls else text})
            else:                                           # tool results -> one user message
                res = [{"tool": names.get(r["tool_use_id"], "?"), "is_error": r["is_error"], "result": r["content"]}
                       for r in c]
                out.append({"role": "user", "content": "Results:\n" + json.dumps(res)})
        return out

    # ── native mode: OpenAI-style tools and tool messages ──
    @staticmethod
    def _openai_tools(tools):
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in tools]

    def _to_openai_native(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                out.append({"role": m["role"], "content": c})
            elif m["role"] == "assistant":
                text = " ".join(b.text for b in c if b.type == "text")
                calls = [{"id": b.id, "type": "function",
                          "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                         for b in c if b.type == "tool_use"]
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:                                           # one "tool" message per result
                for r in c:
                    content = ("ERROR: " if r["is_error"] else "") + r["content"]
                    out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": content})
        return out

    @staticmethod
    def clean_name(name):
        """gpt-oss sometimes decorates tool names: 'functions.set_acquisition', or a name with
        format tokens stuck on ('set_acquisition<|channel|>commentary'). Strip those."""
        name = (name or "").strip()
        name = name.split("<|")[0].split()[0] if name else name
        return name[len("functions."):] if name.startswith("functions.") else name

    def create(self, system, messages, tools):
        self.n += 1
        if self.mode == "native":
            resp = self.client.chat.completions.create(
                model=self.model, messages=self._to_openai_native(system, messages),
                tools=self._openai_tools(tools), tool_choice="auto",
                max_tokens=4096, temperature=0.2, extra_body={"reasoning_effort": self.effort})
        else:
            resp = self.client.chat.completions.create(
                model=self.model, messages=self._to_openai_json(system, messages, tools),
                max_tokens=4096, temperature=0.2, extra_body={"reasoning_effort": self.effort})
        choice = resp.choices[0]
        usage = _block(input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)

        blocks = []
        for tc in (getattr(choice.message, "tool_calls", None) or []):   # native tool calls
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {"_unparseable_arguments": tc.function.arguments}  # validation will reject it
            blocks.append(_block(type="tool_use", id=tc.id, name=self.clean_name(tc.function.name), input=args))
        text = choice.message.content or ""
        if not blocks and text:                                           # JSON written as text
            try:
                obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
                calls = obj["calls"] if isinstance(obj, dict) else None
                blocks = [_block(type="tool_use", id=f"v{self.n}_{i}", name=c["tool"], input=c.get("input") or {})
                          for i, c in enumerate(calls) if isinstance(c, dict) and "tool" in c]
            except (ValueError, KeyError, TypeError):
                blocks = []
        if blocks:
            return _block(stop_reason="tool_use", usage=usage, content=blocks)
        # Nothing usable. Say WHY in the log (lesson 8 counts these).
        why = f"finish_reason={choice.finish_reason}, {'empty' if not text else 'no JSON'}"
        return _block(stop_reason="end_turn", usage=usage,
                      content=[_block(type="text", text=f"(unparseable reply: {why}) {text[:200]}")])


# ── 5. the agent loop for one round ─────────────────────────────────────────

def decide_round(llm, tools, guard, usage_total):
    """Returns (Decision, trace). Falls back to greedy -- and says so -- if the model never
    produces a valid action within MAX_TURNS."""
    rounds_left = tools.env.n_rounds - tools.env.rnd
    messages = [{"role": "user", "content": f"Round {tools.env.rnd + 1} of {tools.env.n_rounds}. "
                                            "Decide this round's acquisition."}]
    trace = []
    for _ in range(MAX_TURNS):
        try:
            resp = llm.create(SYSTEM, messages, TOOLS)
        except Exception as e:                              # API errors: see note in ClaudeLLM
            trace.append({"error": f"{type(e).__name__}: {e}"[:300]})
            break
        usage_total["input"] += resp.usage.input_tokens
        usage_total["output"] += resp.usage.output_tokens
        usage_total["calls"] += 1
        if resp.stop_reason == "refusal":
            trace.append({"refusal": True})
            break
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        trace += [{"text": b.text} for b in resp.content if b.type == "text" and b.text.strip()]
        if not calls:
            trace.append({"no_tool_call": True})            # replied in text only; re-prompt
            messages.append({"role": "user", "content": "Call set_acquisition to finish this round."})
            continue
        results, accepted = [], None
        for c in calls:                                     # run ALL calls, return ALL results together
            if c.name == "set_acquisition":
                try:
                    d = Decision(**c.input)
                    problem = guard.check(d, rounds_left)
                except ValidationError as e:
                    d, problem = None, "invalid action: " + "; ".join(x["msg"] for x in e.errors())
                if problem:
                    out, is_err = problem, True
                else:
                    out, is_err, accepted = "accepted", False, d
                trace.append({"tool": c.name, "input": c.input, "result": out})
            elif c.name in ("get_campaign_state", "get_round_history"):
                out, is_err = json.dumps(getattr(tools, c.name)()), False
                trace.append({"tool": c.name})
            else:                                           # log it: unlogged failures are invisible
                valid = ", ".join(t["name"] for t in TOOLS)
                out, is_err = f"unknown tool {c.name!r}. Valid tools: {valid}.", True
                trace.append({"tool": c.name, "unknown": True, "input": c.input})
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": is_err})
        messages.append({"role": "user", "content": results})
        if accepted:
            return accepted, trace
    fb = Decision(strategy="greedy", beta=0.0, reason="FALLBACK: no valid action from the agent")
    trace.append({"fallback": True})
    return fb, trace


def run_agent_campaign(env, llm, seed, log=None, verbose=False, tools_cls=None, agent_name=None):
    """tools_cls: swap in a different tool set (06_agents/injection_test.py uses a poisoned one).
    agent_name: written into every log record, so lesson 8 can compare agents."""
    env.reset(seed=seed)
    history, guard, usage = [], PolicyGuard(), {"input": 0, "output": 0, "calls": 0}
    tools = (tools_cls or CampaignTools)(env, history)
    done = False
    while not done:
        rnd = env.rnd + 1
        before = dict(usage)
        t0 = time.perf_counter()
        d, trace = decide_round(llm, tools, guard, usage)
        seconds = time.perf_counter() - t0
        _, reward, done, info = env.step(d.beta)
        history.append({"round": rnd, "beta": d.beta, "hit_rate": round(reward, 3)})
        rec = {"agent": agent_name or type(llm).__name__, "seed": seed, "regime": info["regime"],
               "round": rnd, "decision": d.model_dump(), "hit_rate": reward, "recall": info["recall"],
               "seconds": round(seconds, 3), "model_calls": usage["calls"] - before["calls"],
               "input_tokens": usage["input"] - before["input"], "output_tokens": usage["output"] - before["output"],
               "trace": trace}
        if log:
            log.write(json.dumps(rec, default=str) + "\n")
        if verbose:
            print(f"\n-- round {rnd} --")
            for t in trace:
                print("   ", json.dumps(t, default=str)[:160])
            print(f"   => beta={d.beta}  hit rate={reward:.2f}  (hidden from agent: recall={info['recall']:.2f})")
    return info["recall"], [h["beta"] for h in history], usage


def run_fixed(env, beta, seed):
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, info = env.step(beta)
    return info["recall"]


# ── 6. self-test of the validation layers ───────────────────────────────────

def selftest():
    ok = Decision(strategy="ucb", beta=0.5, reason="sigma is informative early on")
    assert ok.beta == 0.5
    for bad in [dict(strategy="ucb", beta=3.0, reason="x" * 20),        # out of range
                dict(strategy="greedy", beta=0.5, reason="x" * 20),     # greedy with beta > 0
                dict(strategy="ucb", beta=0.0, reason="x" * 20),        # ucb with beta = 0
                dict(strategy="explore", beta=1.0, reason="x" * 20),    # unknown strategy
                dict(strategy="ucb", beta=1.0, reason="ok")]:           # reason too short
        try:
            Decision(**bad)
            raise AssertionError(f"accepted invalid action {bad}")
        except ValidationError as e:
            print(f"rejected {bad['strategy']}/{bad['beta']}: {e.errors()[0]['msg']}")
    g = PolicyGuard()
    assert g.check(ok, rounds_left=1) is not None and g.check(ok, rounds_left=3) is None
    print("selftest passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mock", "vllm", "claude"], default="mock")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--vllm-model", default="openai/gpt-oss-20b")
    p.add_argument("--vllm-mode", choices=["native", "json"], default="native",
                   help="native tool calling (needs the server's tool parser) or JSON in text")
    p.add_argument("--regime", choices=["smooth", "hidden", "both"], default="both")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--trace", action="store_true", help="one hidden-regime campaign, every tool call")
    p.add_argument("--show-repair", action="store_true", help="mock proposes an invalid action in round 1")
    p.add_argument("--log", default="06_agents/decisions.jsonl")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()

    if args.backend == "mock":
        llm = MockLLM(show_repair=args.show_repair or args.trace)
    elif args.backend == "vllm":
        llm = VLLMLLM(args.vllm_url, args.vllm_model, reasoning_effort=args.effort, mode=args.vllm_mode)
    else:
        llm = ClaudeLLM(effort=args.effort)
    if args.trace:
        env = ToyCampaign("hidden")
        r, betas, _ = run_agent_campaign(env, llm, seed=10_000, verbose=True)
        print(f"\nfinal recall {r:.3f}, betas {betas}")
        return

    regimes = ["smooth", "hidden"] if args.regime == "both" else [args.regime]
    seeds = list(range(10_000, 10_000 + args.seeds))
    total = {"input": 0, "output": 0, "calls": 0}
    t0 = time.perf_counter()
    with open(args.log, "w") as log:
        print(f"{'regime':8s} {'greedy':>8s} {'beta=0.25':>10s} {'agent':>8s}   agent betas (first seed)")
        for reg in regimes:
            env = ToyCampaign(reg)
            g = [run_fixed(env, 0.0, s) for s in seeds]
            f = [run_fixed(env, 0.25, s) for s in seeds]
            a, first = [], None
            for s in seeds:
                r, betas, u = run_agent_campaign(env, llm, s, log=log)
                a.append(r)
                first = first or betas
                for k in total:
                    total[k] += u[k]
            print(f"{reg:8s} {np.mean(g):8.3f} {np.mean(f):10.3f} {np.mean(a):8.3f}   {first}")
    cost = total["input"] / 1e6 * PRICE_PER_MTOK["input"] + total["output"] / 1e6 * PRICE_PER_MTOK["output"]
    n_camp = len(regimes) * len(seeds)
    print(f"\n{total['calls']} model calls, {total['input']} input + {total['output']} output tokens"
          + (f" = ${cost:.2f} (${cost / n_camp:.3f} per campaign)" if args.backend == "claude"
             else f" ({args.backend}: no API cost)")
          + f", {time.perf_counter() - t0:.0f}s. Decisions logged to {args.log}")


if __name__ == "__main__":
    main()

# EXERCISE 1: Run --selftest, then --trace. Follow one round: which tools are called, in what
#   order, what comes back? Find the rejected action in round 1 and the repair. Where in
#   decide_round() does the error text go back to the model?
# EXERCISE 2: Mock vs baselines (default run). The mock's rule is close to GRPO's policy. Is it
#   good on BOTH regimes? Compare with module 05's GRPO (trained on 'hidden' only).
# EXERCISE 3a (free): sbatch 06_agents/run_vllm_agent.slurm, then read logs/vllm_agent_*.out
#   and 06_agents/decisions_vllm.jsonl. How often did gpt-oss-20b produce unparseable JSON or an
#   invalid action that needed repair? Those rates are your first agent-reliability metrics.
# EXERCISE 3 (Claude, ~$1): --backend claude --seeds 3. Read decisions.jsonl. For each round,
#   does the stated reason match the action and the state? Count rounds where it doesn't. That
#   count is a metric -- this is the eval work from the NeuroPathology half of your plan.
# EXERCISE 3b (free, GPU): the fixed local run (decisions_vllm_run3_fixed.jsonl) chose beta = 1
#   in 44 of 60 rounds. Read 5 of its reasons. The SYSTEM prompt never says how big beta = 1 is.
#   Add one sentence to SYSTEM, e.g. "beta = 0.25 is light exploration; beta = 1 is strong, and
#   on most libraries beta above 0.5 lowers final recall." Re-run. Did the betas change? Did
#   recall? Write down that you gave the model knowledge from lessons 0-5 -- an LLM controller
#   is often only as good as the facts in its prompt.
# EXERCISE 4: Ablate the prompt. Delete the sentence in SYSTEM about exploring lowering this
#   round's hit rate but helping later. Does Claude's beta schedule change? What does that tell
#   you about where the "intelligence" in an LLM controller comes from?
# EXERCISE 5: Add a tool get_counterfactual_hint() that returns env.peek(beta) for each arm.
#   It would help the agent -- why must you NOT ship it? (Module 04: peek needs the oracle on
#   molecules you didn't acquire.) Leakage through tools is the agent version of test-set leakage.
# EXERCISE 6: Port this to adaptive_acq as `class LLMController(Controller)`: plan() calls
#   decide_round() and returns _single(arm, beta, batch_size). Snap beta to the nearest of
#   controller.arms, or let the agent use any beta? Decide, and write the reason in the docstring.
