# Agent RL skills in demand (October 2026), in plain words

**Short answer:** companies hiring for agents and RL mostly want three things:
1. People who can **build the world an agent trains in** (an "RL environment").
2. People who can **score the agent's work fairly** (rewards, graders, evals).
3. People who can **run a GRPO-style training loop** with the standard tools.

Lessons 7 and 8 of this course already cover the first two. This page lists all eight skills that
keep showing up, explains each one simply, and says where to practise it.

*Checked 2026-10-12 against job postings and recent papers (sources at the end). Postings
change weekly, so treat the ranking as a snapshot, not a law.*

## The eight skills at a glance

| # | Skill | One-line meaning | Where you practise it | Status |
|---|---|---|---|---|
| 1 | Agent evals | Measuring whether an agent is actually good, with error bars | Lesson 8 | ✅ covered |
| 2 | RL environments | Packaging a task so a model can practise it: tasks + rules + score | Lesson 7, steps 1–3 | ✅ covered |
| 3 | Verifiable rewards and reward hacking | A score a program can check, and catching agents that cheat it | Lesson 7, steps 3–4 | ✅ covered |
| 4 | GRPO and its variants | The training method behind recent reasoning models, plus its 2025–26 fixes | Lesson 5 + `05_policy_gradient/grpo_variants.py` (new) | ✅ covered |
| 5 | Multi-turn tool-use RL | Training an agent that takes many steps and calls tools, rewarded only at the end | Lesson 6 (agent, not yet trained) | ⚠️ partial |
| 6 | Rubric rewards and LLM-as-judge | Using an LLM with a checklist to score answers no program can check | Lesson 8 step 5; plan week 6 | ⚠️ partial |
| 7 | MCP and function-calling environments | Environments where the "actions" are tool calls through a standard protocol | Plan week 4 | ⏳ planned |
| 8 | The standard training stack | verl, TRL, OpenRLHF, SkyRL to train; vLLM to generate; Verifiers for environments | Lesson 6 uses vLLM only | ❌ gap |

---

## 1. Agent evals

**In one sentence:** before claiming an agent works, measure it on many tasks, look at *how* it
got there as well as *whether*, and report a confidence interval.

**Everyday picture:** a driving test checks both that you arrived (outcome) and that you didn't run
a red light on the way (process).

**What it looks like in practice:** run the agent 50 times, compute success rate ± a 95% CI, then
read 10 trajectories by hand to see *why* it failed.

**Practise:** lesson 8, steps 1–5. **Next step:** apply `08_agent_evals/evalkit.py` to the real
`adaptive_acq` decision logs.

## 2. RL environments

**In one sentence:** an environment is a small program that hands the model a task, runs the
conversation or tool calls, and returns a score.

**Everyday picture:** a flight simulator. The pilot (model) acts, the simulator responds, and an
instructor (the grader) scores the landing.

**The three parts** (the same in every library):

| Part | What it is | In your toy campaign |
|---|---|---|
| Tasks (dataset) | The questions or starting situations | A random library + initial labeled set |
| Harness (rollout) | The loop that runs the turns | `reset()` → `step(beta)` × 6 rounds |
| Rubric (reward) | The scoring code | Final recall of the true top-k |

In Prime Intellect's open **Verifiers** library, loading a ready-made environment is one line,
`vf.load_environment("alphabet-sort")`. Its Environments Hub hosts hundreds of shared environments.

**Practise:** lesson 7, steps 1–2. **Next step:** rewrite `common/toy_campaign.py` as a
Verifiers-style environment (tasks / harness / rubric). This is a strong portfolio piece because
it matches what postings literally ask for.

## 3. Verifiable rewards and reward hacking

**In one sentence:** prefer rewards a program can check (a unit test passes, a number matches),
then try hard to break your own reward before the model does.

**Everyday picture:** if a teacher grades essays by word count, students write long essays, not
good ones. That's reward hacking.

**Practise:** lesson 7, steps 3 (grader) and 4 (reward hacking). Postings phrase this as "design
reward functions and verifiers that resist reward hacking".

## 4. GRPO and its variants

**In one sentence:** for each question, sample a *group* of answers, score them, and push the
model toward the answers that beat the group's average. No separate critic network is needed.

**Everyday picture:** a coach watches four attempts at the same jump and tells the athlete
"do more of what you did in attempts 1 and 2".

**The 2025–26 fixes, all small bookkeeping changes** (run `python 05_policy_gradient/grpo_variants.py`
to see them on numbers):

| Variant | What it changes | Why |
|---|---|---|
| **Dr. GRPO** | Stops dividing by the group's std and by each answer's length | Plain GRPO punishes long wrong answers less per token, so models learn to ramble |
| **DAPO** | Averages the loss over all tokens; drops groups where every answer scored the same; clips the ratio higher on the upside; penalises over-long answers | Fair weighting of tokens, no wasted batches, keeps exploring |
| **GSPO** | One importance ratio per *answer* instead of per token | Steadier training on long answers and mixture-of-experts models |

**Practise:** lesson 5, then the new script and its exercise 3: add a `--variant drgrpo` flag
and compare 5 seeds.

## 5. Multi-turn tool-use RL

**In one sentence:** the agent takes many steps (think, call a tool, read the result, call another
tool…), and the reward often arrives only at the very end.

**Everyday picture:** a chess game scored only win or lose. Which of your 40 moves deserves
the credit? That question is called **credit assignment**, and it is the main open problem here.

**What's new:**
- Frameworks such as AgentGym-RL train 7B models across many multi-turn tasks.
- "Rollout-as-a-service" designs (ProRL Agent, part of NVIDIA NeMo Gym) run the environment as a
  separate service so the trainer stays simple.
- Papers report a failure mode called reasoning collapse, where the agent's thinking becomes
  repetitive during RL.

**Your link:** an AL campaign is already multi-turn: 5–6 rounds, scored by final recall. Lessons
3–5 met credit assignment as "greedy wins every round but loses at the end".

**Next step:** train lesson 6's agent with lesson 5's GRPO, one group of episodes per campaign.

## 6. Rubric rewards and LLM-as-judge

**In one sentence:** when no program can check the answer (a summary, an explanation), give an
LLM a written checklist and let it score, and then *check the judge* against human labels.

**Everyday picture:** an exam with a marking scheme, plus a second marker who spot-checks the
first marker's grades.

**Practise:** lesson 8 step 5 (checking an agent's stated reasons). Week 6 of the plan builds a
validated judge on NeuroPathology.

## 7. MCP and function-calling environments

**In one sentence:** MCP (Model Context Protocol) is a standard way to expose tools to a model. An
MCP environment trains or tests an agent whose actions are those tool calls.

**Everyday picture:** a universal power socket. Any appliance (model) plugs into any outlet (tool
server) the same way.

Turing's staff posting names "MCP-based environments for general function-calling agents" next to
coding and browser-use environments. **Practise:** week 4 of the plan (`search_previous_campaigns`
as an MCP tool).

## 8. The standard training stack

**In one sentence:** in real jobs nobody writes GRPO from scratch. They use a library and spend
their time on environments and rewards.

| Tool | Job |
|---|---|
| **verl**, **TRL**, **OpenRLHF**, **SkyRL** | Training loops (GRPO/PPO) for LLMs |
| **vLLM** | Fast text generation for rollouts (you already use it in lesson 6 and NeuroPathology) |
| **Verifiers** + Environments Hub, **prime-rl** | Shared environments and async RL training |

**Next step:** run TRL's GRPO trainer on a tiny model (about 0.5B parameters) with a reward that
checks a number. Then swap in your toy-campaign reward. One GPU-hour is enough to learn the shape.

---

## What to do next (in order)

1. **This week:** run `grpo_variants.py` and do its exercise 3 (about 1 evening).
2. **Portfolio piece:** turn the toy campaign into a Verifiers-style environment (skill 2). It
   shows skills 2, 3 and 1 in one repo.
3. **Then:** one small TRL or verl GRPO run on a GPU (skill 8), so you can say "I've used the
   standard stack" in an interview.

## Sources

- Job postings (2026): [The LLM Data Company, Founding Senior Research Engineer](https://builtin.com/job/founding-senior-research-engineer/7372266) (asks for verl, PyTorch, GRPO, evals, reward engineering) · [Turing, Staff Research Engineer](https://job-boards.greenhouse.io/turing/jobs/5703799004) (coding, browser-use and MCP-based environments; RL and post-training)
- The "77% of postings mention evals" figure is from this course's own count of Bay Area postings (see [LESSONS.md](LESSONS.md)), not from the sources above.
- Environments: [Verifiers library (Prime Intellect)](https://github.com/PrimeIntellect-ai/verifiers) · [Exploring Environments Hub (Hugging Face blog, Sep 2025)](https://huggingface.co/blog/anakin87/environments-hub) · [Agentic Environment Engineering survey (arXiv, Jun 2026)](https://arxiv.org/abs/2606.12191)
- Multi-turn agentic RL: [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) · [ProRL Agent: rollout-as-a-service](https://arxiv.org/html/2603.18815v1) · [AgentRL](https://arxiv.org/pdf/2510.04206) · [Agentic Reinforced Policy Optimization](https://arxiv.org/pdf/2507.19849) · [Modal: open models for agentic RL in 2026](https://modal.com/resources/best-open-source-models-multi-turn-long-horizon-agentic-rl) · [MolAct: agentic RL for molecular editing](https://arxiv.org/pdf/2512.20135) (closest to your field)
- GRPO variants: [Overview of GRPO variants (arXiv)](https://arxiv.org/pdf/2510.15990v1) · [DAPO summary (Emergent Mind)](https://www.emergentmind.com/topics/dapo) · [Reasoning RL in 2026 (Turing Post)](https://www.turingpost.com/p/reasoning-rl-in-2026). The variant descriptions in skill 4 also draw on the original Dr. GRPO, DAPO and GSPO papers; check the exact settings there before quoting numbers.
