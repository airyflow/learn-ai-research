# Project roadmap: growing al-molecular into an adaptive scientific-AI system

This file is the *project* plan. The day-by-day schedule is in
[`../../career-plan/PLAN.md`](../../career-plan/PLAN.md); the *course* is [LESSONS.md](LESSONS.md).

## Where the project stands (updated 2026-10-12)

- **β is settled: no-go.** A β-sensitivity gate (130 ENHITS runs, rule fixed in advance: needs
  ≥ +0.005 recall at round 5 with CI > 0) failed. At 0.1% batches every β > 0.1 hurts; at
  0.2–0.4% batches small β gains only +0.003 to +0.005. Learned β controllers (LinUCB, GRPO)
  are **not justified** on ENHITS.
- **The simulator is calibrated** (batch 1%, top-k 0.25%; matches real D4 within 0.01).
- **Next direction: the cost axis.** Same recall for less compute: retrain or reuse the
  surrogate, cheap single backbone vs LT-All, candidate-pool prefiltering, stopping early.
  Any agent or bandit should choose among *these* actions, not β.
- The toy environment's `hidden` regime is where exploration pays (0.64 → 0.93). Use it to
  show *when* adaptivity matters; ENHITS behaves like `smooth`.

The stages below were written on 2026-10-11, before the β result. Where a stage says
"choose β", read it as "choose among the cost actions".

## The plan: 7 stages, about 14 weeks

Each stage has three parts. **Learn** (a course module plus reading), **Build** (a change to
al-molecular) and **Evidence** (the table or figure the stage must produce). Evaluation runs
through every stage: paired-by-seed differences with 95% CIs, always against greedy and
random, and the real compute and API cost recorded.

### Stage 0. RL foundations (weeks 1–2): *mostly written*
- **Learn:** modules 01–03 below. Sutton & Barto ch. 1–6 (skim 5–6); Lattimore & Szepesvári,
  *Bandit Algorithms*, ch. 1, 7, 8, 19, 36.
- **Build:** nothing in al-molecular yet. Do the exercises.
- **Evidence:** in your notes, explain in your own words why the bandit in 01 Part C picks greedy
  and why that is wrong on `hidden` (the gap between myopic and delayed value).

### Stage 1. Contextual bandit and a cost-aware action space (weeks 3–4)
- **Learn:** module 04 (LinUCB, full-information training, distribution shift).
- **Build:**
  1. Unblock the simulator build (`adaptive_acq/slurm/submit_build_sim_tasks.sh`), then run
     `train_linucb.py` and the ENHITS evaluation (array 45–54).
  2. **Widen the action space beyond β.** On ENHITS, β is not where the value is. The actions
     the pasted plan lists that change *cost* are more promising: `retrain` vs `reuse surrogate`,
     candidate-pool size (prefilter with FAISS top-N), batch size, and `stop early`. Add them
     to `Campaign`, and add a reward of the form
     `r_t = Δhits − λ₁·GPU-seconds − λ₂·labels`.
- **Evidence:** a table of LinUCB vs fixed β vs schedules on ENHITS (5 seeds, paired CIs).
  Then a recall vs GPU-hours curve. Even if no controller has higher recall, *same recall for
  fewer GPU-hours* is a legitimate win.

### Stage 2. LLM agent controller with function calling (weeks 5–6)
- **Learn:** module 06 (tool-use loop, Pydantic-constrained actions, validation, mock vs real
  backend). Anthropic, ["Building effective agents"](https://www.anthropic.com/engineering/building-effective-agents).
  Claude API tool-use docs.
- **Build:** `adaptive_acq/llm_controller.py` with `class LLMController(Controller)`:
  - **Tools:** `get_state()`, `get_uncertainty_stats()`, `get_history()`, and
    `set_acquisition(strategy, beta, retrain, pool_size)`.
  - **Safety layer:** every action passes Pydantic validation and then constraint checks
    (budget, β range) before execution. The agent **never picks molecules**; UCB does.
  - Every decision is logged with its rationale in `iter_N/controller.json`.
  - **Backends:** `mock` (rule-based, offline, used in tests), `vllm` (local gpt-oss-20b, no API
    cost, used for almost everything), and `claude` (a few dollars of API credit, final table only).
- **Evidence:** three controllers (fixed β, bandit, agent) on the same pool, budget and seeds:
  recall, area under the discovery curve, hits per GPU-hour, and the **agent's own cost**
  (tokens, $, latency per decision). Also a qualitative audit: are the rationales consistent
  with the actions?

### Stage 3. Retrieval (RAG) and MCP tools for the agent (weeks 7–8)
- **Learn:** BM25, dense retrieval, FAISS (which you already use), cross-encoder reranking, and
  retrieval metrics (recall@k, MRR). The MCP spec, by building one small server.
- **Build:** two retrieval tools, each with a measurable job:
  1. `search_previous_campaigns(context)`: FAISS over the context vectors and
     `history.json` of everything in `runs/`. *"In past campaigns that looked like this one,
     what happened at each β?"* This is the more useful tool, because it warm-starts decisions
     from your own data.
  2. `search_literature(query)`: BM25 plus dense retrieval plus a reranker, over a small
     corpus (a few hundred abstracts on D4/AmpC/EnamineHTS screening and AL acquisition).
  - Expose both through an **MCP server**, so any MCP client can call them.
- **Evidence:** retrieval quality on a hand-labeled query set (recall@5, MRR), and an ablation
  of agent vs agent + retrieval on the campaign metrics. If retrieval doesn't change the
  decisions, report that. It is a valid finding.

### Stage 4. Local LLM inference with vLLM (weeks 9–10)
- **Learn:** the vLLM docs (PagedAttention, continuous batching, quantization), and a
  throughput vs latency analysis.
- **Build:** a Slurm script that serves a 7–8B open-weight instruct model with vLLM on one
  H100, plus an `openweight` backend for `LLMController`.
- **Evidence:** a benchmark table (batch size → tokens/s, sequence length → p50/p95 latency,
  fp16 vs AWQ/GPTQ → GPU memory). Then the agent comparison again: how often does the local
  model choose the same action as Claude, and what does that do to recall and cost?

### Stage 5. LoRA fine-tuning of a decision model (weeks 11–12)
- **Learn:** LoRA/QLoRA with the PEFT library, and evaluation of fine-tuned models
  (accuracy, calibration, robustness).
- **Build:** distill a policy into the small model. The simulator gives you **free labels**:
  for each state, `env.peek` plus rollouts give the best arm (module 04's `arm_rewards`). So
  you can fine-tune *state JSON → action* without any human labeling.
- **Evidence:** a ladder of prompted base model → LoRA → LoRA + retrieval → LoRA + retrieval +
  tools, scored on action accuracy against the simulator's best arm, campaign recall, latency
  and memory.

### Stage 6. Policy gradient and GRPO (week 13; the code already exists)
- **Learn:** module 05 (REINFORCE → baseline → GRPO's group-relative advantage, clipping, KL).
  Spinning Up (Achiam), the PPO paper, and the DeepSeekMath paper (GRPO).
- **Build:** run `adaptive_acq/train_grpo.py` (D4 train, AmpC validate, ENHITS report).
- **Evidence:** GRPO vs LinUCB vs agent. This is the test of whether non-myopic credit
  assignment matters on real targets, which module 03 predicts it does only when rounds remain
  *and* exploration is cheap.

### Stage 7. Write-up (week 14)
- Use one results table across every controller, with paired CIs, cost per decision and
  transfer across targets (D4 → AmpC → ENHITS).
- **Rule for claims:** claim only what a table shows. If greedy still wins on ENHITS, the
  write-up says so and makes the case with cost, with the toy `hidden` regime, and with
  knowing *when* to explore.
