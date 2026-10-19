# Lesson 9: a tour of your real code, `al-molecular/adaptive_acq`

You've learned every idea in this folder already. Now find each one in your own project.

**How to do it, in baby steps:**
1. Run `step1_real_loop_on_toy.py` first (10 s). It runs your real `Campaign` and real
   controllers on the course's toy library. If the `hidden` numbers look like lesson 0's,
   the course and the project really are the same machine.
2. Then visit the stops below, **two or three per evening**. For each stop: open the link,
   read only the lines named, and answer the one question in your `notes.md`.
3. Links work from VS Code's Markdown preview. Line numbers are as of commit `072c1d2`.

```bash
cd /N/slate/mengjing/repos/learn-ai-research/learn-agent-rl
python 09_adaptive_acq_tour/step1_real_loop_on_toy.py
```

**What you should see:**

| controller (your real code) | smooth | hidden |
|---|---|---|
| FixedBeta(0), greedy | 0.984 | 0.657 |
| FixedBeta(0.25) | 0.998 | **0.912** |
| Schedule explore→exploit | 0.955 | 0.854 |
| BatchSplitThompson | 0.972 | 0.884 |

---

## Evening 1: the loop (stops 1–4)

### Stop 1. The UCB score, with a β that means the same thing everywhere
**Lesson:** 0 · **Open:** [acquisition.py:21](../../../al-molecular/adaptive_acq/acquisition.py#L21) `effective_beta`, then [line 43](../../../al-molecular/adaptive_acq/acquisition.py#L43) `ucb`
- Same as `ucb()` in `common/toy_campaign.py`: β is rescaled so that β = 1 means "an average
  uncertainty bonus is worth one std of predicted score".
- [Line 17](../../../al-molecular/adaptive_acq/acquisition.py#L17) `DEFAULT_ARMS`: the comment above
  it records *why* the grid was narrowed to 0–0.5. That's an experiment log in code.

**Question:** why would a β controller trained on one surrogate be useless on another without
this rescaling?

### Stop 2. The observation: what the controller is allowed to see
**Lesson:** 2 · **Open:** [state.py:19](../../../al-molecular/adaptive_acq/state.py#L19) `FEATURE_NAMES`, then [line 89](../../../al-molecular/adaptive_acq/state.py#L89) `featurize`
- The same 10 features as `observe()` in the toy.
- Read the module docstring at the top: everything here uses only what a real campaign knows.

**Question:** which feature would tell a controller "uncertainty is actually informative right
now"? (Hint: it compares σ with real errors on the last batch.)

### Stop 3. The reward, a proxy chosen with care
**Lesson:** 7, step 3 · **Open:** [state.py:70](../../../al-molecular/adaptive_acq/state.py#L70) `HIT_FRAC`, and read the comment block just above it
- The reward is "fraction of the batch above a cutoff frozen from the random initial sample".
- The comment records two reward designs that *failed* on real runs, and why.

**Question:** in lesson 7's words, is this reward a grader or a proxy? What does it fail to
measure? (Lesson 7, step 4.)

### Stop 4. One step of the environment
**Lesson:** 7, step 1 · **Open:** [campaign.py:87](../../../al-molecular/adaptive_acq/campaign.py#L87) `Campaign.step`
Match each line to the toy's `env.step()`:

| campaign.py | what it is |
|---|---|
| [line 90](../../../al-molecular/adaptive_acq/campaign.py#L90) `fit_predict(...)` | retrain the surrogate → μ, σ (the expensive line) |
| `featurize(...)` | build the observation |
| `controller.plan(...)` | the **policy** chooses an action |
| `select_batch(...)` | UCB picks molecules; the controller never does |
| `score_fn(new_idx)` | the oracle (docking) |
| `controller.update(...)` | the controller learns from the reward |

[Line 79](../../../al-molecular/adaptive_acq/campaign.py#L79) `recall()` is computed for the
history but never passed to the controller. That's the grader kept out of the agent's reach.

**Question:** which single line would a "reuse the surrogate this round" action skip? (Your
week-3 agent is built around this answer.)

---

## Evening 2: the controllers (stops 5–7)

### Stop 5. The controller interface
**Lesson:** 1 · **Open:** [controllers.py:33](../../../al-molecular/adaptive_acq/controllers.py#L33) `class Controller`
- `plan()` and `update()` are lesson 1's `select()` and `update()`, but a plan can split one
  batch across several βs.
- [Line 56](../../../al-molecular/adaptive_acq/controllers.py#L56) `FixedBeta` and [line 70](../../../al-molecular/adaptive_acq/controllers.py#L70) `Schedule` are the baselines every learned controller must beat.

**Question:** what does `Schedule("pulse", ...)` do, and why was it needed for the β gate
(stop 9)?

### Stop 6. The bandits
**Lessons:** 1 and 4 · **Open:** [controllers.py:109](../../../al-molecular/adaptive_acq/controllers.py#L109) `BatchSplitThompson`, then [line 172](../../../al-molecular/adaptive_acq/controllers.py#L172) `LinUCB`
- `BatchSplitThompson` is lesson 1 Part B done carefully. `n_eff_cap` stops 2,000 molecules
  from one batch counting as 2,000 independent trials.
- `LinUCB.update` with `counterfactual` is lesson 4's full-information training.

**Question:** both are rewarded by this round's hits. From lesson 1 Part C, what would you
predict they learn on ENHITS?

### Stop 7. The learned policy and GRPO
**Lesson:** 5 · **Open:** [policy.py:18](../../../al-molecular/adaptive_acq/policy.py#L18) `BetaPolicy`, then [train_grpo.py:73](../../../al-molecular/adaptive_acq/train_grpo.py#L73) `grpo_update`
- `BetaPolicy` is the same network as lesson 5's `Policy`, which starts uniform.
- `grpo_update` is lesson 5's `update()`: ratio, clipping, the k3 KL estimate, entropy.

**Question:** in [train_grpo.py:51](../../../al-molecular/adaptive_acq/train_grpo.py#L51) `_rollout`,
what is the return R, and why is it fine to use true recall here but *not* in the controller's
reward?

---

## Evening 3: the experiment around the code (stops 8–10)

### Stop 8. The simulator, an environment that had to match reality
**Lesson:** 7 · **Open:** [sim_env.py:38](../../../al-molecular/adaptive_acq/sim_env.py#L38) (the comment above `SIM_BATCH_FRAC`), then [line 189](../../../al-molecular/adaptive_acq/sim_env.py#L189) `run_episode` and [line 210](../../../al-molecular/adaptive_acq/sim_env.py#L210) `episode_return`
- The comment records a recalibration on Oct 11. With the first settings the simulator's
  recall stayed around 0.03–0.06 while real D4 reaches 0.51, and "every beta looked alike".
  Now greedy's simulated recall curve matches real D4 round by round (0.12/0.26/0.37/0.45/0.51
  vs 0.11/0.27/0.37/0.45/0.51). An environment that doesn't match reality can't teach anything.
- `run_episode` uses the same seed for the same start (lesson 7 step 1: determinism).
- `episode_return` is the grader.

**Question:** if every β gives about the same reward in the simulator, what can a bandit or
GRPO learn there? How did comparing the simulated recall curve with a real one catch the problem?

### Stop 9. The β gate: a decision rule fixed *before* the results
**Lesson:** 8, step 4 · **Open:** [beta_sensitivity.py:1](../../../al-molecular/adaptive_acq/beta_sensitivity.py#L1) (the docstring), [line 27](../../../al-molecular/adaptive_acq/beta_sensitivity.py#L27) `MIN_GAIN`, [line 41](../../../al-molecular/adaptive_acq/beta_sensitivity.py#L41) `paired`
- Paired by seed, with a 95% interval, against a rule written down *before* the 130 runs:
  "a gain of at least +0.005 with CI > 0, then replicate". Result: no-go.
- This is the most research-engineer file in the project. Deciding the bar first is what makes
  a null result trustworthy.

**Question:** why does the docstring expect about one false positive among ~20 cells, and
what does the rule do about it?

### Stop 10. How it plugs into the real 2.1M-molecule run
**Open:** [explorer.py:22](../../../al-molecular/adaptive_acq/explorer.py#L22) `AdaptiveMVEExplorer`, [line 59](../../../al-molecular/adaptive_acq/explorer.py#L59) `_on_round`, [line 76](../../../al-molecular/adaptive_acq/explorer.py#L76) `_restore`
- The same `Campaign` runs on the real pool. `_on_round` writes `controller.json` and
  `last_batch.npz` every round, and `_restore` resumes from them after a crash.
- [run_adaptive.py:33](../../../al-molecular/adaptive_acq/run_adaptive.py#L33) `BACKBONES`: the
  seven embedding models. "Cheap backbone vs LT-All" is a choice among these.

**Question (design, for week 3):** your cost-action agent needs to choose "retrain or reuse"
and "one backbone or all five". `Controller.plan()` only returns `(arm, beta, count)`. Sketch
in 5 lines what you would add to `Campaign.step` and to the plan so the agent can make those
choices. There's no single right answer; bring your sketch to week 3, day 1.
