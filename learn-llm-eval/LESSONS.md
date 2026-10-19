# Lessons, in plain words

The whole course answers **one question**: *my LLM pulls variables out of autopsy reports,
so how do I know whether it works, and whether my conclusions are real?*

Every lesson runs on **fake data** (`common/fake_reports.py`) shaped like your NeuroPathology
project: 160 reports, 10 variables, some not documented, two fake LLMs (A strong, B weak), and
5 runs each. The fake LLM makes the same kinds of mistakes a real one does. Fake data means
you can run everything anywhere and put it on GitHub. Real NACC data never leaves the cluster.

Each lesson ends with an exercise pointing at **where your real project already does this**,
or, for the two gaps (LLM-as-judge, regression tests), what to build.

| Lesson | The question it answers |
|---|---|
| 0 | What does the data look like? |
| 1 | How often is each extraction right, and how is it wrong? |
| 2 | Does it give the same answer twice? |
| 3 | When it says "90% sure", is it right 90% of the time? |
| 4 | How sure am I of my accuracy number? |
| 5 | Am I fooling myself? (many tests, cherry-picked folds) |
| 6 | Can an LLM grade the extractions, and can I trust that grader? |
| 7 | Do extraction errors change the downstream result? |
| 8 | Will I notice if a change makes things worse? |
| 9 | Where does each lesson live in my real project? |

**How to do a lesson** (one evening each):
1. Read the lesson below (5 minutes).
2. Run the command and compare with **"What you should see"**.
3. Do the core exercises.
4. Write 3 sentences in `notes.md`.

Setup, once per terminal:
```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/learn-llm-eval
```

---

## Lesson 0: look before you count (`00_data/look_at_the_data.py`, 2 s)

**The idea.** Before any metric, look at the ground truth and at one extraction next to it.

**What you should see:** a table of 3 fake reports, how often each variable is undocumented
(Braak 5%, corpus callosum 57%), and report R000 field by field. Even this one report has two
wrong values (hippocampal sclerosis, TDP-43), both stated with confidence 0.76–0.85.

**Core exercises:** 1 and 2 (find one example of each outcome).

---

## Lesson 1: right, and how wrong (`01_field_metrics/field_metrics.py`, 3 s)

**The idea.** An extraction has **five** possible outcomes, not two:

| | model gave a value | model said "not documented" |
|---|---|---|
| **report documents it** | correct, or **wrong value** | **omitted** |
| **report is silent** | **hallucinated** | correct |

Each mistake needs a different fix, so count them separately.

**What you should see:**
- Overall accuracy is 0.84, but per variable it ranges from **1.00** (infarcts) down to
  **0.57** (corpus callosum). One overall number hides weak variables.
- **Averages:** over documented cells, "every cell counts once" gives 0.861 and "every variable
  counts once" gives 0.826. Say which one you report.
- **Hallucination rate:** **0.231** if you divide by "cells where the report is silent", or
  **0.049** if you divide by "all cells". It's the same mistakes either way. **Always state
  the denominator.** Your real project hit exactly this (exercise 4).

**Core exercises:** 1, 2 and 4.

---

## Lesson 2: run it again (`02_consistency/consistency.py`, 3 s)

**The idea.** The same report can get different answers on different runs (seeds). That costs
you (one run's accuracy is partly luck) and helps you (disagreement is a free warning sign).

**What you should see:**
- All 5 seeds agree on 100% of `infarcts` but only 36% of `cc_body_mm`.
- One run: 0.83–0.86. **Majority vote of 5 runs: 0.88.**
- **The key table:** when all 5 agree, the answer is right **99.6%** of the time; when only
  2 of 5 agree, **43%**. Disagreement tells you where to look.

**Core exercises:** 1 and 2 (design a "send to a human" rule).

---

## Lesson 3: is "90% sure" really 90%? (`03_calibration/calibration.py`, 3 s)

**The idea.** A confidence is *calibrated* if answers stated at 0.9 are right about 90% of the
time. A reliability table checks this. *ECE* sums up the gap in one number (0 = perfect).

**What you should see:**
- Answers stated at about **0.87 confidence are right only 65%** of the time: overconfident.
  ECE = 0.10.
- **Seed agreement predicts correctness better than the model's own confidence** (AUC 0.96
  vs 0.93). A model grading itself is a weak signal; measure it before trusting it.

**Core exercises:** 1 and 3.

---

## Lesson 4: how sure am I of my number? (`04_bootstrap/bootstrap.py`, 2 s)

**The idea.** The *bootstrap*: resample your data with replacement 2,000 times and see how
much the number moves; the middle 95% is the CI. The one rule: **resample the independent
unit, the report**, not the individual cell.

**What you should see:**
- **Cell CI** (width 0.035) is narrower than the **report CI** (0.042): it pretends to have
  more independent data. The gap is small with this fake data and usually larger with real
  reports.
- **Model A vs B**, difference +0.16: the **paired** CI (width 0.042) is much narrower than
  the unpaired one (0.064).
- **CI width vs reports labeled:** 0.105 at 20 reports, 0.042 at 160. Four times the data
  halves the width, roughly.

**Your real project already does this** (paired, report-level, B = 2000: exercise 4).

**Core exercises:** 1, 2 and 4.

---

## Lesson 5: how to fool yourself (`05_multiple_testing/cherry_picking.py`, 3 s)

**The idea.** Two classic traps, shown on pure noise so any "finding" is false:
- **Many tests:** test 100 useless variables at p < 0.05 and about 5 "pass".
  **Benjamini–Hochberg FDR** correction removes them.
- **Cherry-picked folds:** average a model's gain over only the cross-validation folds
  where it helped, and you get a gain from *any* variable, even noise.
- **Null control:** shuffle the outcome labels and rerun. A "gain" that survives shuffling
  comes from the pipeline, not from biology.

**What you should see:**
- **6 false passes before FDR, 0 after.**
- Noise variable: **−0.005** over all folds, **+0.005** over the "positive folds".
- With shuffled labels: **+0.016** over the "positive folds".

This is the bug you found in a collaborator's analysis. Your repo documents it and its fix
(exercise 4).

**Core exercises:** 1 and 3 (explain it in two interview sentences).

---

## Lesson 6: can an LLM grade the extractions? (`06_llm_judge/judge_agreement.py`, 7 s)

**The idea.** An *LLM-as-judge* reads the report and the extracted value and says
"supported" or "not". It is itself a model, so **validate it against human labels before
using it.** The simulated judge here has two realistic flaws: too lenient with wrong
answers, and more lenient on long reports.

**What you should see:**
- **Raw agreement 0.88**, but *Cohen's kappa* only **0.55**, CI [0.50, 0.61]. Most
  extractions are correct, so a judge that always said "fine" would already agree 84% of
  the time. Kappa corrects for that.
- It **catches 61% of real errors** (recall); 63% of what it flags is really wrong
  (precision).
- **Length bias:** it catches **86%** of errors in short reports but **51%** in long ones.
- **Labels needed:** the kappa CI is 0.33 wide with 20 human-labeled reports, 0.11 with 160.

**This is a gap in your project:** there's no judge yet. Exercise 4 designs one for week 6.

**Core exercises:** 1, 2 and 4.

---

## Lesson 7: do errors matter downstream? (`07_error_propagation/error_propagation.py`, 25 s)

**The idea.** Your variables feed a disease model. Compare the downstream AUC using **true**
values against **extracted** values on the same folds. The gap is what extraction errors cost.

**What you should see:**

| features | AUC | vs true |
|---|---|---|
| true | 0.756 | — |
| LLM, one run | 0.741 | −0.015 |
| LLM, 5-run vote | 0.745 | −0.011 |

Then one field at a time: **hippocampal sclerosis** is 91% accurate but costs the most
(−0.018), because the outcome depends on it. TDP-43 is only 65% accurate but its change is
within noise. **Accuracy alone doesn't tell you which errors matter.** Changes smaller than
about ±0.01 are noise at 160 reports.

**Core exercises:** 1 and 2.

---

## Lesson 8: notice when things get worse (`08_regression_evals/regression_eval.py`, 2 s)

**The idea.** A *regression eval* is an automatic check after every change: a **frozen gold
set**, a **metric** and a **threshold**. The trap: LLM runs wobble, so a threshold smaller
than the wobble raises false alarms.

**What you should see:**
- **Noise floor:** accuracy wobbles by 0.03 between seeds when *nothing* changed.
- **Naive test** (any drop from the accepted run fails): **4 false alarms out of 4.**
- **Better test** (5-seed mean, fail only if it drops by more than 3× the seed std): passes
  the unchanged pipeline and **fails the real regression** (model B, drop 0.15).

**This is a gap in your project:** `test/` is empty. Exercise 4 builds the test for week 5.

**Core exercises:** 1, 2 and 4.

---

## Lesson 9: the tour of your real code ([09_neuropath_tour/README.md](09_neuropath_tour/README.md))

Where every lesson lives in `NeuroPathlogy`, stop by stop, with file and line. You'll see
that you have already built most of this. The tour ends with the two things to build
(lessons 6 and 8) and how to package the rest into one evaluation command.
