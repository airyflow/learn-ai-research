# Lesson 9: a tour of your real NeuroPathology code

You learned each idea on fake data. Here is where each one lives in
`/N/slate/mengjing/repos/NeuroPathlogy`. **Most of it you have already built.** The tour
ends with the two gaps and one packaging job, which are weeks 5–6 of your plan.

Line numbers are as of commit `2ad08ef`. Links work from VS Code's Markdown preview.
**Privacy:** read the `.py` code only. Don't copy report text or real values into this course,
your notes on GitHub, or anything public.

Do 3–4 stops per evening. For each: open it, read only the lines named, and answer the
question in `notes.md`.

---

## Evening 1: extraction and the five outcomes

### Stop 1. The schema and the retry loop (the "agent" in your extractor)
[src/primary_extraction/main.py:641](../../../NeuroPathlogy/src/primary_extraction/main.py#L641) `extract()`; Pydantic check at [line 714](../../../NeuroPathlogy/src/primary_extraction/main.py#L714), retry on `ValidationError` at [line 728](../../../NeuroPathlogy/src/primary_extraction/main.py#L728)
- Up to 3 attempts. Invalid JSON is repaired or re-asked; a validation error is sent back to
  the model so it can correct itself. It's the same "validate, then repair" loop as
  learn-agent-rl lesson 6.

**Question:** what happens after the 3rd failed attempt, and is that logged?

### Stop 2. The audit fields: evidence and confidence
[src/primary_extraction/schema1a.py:27](../../../NeuroPathlogy/src/primary_extraction/schema1a.py#L27) `FieldAnnotation`, used at [line 818](../../../NeuroPathlogy/src/primary_extraction/schema1a.py#L818)
- Each variable gets `confidence` (0–1), a verbatim `evidence` quote and a `note`. That's the
  confidence in lesson 3, and the input to the judge in lesson 6.

**Question:** which of these three fields would an LLM judge need, and which could it check
mechanically, without an LLM?

### Stop 3. Scoring against ground truth (lesson 1)
[model_comparison/llm_specific_metrics_full161.py:248](../../../NeuroPathlogy/model_comparison/llm_specific_metrics_full161.py#L248) `sentinel_aware_match`; accuracy and hallucination loop at lines 338–365; **the denominator bug** explained at lines 306–318
- The hallucination rate counts cells where the ground truth is empty but the 5-seed vote
  isn't. Fields with no ground-truth column used to inflate the denominator, and that
  flipped which model looked better: lesson 1 Part D in real life.

**Question:** what does `__NA__` mean, and why must those cells be skipped?

---

## Evening 2: consistency, calibration, uncertainty

### Stop 4. Seeds (lesson 2)
[analysis/analyze_seeds.py:68](../../../NeuroPathlogy/analysis/analyze_seeds.py#L68) `consistency()`; [analysis/compute_confidence_metrics.py:111](../../../NeuroPathlogy/analysis/compute_confidence_metrics.py#L111) `compute_consensus`, [line 135](../../../NeuroPathlogy/analysis/compute_confidence_metrics.py#L135) `seed_agreement_rate`

**Question:** how are ties between seeds handled here, compared with lesson 2's `vote()`?

### Stop 5. Calibration (lesson 3)
[analysis/extra_model_metrics.py:89](../../../NeuroPathlogy/analysis/extra_model_metrics.py#L89) `ece`, [line 69](../../../NeuroPathlogy/analysis/extra_model_metrics.py#L69) `risk_coverage`; the model's own confidence is read at [compute_confidence_metrics.py:99](../../../NeuroPathlogy/analysis/compute_confidence_metrics.py#L99)
- Your ECE uses **seed agreement** as the confidence. Lesson 3 Part C found that agreement is
  the better signal, which is consistent with that choice.

**Question:** what does a risk-coverage curve show that ECE doesn't?

### Stop 6. The bootstrap (lesson 4)
[analysis/build_llm_bootstrap_metrics.py](../../../NeuroPathlogy/analysis/build_llm_bootstrap_metrics.py#L1): docstring at lines 1–22; [`metrics_for_weights` at line 98](../../../NeuroPathlogy/analysis/build_llm_bootstrap_metrics.py#L98); the paired loop at lines 135–144
- The report is the resampling unit, B = 2000, and each draw is reused for all 3 models, so the
  comparisons are paired. That's lesson 4 Part B, already done.

**Question:** why "multinomial" weights instead of `rng.choice` on report ids? (Same result,
faster.)

---

## Evening 3: honest conclusions

### Stop 7. Multiple testing and null controls (lesson 5)
[analysis/rederived_analysis.py:101](../../../NeuroPathlogy/analysis/rederived_analysis.py#L101) `scan` with BH-FDR at [line 111](../../../NeuroPathlogy/analysis/rederived_analysis.py#L111); selection inside each training fold at [line 195](../../../NeuroPathlogy/analysis/rederived_analysis.py#L195); permuted-label null at [analysis/holdout_selection_validation.py:137](../../../NeuroPathlogy/analysis/holdout_selection_validation.py#L137)

**Question:** why must feature selection run *inside* each training fold? What leaks if it
runs once on all the data?

### Stop 8. The cherry-picked folds, documented (lesson 5 Part B)
[BUGFIXES_AND_CORRECTIONS.md, lines 121–143](../../../NeuroPathlogy/BUGFIXES_AND_CORRECTIONS.md#L121)
- "Positive folds" were used as a filter, and the fix reports them only as a diagnostic.

**Question:** write the two-sentence interview version of this bug and its fix.

### Stop 9. Downstream models (lesson 7)
[analysis/downstream_auc_by_model.py:62](../../../NeuroPathlogy/analysis/downstream_auc_by_model.py#L62); single held-out split with a bootstrap CI on the AUC gain at [rederived_analysis.py:152](../../../NeuroPathlogy/analysis/rederived_analysis.py#L152)
- It refuses to run with uncommitted changes (lines 153–156), so every result comes from a
  known version of the code. That's a reproducibility habit worth mentioning in interviews.

**Question:** how does a single pre-registered held-out split avoid the problems of lesson 5?

---

## What to build (weeks 5–6 of your plan)

| Gap | Lesson | Build |
|---|---|---|
| No LLM-as-judge (a grep for "judge" finds nothing) | 6 | A judge using the existing `evidence` quotes; validated on ~40 hand-labeled reports with kappa, recall and a length-bias check |
| `test/` is empty | 8 | `test/test_regression.py`: frozen gold set, 5-seed mean, threshold above the noise floor |
| ~120 one-off scripts in `analysis/` | 1–4 | One package, `npeval`, with one command producing per-field outcomes, consistency, calibration and report-level paired CIs. Reuse your existing functions; don't rewrite them |

The first two are new skills for your resume ("validated LLM-as-judge", "regression evals").
The third turns what you already did into something an interviewer can run.
