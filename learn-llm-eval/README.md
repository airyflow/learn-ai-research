# learn-llm-eval

A hands-on course in **evaluating LLM systems**, taught through your NeuroPathology project:
*an LLM extracts variables from autopsy reports. How do you know it works, and how do you know
your conclusions are real?* It's the third course, after `learn-al` and `learn-agent-rl`, in
the same format: small scripts that run in seconds, each with exercises.

**Start here: [LESSONS.md](LESSONS.md).** Unfamiliar word? See [GLOSSARY.md](GLOSSARY.md).

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/learn-llm-eval
python common/fake_reports.py        # the fake dataset: should print accuracy A 0.842, B 0.680
```

**Everything here uses fake data** ([common/fake_reports.py](common/fake_reports.py)), shaped like the real project
and with the same kinds of LLM mistakes. That means the course can go on GitHub. Real NACC
reports and values must never be copied into it.

| # | Script | The question | Time |
|---|---|---|---|
| 0 | `00_data/look_at_the_data.py` | what does the data look like? | 2 s |
| 1 | `01_field_metrics/field_metrics.py` | right or wrong, and *how* wrong (5 outcomes, denominators) | 3 s |
| 2 | `02_consistency/consistency.py` | same answer twice? (seeds, majority vote) | 3 s |
| 3 | `03_calibration/calibration.py` | is "90% sure" really 90%? (reliability, ECE) | 3 s |
| 4 | `04_bootstrap/bootstrap.py` | how sure is my number? (report-level, paired CIs) | 2 s |
| 5 | `05_multiple_testing/cherry_picking.py` | am I fooling myself? (FDR, positive folds, null control) | 3 s |
| 6 | `06_llm_judge/judge_agreement.py` | can an LLM grade it, and can I trust that grader? (kappa, bias) | 7 s |
| 7 | `07_error_propagation/error_propagation.py` | do extraction errors change the downstream AUC? | 25 s |
| 8 | `08_regression_evals/regression_eval.py` | will I notice when a change makes it worse? | 2 s |
| 9 | `09_neuropath_tour/README.md` | where is each lesson in my real code? | 3 evenings |

## What you'll find

Your NeuroPathology project **already does** lessons 1–5 and 7: report-level paired bootstrap,
ECE from seed agreement, BH-FDR inside training folds, null controls, and a documented fix
for cherry-picked folds. The course gives those skills names and clear explanations, so you
can talk about them in interviews.

It has **two gaps**, which are the build tasks for weeks 5–6 of your plan:
- **An LLM-as-judge** (lesson 6).
- **Regression tests** (lesson 8).

The third task is packaging the scattered analysis scripts into one command.

## How this connects to the job market

In the Bay Area agent/LLM postings checked in October 2026, **evals appeared in 77%**. The
named skills match these lessons: metrics and benchmarks, LLM-as-judge with human agreement,
statistical rigor, and regression evals in CI.
