# Glossary

Each term is defined in plain words, followed by what it means in your NeuroPathology project.

## Scoring extractions (lessons 0–1)

| Term | Plain meaning | In your project |
|---|---|---|
| **ground truth (GT)** | the trusted correct answer | NACC-coded values for each report |
| **cell** | one (report, variable) pair | one variable in one report |
| **documented / not documented** | whether the report states the value at all | GT present / GT empty |
| **wrong value** | the report states X, the model says Y | Braak V read as IV |
| **omission** | the report states it, the model says "not documented" | a missed brain weight |
| **hallucination** | the report is silent, the model gives a value anyway | TDP-43 "present" with no mention |
| **denominator** | what you divide by; changes the number a lot | hallucinations ÷ silent cells, not ÷ all cells |
| **micro / macro average** | every cell counts once / every variable counts once | macro gives weak variables equal weight |
| **sentinel value** | a special code meaning "not applicable", not a real value | `__NA__` (not on that form version) |

## Runs and confidence (lessons 2–3)

| Term | Plain meaning |
|---|---|
| **seed** | one run of the model; different seeds can give different answers |
| **majority vote** | the most common answer across seeds |
| **seed agreement** | share of seeds giving the vote's answer (5/5, 4/5, ...) |
| **calibration** | stated confidence matches actual accuracy |
| **overconfident** | states higher confidence than it earns |
| **reliability table / diagram** | accuracy per confidence group, compared with the stated confidence |
| **ECE** | expected calibration error: the average gap between stated and actual (0 = perfect) |
| **risk–coverage curve** | error rate if you only keep the most confident X% of answers |
| **AUC** | how well a score separates two groups: 0.5 = useless, 1 = perfect |

## Uncertainty and honest conclusions (lessons 4–5)

| Term | Plain meaning |
|---|---|
| **bootstrap** | resample your data with replacement many times to see how much a number moves |
| **95% CI** | the range the number plausibly falls in; if a difference's CI includes 0, you can't claim it |
| **resampling unit** | what you resample; must be independent: the **report**, not the cell |
| **paired comparison** | compare two models on the same reports, taking the difference per report |
| **multiple testing** | many tests at p < 0.05 means about 5% false positives |
| **FDR / Benjamini–Hochberg** | a correction that controls the share of false findings among your findings |
| **cherry-picking** | choosing the subset of results that looks good (e.g. "positive folds") |
| **null control / permutation test** | rerun everything with shuffled labels; any "effect" left is produced by the pipeline |
| **leakage** | information from the test data sneaking into training or selection |
| **pre-registration** | fixing the analysis and decision rule *before* seeing results |

## Judges, downstream and regression (lessons 6–8)

| Term | Plain meaning |
|---|---|
| **LLM-as-judge** | an LLM that grades another model's output |
| **human agreement** | how often the judge agrees with human labels |
| **Cohen's kappa** | agreement corrected for what you'd get by luck; 0 = luck, 1 = perfect |
| **precision / recall (of a judge)** | of what it flags, how much is really wrong / of what is really wrong, how much it flags |
| **judge bias** | the judge is systematically wrong for some group (e.g. long reports) |
| **error propagation** | how extraction errors change the result of the model that uses the extractions |
| **regression eval** | an automatic check after every change: is it still at least as good? |
| **gold set** | a frozen set of reports with trusted labels, used for regression evals |
| **noise floor** | how much a metric wobbles when nothing changed; thresholds must sit above it |
