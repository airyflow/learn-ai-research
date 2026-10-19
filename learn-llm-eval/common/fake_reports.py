"""A fake neuropathology dataset: ground truth + simulated LLM extractions.

IN PLAIN WORDS
Real NACC reports can't leave the cluster or go into a public repo, so every lesson practises
on this fake version first. It copies the SHAPE of your NeuroPathology project:

  - 160 autopsy "reports", each with 10 variables (ordinal stages, yes/no findings, measurements)
  - some variables are NOT DOCUMENTED in some reports (missing on purpose, like real reports)
  - a fake LLM "extracts" every variable, 5 times (seeds 0-4), with a self-reported confidence
  - two fake models: "A" (stronger, like gpt-oss-20b) and "B" (weaker)
  - a clinical outcome ("dementia") for the downstream-model lessons

The fake LLM makes the same KINDS of mistakes a real one does:
  wrong value      the report says Braak V, the model says IV
  hallucination    the report never mentions TDP-43, the model says "present" anyway
  omission         the report gives the brain weight, the model says "not documented"
  instability      the same report gives different answers on different seeds
  overconfidence   it says confidence 0.9 on answers that are right only ~75% of the time

How the mistakes are made (you don't need this to do the lessons): each (report, variable) has a
hidden "hardness". Harder variables, longer reports and the weaker model B are all harder. Each
seed adds a little noise to the hardness; the model is wrong when hardness + noise crosses a
line. So some items are wrong on every seed, some never, and the borderline ones flip between
seeds -- just like real runs.

    from common.fake_reports import FIELDS, ground_truth, extractions
    gt  = ground_truth()                          # one row per report, NaN = not documented
    ext = extractions(gt, models=("A",), seeds=range(5))   # one row per (model, seed, report, field)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_REPORTS = 160


@dataclass(frozen=True)
class Field:
    name: str
    kind: str            # "ordinal" | "binary" | "category" | "number"
    levels: tuple        # allowed values (ordinal/binary/category); (mean, sd, decimals) for numbers
    p_missing: float     # chance the report does not document it
    difficulty: float    # higher = the LLM gets it wrong more often


FIELDS = (
    Field("braak_stage", "ordinal", tuple(range(7)), 0.05, -0.6),
    Field("thal_phase", "ordinal", tuple(range(6)), 0.10, -0.2),
    Field("cerad_score", "ordinal", tuple(range(4)), 0.10, -0.1),
    Field("lewy_body_type", "category", ("none", "brainstem", "limbic", "neocortical", "amygdala"), 0.15, 0.5),
    Field("infarcts", "binary", (0, 1), 0.05, -0.8),
    Field("hippocampal_sclerosis", "binary", (0, 1), 0.25, 0.1),
    Field("tdp43", "binary", (0, 1), 0.35, 0.6),
    Field("gvd_severity", "ordinal", tuple(range(4)), 0.55, 0.7),
    Field("brain_weight_g", "number", (1150.0, 130.0, 0), 0.08, -0.5),
    Field("cc_body_mm", "number", (5.5, 1.0, 1), 0.60, 0.8),
)
FIELD_NAMES = tuple(f.name for f in FIELDS)
MODEL_OFFSET = {"A": 0.0, "B": 0.45}      # model B finds everything a bit harder
THRESHOLD = 0.9                            # hardness + noise above this = a mistake
SEED_NOISE = 0.35                          # how much answers wobble between seeds


def _rng(*keys):
    """A random generator fixed by the keys, so every run of every lesson is identical."""
    return np.random.default_rng([2026, *[int(k) for k in keys]])


def ground_truth(n=N_REPORTS):
    """One row per report: report_id, length_words, dementia (outcome), and one column per
    field. NaN means the report doesn't document that field."""
    rng = _rng(0)
    ad = rng.normal(size=n)                                     # hidden Alzheimer severity
    lewy = rng.normal(size=n)                                   # hidden Lewy severity
    cols = {
        "report_id": [f"R{i:03d}" for i in range(n)],
        "length_words": np.round(np.exp(rng.normal(7.5, 0.35, n))).astype(int),
        "braak_stage": np.clip(np.round(3 + 1.6 * ad + rng.normal(0, 0.7, n)), 0, 6) + 0.0,
        "thal_phase": np.clip(np.round(2.6 + 1.4 * ad + rng.normal(0, 0.7, n)), 0, 5) + 0.0,
        "cerad_score": np.clip(np.round(1.5 + 0.9 * ad + rng.normal(0, 0.5, n)), 0, 3) + 0.0,  # +0.0: no "-0.0"
        "lewy_body_type": np.array(FIELDS[3].levels)[np.clip(np.round(0.6 + 1.2 * lewy), 0, 4).astype(int)],
        "infarcts": (rng.random(n) < 0.3).astype(float),
        "hippocampal_sclerosis": (rng.random(n) < 0.15 + 0.1 * (ad > 1)).astype(float),
        "tdp43": (rng.random(n) < 0.25 + 0.2 * (ad > 0.5)).astype(float),
        "gvd_severity": np.clip(np.round(1.2 + 0.7 * ad + rng.normal(0, 0.6, n)), 0, 3) + 0.0,
        "brain_weight_g": np.round(1150 - 60 * ad + rng.normal(0, 110, n)),
        "cc_body_mm": np.round(5.5 - 0.5 * ad + rng.normal(0, 0.8, n), 1),
    }
    # Dementia depends on the pathology, plus things no report captures (noise).
    logit = -0.3 + 1.3 * ad + 0.6 * (lewy > 0.8) + 0.5 * cols["hippocampal_sclerosis"] + 0.4 * cols["tdp43"]
    cols["dementia"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    gt = pd.DataFrame(cols)
    for j, f in enumerate(FIELDS):                               # documentation gaps
        miss = _rng(1, j).random(n) < f.p_missing
        gt[f.name] = gt[f.name].astype(object).where(~miss, np.nan)
    return gt


def _wrong_value(f, true, rng):
    if f.kind == "ordinal":
        step = rng.choice([-1, 1])
        lo, hi = min(f.levels), max(f.levels)
        v = true + step
        return float(v if lo <= v <= hi else true - step)
    if f.kind == "binary":
        return float(1 - true)
    if f.kind == "category":
        return rng.choice([x for x in f.levels if x != true])
    mean, sd, dec = f.levels                                      # number: picked the wrong number
    return float(np.round(true * rng.uniform(0.8, 1.2), int(dec)))


def _made_up_value(f, rng):
    if f.kind in ("ordinal", "category"):
        return f.levels[rng.integers(len(f.levels))] if f.kind == "category" else float(rng.choice(f.levels))
    if f.kind == "binary":
        return float(rng.random() < 0.3)
    mean, sd, dec = f.levels
    return float(np.round(rng.normal(mean, sd), int(dec)))


def extractions(gt, models=("A",), seeds=range(5)):
    """One row per (model, seed, report_id, field) with the extracted `value` (NaN = the model
    said "not documented") and the model's self-reported `confidence` (0-1)."""
    z_len = (np.log(gt["length_words"]) - np.log(gt["length_words"]).mean()) / np.log(gt["length_words"]).std()
    rows = []
    for m_i, model in enumerate(models):
        for i, rid in enumerate(gt["report_id"]):
            for j, f in enumerate(FIELDS):
                hard = f.difficulty + MODEL_OFFSET[model] + 0.3 * z_len.iloc[i] + _rng(2, i, j).normal(0, 0.5)
                true = gt.at[i, f.name]
                documented = not (isinstance(true, float) and np.isnan(true))
                for s in seeds:
                    rng = _rng(3, m_i, i, j, s)
                    e = hard + rng.normal(0, SEED_NOISE)
                    if documented:
                        if e <= THRESHOLD:
                            value = true
                        elif rng.random() < 0.3:
                            value = np.nan                        # omission
                        else:
                            value = _wrong_value(f, true, rng)
                    else:
                        value = _made_up_value(f, rng) if e > THRESHOLD + 0.2 else np.nan   # hallucination
                    # Overconfident: confidence drops only a little when the model is wrong.
                    conf = float(np.clip(0.97 - 0.10 * max(e, 0) + rng.normal(0, 0.05), 0.3, 1.0))
                    rows.append((model, s, rid, f.name, value, round(conf, 2)))
    return pd.DataFrame(rows, columns=["model", "seed", "report_id", "field", "value", "confidence"])


def is_missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def outcome(true, pred):
    """Classify one extraction against the ground truth into one of five outcomes."""
    t_miss, p_miss = is_missing(true), is_missing(pred)
    if t_miss and p_miss:
        return "correct_missing"       # report silent, model said "not documented": right
    if t_miss:
        return "hallucinated"          # report silent, model gave a value
    if p_miss:
        return "omitted"               # report documents it, model said "not documented"
    return "correct" if true == pred else "wrong_value"


def with_truth(gt, ext):
    """Join each extraction to its true value and label its outcome."""
    long_gt = gt.melt(id_vars=["report_id"], value_vars=list(FIELD_NAMES), var_name="field", value_name="true")
    df = ext.merge(long_gt, on=["report_id", "field"])
    df["outcome"] = [outcome(t, p) for t, p in zip(df["true"], df["value"])]
    df["is_correct"] = df["outcome"].isin(["correct", "correct_missing"])
    return df


if __name__ == "__main__":
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=("A", "B"), seeds=range(5)))
    print(df.groupby("model")["is_correct"].mean().round(3).to_string())
    print(df.groupby(["model", "outcome"]).size().unstack().to_string())
