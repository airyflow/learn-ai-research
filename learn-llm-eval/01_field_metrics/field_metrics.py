"""Lesson 1: score every extraction -- and don't let one number hide the mistakes.

Run:   python 01_field_metrics/field_metrics.py     (3 seconds)

IN PLAIN WORDS
Comparing an extraction to the truth gives one of FIVE outcomes, not two:

                           model gave a value        model said "not documented"
  report documents it      correct / WRONG VALUE     OMITTED
  report is silent         HALLUCINATED              correct (correct_missing)

Three different mistakes need three different fixes:
  wrong value  -> better reading / prompting       omitted -> the model missed something
  hallucinated -> the model invented something (the most dangerous kind in clinical data)

Part A: the five outcomes counted.
Part B: accuracy per variable -- the overall number hides weak variables.
Part C: two ways to average (over cells vs over variables), and when they differ.
Part D: hallucination rate, with the denominator stated -- "hallucinations out of WHAT?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import extractions, ground_truth, with_truth  # noqa: E402

if __name__ == "__main__":
    gt = ground_truth()
    df = with_truth(gt, extractions(gt, models=("A",), seeds=[0]))      # one model, one run

    print("Part A: the five outcomes (model A, seed 0)")
    print(df["outcome"].value_counts().to_string(), "\n")

    print("Part B: accuracy per variable")
    per_field = df.groupby("field")["is_correct"].mean().sort_values()
    print(per_field.round(3).to_string(), "\n")

    print("Part C: two averages -- over cells, or over variables?")
    print("  Here every variable has 160 cells (silent reports count too), so the two agree:")
    print(f"    all cells: micro {df['is_correct'].mean():.3f}   macro {per_field.mean():.3f}")
    doc = df[df["true"].notna()]                 # only cells the report DOCUMENTS
    doc_field = doc.groupby("field")["is_correct"].mean()
    print("  Counting only documented cells, variables have different counts, so they split:")
    print(f"    documented cells: micro {doc['is_correct'].mean():.3f}   macro {doc_field.mean():.3f}")
    print("  Micro is pulled toward the often-documented (easy) variables. Say which one you report.\n")

    print("Part D: hallucination rate -- always say what you divided by")
    silent = df[df["true"].isna()]
    print(f"  hallucinated / cells where the report is silent:  "
          f"{(silent['outcome'] == 'hallucinated').mean():.3f}   (n={len(silent)})")
    print(f"  hallucinated / ALL cells:                         "
          f"{(df['outcome'] == 'hallucinated').mean():.3f}   (n={len(df)})")
    print("  Same hallucinations, different denominators, very different numbers.")

# EXERCISE 1: Which variable is worst? Is its problem mostly wrong values, omissions or
#   hallucinations? (Hint: df[df.field == "tdp43"]["outcome"].value_counts())
# EXERCISE 2: Report micro and macro accuracy in one sentence for a paper. Which would a
#   reviewer want, and why might you report both?
# EXERCISE 3: Run with models=("B",). Which outcome grows most for the weaker model?
# EXERCISE 4 (real code): your project hit exactly Part D's problem. Read the comment block at
#   model_comparison/llm_specific_metrics_full161.py lines 306-318: fields with no ground-truth
#   column were inflating the denominator, and it flipped which model looked better.
