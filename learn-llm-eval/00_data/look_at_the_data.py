"""Lesson 0: look at the data before you measure anything.

Run:   python 00_data/look_at_the_data.py      (2 seconds)

IN PLAIN WORDS
Every evaluation starts by LOOKING. Before computing a single accuracy, check:
  1. what the ground truth looks like, and how much of it is missing (not documented)
  2. what one extraction looks like next to the truth, field by field
That's all this script does. It uses fake data shaped like your NeuroPathology project
(common/fake_reports.py), so you can practise freely and share the code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.fake_reports import FIELD_NAMES, extractions, ground_truth, with_truth  # noqa: E402

if __name__ == "__main__":
    gt = ground_truth()
    print(f"{len(gt)} fake reports. First 3 rows of the ground truth (NaN = not documented):\n")
    print(gt.head(3).T.to_string())

    print("\nHow often each variable is NOT documented in the report:")
    print(gt[list(FIELD_NAMES)].isna().mean().sort_values().round(2).to_string())

    ext = extractions(gt, models=("A",), seeds=[0])
    one = with_truth(gt, ext)
    one = one[one["report_id"] == "R000"][["field", "true", "value", "confidence", "outcome"]]
    print("\nReport R000: model A, seed 0, next to the truth:\n")
    print(one.to_string(index=False))

# EXERCISE 1: Which variables are missing most often? For those, what is the easiest mistake
#   for an LLM to make? (Hint: say something when the report says nothing.)
# EXERCISE 2: Change "R000" to a few other ids. Find one row for each outcome:
#   correct, correct_missing, wrong_value, omitted, hallucinated.
#   Write one sentence per outcome in your own words. Lesson 1 counts them.
# EXERCISE 3 (real data, on the cluster only): open one NeuroPathology output JSON and list its
#   top-level keys. Where are the values, and where are the confidence and evidence?
#   (Answer: domain sub-dicts, then field_annotations -> {confidence, evidence, note}.)
