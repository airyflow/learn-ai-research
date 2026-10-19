"""Step 4: read the outputs of REAL al-molecular runs (read-only, nothing is changed).

Run:   python 00_how_al_molecular_works/step4_read_real_runs.py      (CPU, instant)

Every run of run_experiment.py writes a folder under al-molecular/runs/:
    runs/<name>/
        iter_1/state.json        one round's summary  (also used by --resume)
        iter_1/scores.pkl        every {smiles: score} docked so far
        ...
        history.json             the list of all rounds' summaries   <- the learning curve
        all_explored_final.csv   every docked molecule, best first
Here we read history.json for the fixed fusion runs (method [3]), 5+ seeds each,
and rebuild the README's results table ourselves.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[3] / "al-molecular" / "runs"

# 1. One run, one round at a time
one = RUNS / "fusionfix_greedy_frac0.001_seed42" / "history.json"
print(f"{one.parent.name}/history.json:")
for rec in json.loads(one.read_text()):
    print(f"  round {rec['round']}  labeled={rec['n_labeled']:>6,}  best={rec['best_score']:6.1f} kcal/mol  "
          f"top-1000 recall={rec['topk_recall']:.1%}  ({rec['elapsed']:.0f}s)")

# 2. All seeds, grouped by (acquisition, batch fraction)
#    folder name looks like: fusionfix_ucb_frac0.002_seed44
groups = defaultdict(list)
for run_dir in sorted(RUNS.glob("fusionfix_*_seed*")):
    hist = run_dir / "history.json"
    if not hist.exists():                      # a run that crashed or is still going
        continue
    history = json.loads(hist.read_text())
    if len(history) < 5:                       # unfinished: don't mix it into the averages
        continue
    _, acq, frac, _seed = run_dir.name.split("_")
    groups[(frac, acq)].append(history[-1]["topk_recall"])

print("\nTop-1000 recall after round 5 (fusion surrogate, fixed version):")
print("  batch        greedy               ucb")
for frac in sorted({f for f, _ in groups}):
    cells = []
    for acq in ("greedy", "ucb"):
        v = np.array(groups.get((frac, acq), []))
        cells.append(f"{v.mean():.3f} +- {v.std():.3f} (n={len(v)})" if len(v) else "-")
    pct = float(frac.removeprefix("frac")) * 100
    print(f"  {pct:.1f}%    {cells[0]:20s} {cells[1]}")

print("""
Compare with the table under "Fusion surrogate bugs fixed" in al-molecular/README.md.
Smaller batch = fewer molecules docked in total = lower recall. Greedy vs UCB:
the gap is at most a few thousandths, tiny next to the effect of batch size.
That (plus 145 more runs in adaptive_acq/) is why the project concluded the
acquisition rule is not where the gains are.""")

# EXERCISE: open runs/<one run>/all_explored_final.csv with pandas and print the
#           10 best molecules (pd.read_csv(...).head(10)). They are real SMILES strings.
