# Beginner tour of al-molecular

A guided tour of the **whole** al-molecular project, for someone who knows **no Python and
no machine learning**. Each notebook runs the project's real code on **real data**: 21,044
real molecules with real docking scores and real embeddings from three AI models. Every new
Python or science idea is explained the first time it appears.

## Before you start (once)

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
python /N/slate/mengjing/repos/learn-ai-research/learn-al/beginner_tour/build_mini_dataset.py
```

The last line builds the small real dataset (about 10 seconds to a minute, 206 MB) into
`learn-ai-research/data/enhits_mini/`. It's real research data, so it is **not** in git. Then
open the notebooks in VS Code, choose the **py310** kernel, and run cells top to bottom with
**Shift + Enter**.

## The notebooks

| # | Notebook | What you'll learn | Part of al-molecular | Runtime |
|---|---|---|---|---|
| 1 | [1_meet_the_data.ipynb](1_meet_the_data.ipynb) | docking scores, SMILES, drawing molecules, embeddings, the repo's folders | `data/`, `embed/` outputs, the folder layout | ~1.5 min |
| 2 | [2_your_first_experiment.ipynb](2_your_first_experiment.ipynb) | the active-learning loop; running the real experiment; its output files; seeds and budgets | `run_experiment.py` (`main`, `MVEExplorer.run`), `runs/` | ~3.5 min |
| 3 | [3_how_the_model_learns.ipynb](3_how_the_model_learns.ipynb) | what "learning" means, PyTorch from zero, train/test, guesses with uncertainty, which backbone predicts best | `surrogates.py`, `molpal/models/losses.py` | ~1 min |
| 4 | [4_choosing_molecules.ipynb](4_choosing_molecules.ipynb) | explore vs exploit (greedy vs UCB); three ways to combine backbones | `molpal/acquirer/metrics.py`, fusion surrogates, LT-All | ~6 min |
| 5 | [../00_how_al_molecular_works/classes_and_call_flow.ipynb](../00_how_al_molecular_works/classes_and_call_flow.ipynb) | classes, inheritance, and the real call tree of one round | how `run_experiment.py`, `molpal` and `surrogates.py` connect | ~1.5 min |
| 6 | [6_smarter_strategies.ipynb](6_smarter_strategies.ipynb) | controllers that change exploration each round; functions as values; writing your own controller | `adaptive_acq/` | ~1.5 min |
| 7 | [7_scaling_up.ipynb](7_scaling_up.ipynb) | memory, memory-mapped files, why reading order matters, chunks, shards, Slurm | `molpal/featurizer.py`, `_get_X`, `predict_pool_shard_worker.py`, `slurm/`, `embed/` | ~1.5 min |
| 8 | [8_tests_and_bugs.ipynb](8_tests_and_bugs.ipynb) | silent vs loud bugs; three real bugs reproduced; running and writing tests | `tests/`, the 2026-10-10 fusion fixes | ~1 min |

Notebook 5 is a little more advanced. If it feels heavy, skip to 6 and come back later. For
more practice with basic Python first, try
[../00_how_al_molecular_works/python_from_zero.ipynb](../00_how_al_molecular_works/python_from_zero.ipynb).

## What you'll find out along the way

All on real data:

* With 6% of the molecules docked, active learning finds about 95% of the best 100. Random docking finds about 6%.
* The three backbones differ enormously: MoLFormer's embeddings predict docking scores far worse than Uni-Mol2's or MHG-GED's.
* Exploring (UCB) doesn't beat simply trusting the model (greedy), here or in the project's 145-run study.
* Combining backbones by equal vote is dragged down by a weak one. Learned combinations figure out which to trust.
* Reading 500 scattered rows from a 6.5 GB file is tens to hundreds of times slower than reading 500 rows in a row.
* Three silent bugs once cost the fusion model up to about 6 points of recall. Tests now guard against them.

## Files

| File | Job |
|---|---|
| `tour.py` | small helpers every notebook imports: data paths, `load_mini()`, `draw()`, `register_mini()` |
| `build_mini_dataset.py` | builds the real mini dataset (run once) |
