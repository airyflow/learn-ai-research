# 00 — How al-molecular works (start here)

A beginner's tour of what `al-molecular` does and how its main pieces fit together.
There are four small scripts, and each one runs on a CPU in under a minute. Steps 1–3
use fake data. Step 4 reads results from real runs. Every toy piece comes with a pointer
to the real code that does the same job.

**New to Python?** Start with `python_from_zero.ipynb`. It teaches the Python you need (variables, lists, dictionaries, loops, functions, NumPy) using the same molecule story, with real molecules such as aspirin and caffeine.

**Ready to read the real code?** `classes_and_call_flow.ipynb` (intermediate) teaches
classes, inheritance, composition, factories, `**kwargs`, `@property`, argparse and
checkpointing by running the repo's real `MVEExplorer` on a small fake pool. It prints
the actual call tree of an AL round (helper: `calltrace.py`).

**Looking up a class?** `CLASS_GUIDE.md` explains every class in the repo, layer by layer:
what it holds, what its methods do, who creates and calls it, and whether it's actually used.

**Prefer a notebook?** `walkthrough.ipynb` covers the same material cell by cell, with
plots. Open it in VS Code and pick the `py310` kernel.

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/learn-al
python 00_how_al_molecular_works/step1_the_loop.py
python 00_how_al_molecular_works/step2_greedy_vs_ucb.py
python 00_how_al_molecular_works/step3_real_surrogates.py   # ~1 min on CPU
python 00_how_al_molecular_works/step4_read_real_runs.py
```

---

## 1. The problem, in plain words

Imagine 2 million keys and one lock. Testing a key (docking a molecule against a
protein) is slow and expensive, so you can only try about 1% of them. **Goal: find as
many of the 1,000 best keys as you can within that budget.**

The trick is called **active learning**:

```
        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
  dock a random    train a cheap      score every       pick the best-      dock them,
  handful    ───►  model on all  ───► undocked     ───► looking BATCH  ───► add to the
  (round 0)        docked so far      molecule          (acquisition)       docked set
                   (surrogate)        (predict)                             │
                        ▲                                                   │
                        └───────────────── repeat N rounds ◄────────────────┘
```

At the end you count how many of the true top-1000 you docked. That count is the
**top-1000 recall** shown in every results table in the repo.

## 2. Words you'll see everywhere

| Word | Meaning | In the repo |
|---|---|---|
| **pool** | all candidate molecules (as SMILES strings) | `molpal/libraries/EnamineHTS.csv.gz`, 2.1M |
| **oracle** | the "true" score of each molecule (pre-computed docking scores, looked up instead of re-docked). **Lower = better.** | `data/EnamineHTS_scores.csv.gz` → `load_oracle()` gives `{smiles: score}` |
| **labeled set** | molecules you've "docked" so far | `labeled_idx`, `labeled_scores` in `MVEExplorer` |
| **backbone** | a big pretrained model that turns a molecule into a vector (**embedding**) | MoLFormer (reads text), GROVER (reads the graph), Uni-Mol / Uni-Mol2 (reads the 3D shape), SMI-TED, MHG-GED |
| **embedding** | that vector. Computed **once** for the whole pool and saved as an `(N, D)` file | `embed/`, read by `EmbeddingFeaturizer` |
| **surrogate** | the small model trained each round. It has `fit(X, y)` and `predict(X) → (mu, sigma)` | `surrogates.py` |
| **mu, sigma** | the surrogate's guess, and how unsure it is | |
| **acquisition** | the rule that turns `(mu, sigma)` into "pick these" | `molpal/acquirer/metrics.py`: `greedy` = `mu`, `ucb` = `mu + beta*sigma` |
| **round / batch** | one trip around the loop / how many molecules it docks | `--n-rounds 5`, `--batch-size` = 0.1%, 0.2% or 0.4% of the pool |
| **seed** | a random-number starting point. Results are averaged over 5 seeds | `--seed 42..46` |

## 3. The four steps

| Step | What you learn | What it shows |
|---|---|---|
| **1** `step1_the_loop.py` | The bare loop: init → fit → predict → pick → label | AL finds **94%** of the top-50, random search finds 18%, with the same budget |
| **2** `step2_greedy_vs_ucb.py` | Uncertainty (sigma) and the two acquisition rules, using molpal's real `greedy`/`ucb` | A 4-molecule example where UCB picks a different molecule than greedy, then a 5-seed comparison where the two end up tied |
| **3** `step3_real_surrogates.py` | Backbones and fusion, using the repo's real `SingleBackboneMVESurrogate` and `EnsembleFusionSurrogate` | How rank fusion computes mu/sigma, and why an equal-vote fusion can be pulled down by a weak backbone |
| **4** `step4_read_real_runs.py` | What a real run leaves on disk, and how to rebuild a results table from it | The fixed fusion results: 0.788 / 0.906 / 0.961 recall at 0.1 / 0.2 / 0.4% batches |

Each script ends with `# EXERCISE` lines. Do them. They are the fastest way to get a
feel for the moving parts.

## 4. Toy → real code

Every line of step 1 has a counterpart in [run_experiment.py](../../../al-molecular-tutorial/run_experiment.py):

| Toy (step 1) | Real code |
|---|---|
| `oracle = dict(zip(...))` | `load_oracle()`, [run_experiment.py:302](../../../al-molecular-tutorial/run_experiment.py#L302) |
| `embeddings = ...` | `load_embeddings()` → `EmbeddingFeaturizer.load()`, [run_experiment.py:246](../../../al-molecular-tutorial/run_experiment.py#L246) |
| `true_top_k`, `recall()` | `true_top_k_set()` :457, `MVEExplorer._recall()` |
| random init with `r.choice` | [run_experiment.py:617](../../../al-molecular-tutorial/run_experiment.py#L617) |
| sign flip `-1.0 * scores` | `self._sign = -1.0`, [run_experiment.py:535](../../../al-molecular-tutorial/run_experiment.py#L535), used at :647 |
| `model.fit(...)` | `self.model.train(...)`, [run_experiment.py:649](../../../al-molecular-tutorial/run_experiment.py#L649) |
| `model.predict(...)` on the undocked molecules | `_chunked_get_means_and_vars()`, :657 (predicts the pool in 50k-row chunks to save memory) |
| `np.argsort(mu)[::-1][:BATCH]` | [run_experiment.py:664](../../../al-molecular-tutorial/run_experiment.py#L664) |
| the `print` each round | the `Round 01/05 labeled=... recall=...` lines in every log, plus `iter_N/state.json` |

`--surrogate <name>` is turned into a class by `mve()` in
[molpal/models/\_\_init\_\_.py:109](../../../al-molecular-tutorial/molpal/models/__init__.py#L109).

## 5. The methods being compared

The repo compares several ways to build the surrogate. In order of history:

| `--surrogate` / mode | Class | Idea |
|---|---|---|
| `--mode molpal --model mpn` | (chemprop D-MPNN) | **Baseline [1]** from the MolPAL paper. A graph neural net trained from scratch each round. Accurate but slow |
| `ft_molformer_single` | `SingleBackboneFinetuneScheduleSurrogate` | **[2]** One backbone (MoLFormer), **fine-tuned** each round. Also slow, because the whole pool must be re-embedded |
| `single` | `SingleBackboneMVESurrogate` | One **frozen** backbone plus a small net with an MVE head (predicts mu and sigma) |
| `ensemble` | `EnsembleFusionSurrogate` | **[3] "our fusion"**: one small net per frozen backbone, combined by **ranks** (step 3). Had 3 bugs, fixed 2026-10-10 (see `runs/fusionfix_*`) |
| `learned` | `LearnedFusionSurrogate` | Same idea, but the combination weights are **learned** (RidgeCV) |
| `ltall` | `LTAllSurrogate` | The current default: all backbones glued into one input, learned per-backbone weights, 5-member ensemble for sigma. Used by `adaptive_acq/` |

The "frozen" idea is the key design choice. The expensive backbones run once, up front.
After that, each round only trains tiny nets on stored vectors, so a whole campaign
takes minutes instead of days.

## 6. What the project has found so far

From [README.md](../../../al-molecular-tutorial/README.md),
[reports/figure4_and_fusion_comparison.html](../../../al-molecular-tutorial/reports/figure4_and_fusion_comparison.html) and
[docs/adaptive_acq_results.md](../../../al-molecular-tutorial/docs/adaptive_acq_results.md):

1. **Frozen fusion is nearly as good and far cheaper.** It reaches about 95% recall at the
   largest batch, close to MPN and fine-tuned MoLFormer (about 95–98%), at 100–1000×
   less compute.
2. **The acquisition rule barely matters.** Across 145 runs, no fixed β, β schedule or
   learned β controller beat plain greedy.
3. **What the model can see matters most.** The biggest gains came from fixing bugs
   (+0.058) and from better backbone embeddings (+0.07). The best training tweak, a
   top-weighted loss, added about +0.010 on ENHITS and +0.055 on D4.

## 7. A map of the repo folders

Start with the **bold** ones and skip the rest for now.

| Folder / file | What it's for |
|---|---|
| **`run_experiment.py`** | Entry point. Holds the AL loop (`MVEExplorer`, `MolPALExplorer`) and `main()` |
| **`surrogates.py`** | All surrogate classes |
| **`molpal/`** | The vendored MolPAL library: acquisition metrics, featurizer, MPN model |
| **`runs/`** | Output of every experiment (one folder per run) |
| **`docs/`, `reports/`, `README.md`** | Write-ups of results and bugs |
| `embed/` | Turns SMILES into embeddings with each backbone (heavy GPU jobs, run once per pool) |
| `slurm/` | Scripts that submit runs to the cluster. See `slurm/al_runs/enamine/submit_al_runs.sh` for the exact commands behind each method |
| `adaptive_acq/` | Follow-up project: learning β, more rounds, prefiltering, top-weighted loss |
| `predict_pool_shard_worker*.py`, `shared_embedding_store.py`, `smiles_chunking.py` | Scale-up machinery for the 100M-molecule pools (AmpC, D4). Covered in module 06 |
| `grover/`, `unimol1/`, `unimol2/`, `ibm_materials/` | Backbone model code |

## 8. What a real run looks like

This is the fusion command, copied from `slurm/al_runs/enamine/submit_al_runs.sh`. The
batch size here is 0.1% of the pool:

```bash
python run_experiment.py --mode mve --surrogate ensemble --backbones grover molformer unimol \
    --acq greedy --init-size 2104 --batch-size 2104 --n-rounds 5 --topk 1000 \
    --run-dir runs/my_first_fusion_run
```

It loads several GB of embeddings and wants a GPU, so run it through Slurm or on a GPU
node, not on the login node. `--pool-limit` exists for smoke tests, but the full
embeddings are still loaded before it trims them.

## 9. Where to go next

- **Module 06** (`06_al_molecular_tour/`): the systems side. Memory-mapped embedding
  stores, shard workers, checkpoint/resume, and a case study of the fusion sigma=0 bug.
- Put a `breakpoint()` just after [run_experiment.py:657](../../../al-molecular-tutorial/run_experiment.py#L657)
  and look at `mu`, `var` and `pool_idx` during a real run (module 02 shows how).
