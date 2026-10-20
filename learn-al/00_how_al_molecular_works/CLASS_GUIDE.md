# Every class in al-molecular, explained

A reference companion to `classes_and_call_flow.ipynb`. It covers **all 128 class
definitions** (123 distinct names) in al-molecular's own code and its vendored MolPAL
library. Packages that only hold backbone
neural-network internals (`muben/`, `unimol1/`, `unimol2/`, and the `grover/` and
`ibm_materials/` submodules) are summarized once per package in §11. Explaining each of
their attention layers one by one wouldn't help you understand the active-learning system.

Read §0 first. After that, jump to whatever you're reading in the code. Line numbers are
where each class starts (as of 2026-10-20).

**Status tags:**

| Tag | Meaning |
|---|---|
| ★ | Used by the project's main experiments (results in `runs/`, `reports/`, `docs/`) |
| ◆ | Selectable from the command line, but not part of the headline results |
| ○ | Reachable only through a factory name in code (no `run_experiment.py` flag) |
| △ | Scaffolding: defined but not wired into anything yet |
| · | MolPAL library internals that al-molecular does **not** use (its own loop replaces them) |

---

## 0. The map

Every experiment is a stack of four layers. Each layer holds the one below it as an
attribute (composition):

```
LAYER 1  the loop          MVEExplorer / ParallelMVEExplorer / AdaptiveMVEExplorer      (run_experiment.py, adaptive_acq/)
                           MolPALExplorer / ParallelMolPALExplorer                       (MPN baseline)
              │ .model
LAYER 2  the model wrapper EmbeddingMVEModel / SingleBackboneEmbeddingModel              (molpal/models/mvemodels.py)
         (a molpal Model)  MPNTwoOutputModel, RFModel, NNModelTorch, ...                 (molpal/models/*)
              │ .surrogate          (embedding path only)
LAYER 3  the surrogate     SingleBackboneMVESurrogate, EnsembleFusionSurrogate,          (surrogates.py)
                           LTAllSurrogate, ... 13 in total
              │ ._model / ._models / ._surrogates
LAYER 4  building blocks   _DualMVEModel, _MVEHead, _LTAllMLP, CombinedLoss, ...          (surrogates.py, molpal/models/losses.py)
```

Around the stack:

* **Data in:** `EmbeddingFeaturizer` (embeddings) or `Featurizer` (fingerprints), plus the oracle dict.
* **Embeddings made beforehand:** `embed/` scripts, using the vendored backbone packages (§8, §11).
* **Fine-tuning inside the loop:** `BackboneFinetuner` (§6), used by the `ft_*` surrogates.
* **Follow-up project:** `adaptive_acq/` swaps the acquisition step for `Campaign` plus a β `Controller` (§7).

**The one contract that ties it together.** Every surrogate has
`fit(X, y, epochs=..., batch=...)` and `predict(X) -> (mu, sigma)`, where `y` and `mu`
are **sign-flipped docking scores (higher = better)** and `sigma` is a standard
deviation. The wrapper turns sigma into a variance for the acquisition functions.

**Which classes run for a given command:**

| Command (`run_experiment.py ...`) | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| `--mode mve --surrogate single --backbone X` | `MVEExplorer` | `SingleBackboneEmbeddingModel` | `SingleBackboneMVESurrogate` |
| `--mode mve --surrogate ensemble` | `MVEExplorer` | `EmbeddingMVEModel` | `EnsembleFusionSurrogate` (holds one `SingleBackboneMVESurrogate` per backbone) |
| `--mode mve --surrogate learned` | `MVEExplorer` | `EmbeddingMVEModel` | `LearnedFusionSurrogate` |
| `--mode mve --surrogate ltall` | `MVEExplorer` | `EmbeddingMVEModel` | `LTAllSurrogate` |
| `--mode mve --surrogate ft_molformer_single` | `MVEExplorer` | `EmbeddingMVEModel` | `SingleBackboneFinetuneScheduleSurrogate` (+ `BackboneFinetuner`) |
| `--mode mve --surrogate ft_fusion` | `MVEExplorer` | `EmbeddingMVEModel` | `FTFusionSurrogate` (+ 2 × `BackboneFinetuner`) |
| any of the above + `--parallel-predict` | `ParallelMVEExplorer` | same | same |
| `--mode molpal --model mpn --conf-method mve` | `MolPALExplorer` | `MPNTwoOutputModel` (holds `MPNN`) | none (the model is the surrogate) |
| `--mode molpal --model mpn --parallel-predict` | `ParallelMolPALExplorer` | `MPNTwoOutputModel` | none |
| `--mode molpal --model rf` / `nn` | `MolPALExplorer` | `RFModel` / `NNModelTorch` | none |
| `adaptive_acq/run_adaptive.py` | `AdaptiveMVEExplorer` + `Campaign` + a `Controller` | `EmbeddingMVEModel` | `LTAllSurrogate` by default |

---

## 1. The loop: `run_experiment.py`

### `DictObjective` ★ (line 349)

**What:** a tiny "objective": something you call with a list of SMILES to get scores.

* `__init__(oracle, minimize=True)` stores the `{smiles: score}` dict and a sign `c = -1` (we minimize docking scores).
* `__call__(smis)` returns `{smiles: c * score}`, or `None` for molecules with no score. Defining `__call__` is what makes the object callable like a function.
* `path` is a `@property` that returns `None`. It exists only because molpal's `Objective` interface has one.

It mirrors molpal's `LookupObjective` (§10) without needing a file on disk. The explorers
read `self.oracle[...]` directly, so this class is mostly a compatibility shim.

### `MVEExplorer` ★ (line 495)

**What:** *the* active-learning loop for every embedding-based method. Once you understand
this class, you understand the project.

* **Holds:** `pool_smiles` (array), `oracle` (dict), `model` (Layer 2, built by the `mve()` factory), `acq_fn` (`greedy` or `ucb`, from `get_metric`), `labeled_idx` (set of pool positions), `labeled_scores` (`{smiles: score}`), `true_top_k` (the answer key, used **only** for grading), `_sign = -1.0`, `run_dir`.
* **`__init__`** has two branches. On a fresh start it draws `init_size` random molecules with `rng.choice`. When resuming (`resume_scores` given) it rebuilds the labeled set from a checkpoint and reloads earlier rounds' history. It also takes an optional `usable_mask`, used only by the parallel subclass.
* **`run()`** loops over rounds: build `(xs, ys)` from the labeled set with the sign flip → `model.train(xs, ys, epochs, batch)` → mask out labeled molecules → `_chunked_get_means_and_vars(model, pool)` (50k-row chunks; rank-fusion surrogates take a separate path, see `combine()`) → `acq_fn` → `argsort` → take the top `batch_size` → look them up in the oracle → `_recall()` → `_checkpoint()`. After the last round it calls `_save_final()`.
* **Helpers:** `_best()` (lowest score so far), `_recall()` (fraction of `true_top_k` labeled), `_checkpoint(rnd, record)` (writes `iter_N/state.json` + `iter_N/scores.pkl`), `_save_final(history)` (writes `history.json` + `all_explored_final.csv`).
* **Created by:** `main()` when `--mode mve` without `--parallel-predict`.
* **Row-keyed mode** (`--row-keyed`, added 2026-10-19, currently only with `--parallel-predict`): for pools above about 100M molecules. Instead of the `oracle` dict and the `labeled_scores` / `true_top_k` sets, molecules are identified by **row number**. Scores come from a memory-mapped `scores.npy`, and the bookkeeping uses boolean arrays (`labeled_mask`, `topk_mask`) and `argpartition` instead of sorting. A `{smiles: score}` dict for the 1.53B pool would need about 370 GB. Checkpoints then write `iter_N/labeled_idx.npy` instead of `scores.pkl`, and `self.row_keyed` tells the methods which bookkeeping to use.

### `ParallelMVEExplorer(MVEExplorer)` ★ (line 722)

**What:** the same loop for pools too large to score in one process (AmpC 99.5M, D4 116M).
Only the *prediction* step changes. Training stays in-process, because the labeled set is small.

* **Inheritance in action:** it overrides `__init__`, `run` and `_checkpoint`, and inherits `_best`, `_recall` and `_save_final` unchanged.
* **`__init__`** works out which shards to exclude **before** calling `super().__init__()`, because the parent's `__init__` draws the random initial set using `usable_mask`. Changing the mask afterwards would be too late.
* **`run()`** each round: trains the surrogate, saves it to `coord_dir` (`round_{r}_surrogate.pt`), writes a *ready* marker file, then waits (`_wait_for_shard`) for every worker process (`predict_pool_shard_worker.py`, one Slurm array task per shard) to write its predictions and a `.done` marker. It then stitches the predictions together and acquires as usual. All coordination goes through files; there is no network communication.
* **`_get_X_cached`, `_snapshot_emb_cache`, `_train_cached`:** cache the labeled molecules' embedding rows, so training doesn't re-read scattered rows from a 100M-row memory-mapped file every round.
* **`_checkpoint`** extends the parent's version with what a resume needs at this scale.
* **Created by:** `main()` with `--mode mve --parallel-predict --num-shards N --coord-dir D`.
* **Runnable miniature:** `learn-al/06_al_molecular_tour/mini_shard_workers.py`.

### `MolPALExplorer` ★ (line 1191)

**What:** the loop for the **MolPAL baseline** (method [1], Chemprop MPN), plus RF and NN.
It's a separate class, not a subclass of `MVEExplorer`, because there are no embeddings:
the model reads SMILES (MPN) or fingerprints (RF/NN) itself.

* **Holds:** `featurizer = Featurizer(fingerprint="pair", radius=2, length=2048)`, `model` from molpal's `model()` factory (`--model mpn --conf-method mve` gives `MPNTwoOutputModel`), and `acq_fn` plus `needs_var` (whether the acquisition needs variances).
* **`run()`** follows the same steps as `MVEExplorer.run`, except it calls `model.train(xs, ys, featurizer=...)` and `_chunked_predict_molpal(...)`.
* Has its own `_best`, `_recall`, `_checkpoint` and `_save_final`, written the same way. That duplication is a cost of not sharing a base class.
* No resume support (`--resume` is wired only for the MVE branches and the parallel MPN branch).

### `ParallelMolPALExplorer(MolPALExplorer)` ★ (line 1309)

**What:** MPN with sharded pool prediction, via `predict_pool_shard_worker_mpn.py`, over the
same marker-file protocol as `ParallelMVEExplorer`. It exists because predicting AmpC's
98.5M molecules with MPN takes about 47 hours per pass on one GPU.

* Overrides `__init__`, `run` and `_wait_for_shard`. Builds its model with `build_mpn_model()`, so the orchestrator and the workers construct an identical architecture.
* Keeps the full library list plus a `usable_mask` instead of filtering, for the same memory reasons as `ParallelMVEExplorer`.

---

## 2. The model wrapper layer: `molpal/models/`

All Layer-2 classes inherit from one abstract base class.

### `Model(ABC)` ★ (`molpal/models/base.py:15`)

**What:** MolPAL's **interface** for anything that can be trained and predict. You can't
create one directly: `Model(...)` raises `TypeError: Can't instantiate abstract class...`.

* **Abstract (each subclass must define):** `provides` (a `@property` returning a set such as `{"means", "vars"}`), `type_` (e.g. `"mve"`, `"mpn"`), `train(xs, ys, *, featurizer, retrain)`, `get_means(xs)`, `get_means_and_vars(xs)`, `save(path)`, `load(path)`.
* **Concrete (shared):** `__init__(test_batch_size, **kwargs)`, `apply(...)`, and `__call__`, which just calls `apply`, so `model(xs)` works.

### `EmbeddingMVEModel(Model)` ★ (`molpal/models/mvemodels.py:17`)

**What:** the adapter between the loop (which speaks SMILES) and a surrogate (which speaks
NumPy arrays). Every `--mode mve` run uses it.

* **Holds:** `surrogate` (Layer 3), `emb_dict` (`{backbone: (N, D) array}`, often memory-mapped), `smi2idx` (`{smiles: row}`), `embeddings_refreshed` (a flag the fine-tuning surrogates set).
* **`_get_X(xs)`:** SMILES → row numbers → slice each backbone's array → glue side by side (`np.concatenate(axis=1)`). Most of its length is a performance workaround: read one contiguous slice and patch the few mismatched rows (duplicate SMILES), because scattered reads from a single-disk file hung for minutes on AmpC. It also falls back to plain indexing near the end of the pool, which fixed a crash in `fusionfix_*_frac0.004_seed42`.
* **`train(xs, ys, *, epochs=None, batch=None, ...)`:** builds `X` with `_get_X` and calls `surrogate.fit(X, ys, ...)`. It passes `epochs`/`batch` only when they're set, and also passes `labeled_smiles=xs` when `getattr(surrogate, "needs_smiles", False)` (the fine-tuning surrogates).
* **`get_means_and_vars(xs)`:** `mu, sigma = surrogate.predict(_get_X(xs))` → returns `mu, sigma ** 2`, because molpal's acquisition functions expect a variance.
* `get_means`, `apply`; `save`/`load` are no-ops (nothing is saved).
* **Created by:** the `mve()` factory, for every surrogate type except `single`.

### `SingleBackboneEmbeddingModel(EmbeddingMVEModel)` ★ (`mvemodels.py:198`)

**What:** the same wrapper, restricted to **one** backbone. It overrides only `__init__`:
it calls `super().__init__(...)`, checks the backbone exists, then sets
`self.emb_dict = {backbone: emb_dict[backbone]}`, so the inherited `_get_X` concatenates a
single array. Created by `mve()` for `--surrogate single`.

### MolPAL's own models (Layer 2 for `--mode molpal`)

These are both wrapper and surrogate in one: they featurize, train and predict
themselves. The table shows which `--model`/`--conf-method` reaches each one through
molpal's `model()` factory (`molpal/models/__init__.py`).

| Class (file:line) | Tag | Reached by | What it is |
|---|---|---|---|
| `MPNTwoOutputModel(Model)` (`mpnmodels.py:536`) | ★ | `--model mpn --conf-method mve` | **The MolPAL baseline.** Chemprop D-MPNN with a mean-and-variance (MVE) output. Holds an `MPNN` |
| `MPNModel(Model)` (`mpnmodels.py:401`) | ◆ | `--model mpn --conf-method none` | Same network, means only (no uncertainty) |
| `MPNDropoutModel(Model)` (`mpnmodels.py:459`) | ◆ | `--model mpn --conf-method dropout` | Uncertainty from MC dropout (dropout left on at prediction, several passes) |
| `MPNN` (`mpnmodels.py:44`) | ★ | inside the three above | The actual Chemprop wrapper: `make_datasets` (SMILES → `MoleculeDataset`), `train` (PyTorch Lightning `LitMPNN` with early stopping; Ray-based DDP exists but is never enabled here), `predict` |
| `NNModelTorch(Model)` (`nn_torch.py:37`) | ★ | `--model nn` | Feed-forward net on fingerprints, 2×100 ReLU, MC-dropout uncertainty (10 passes). A PyTorch rewrite of MolPAL's Keras NN, because TensorFlow isn't installed. Used for the Figure-4 reproduction |
| `_NNTorch(nn.Module)` (`nn_torch.py:24`) | ★ | inside `NNModelTorch` | The network itself |
| `RFModel(Model)` (`sklmodels.py:33`) | ★ | `--model rf` | scikit-learn random forest. Variance = spread of the individual trees' predictions. Figure-4 reproduction |
| `LightGBMModel(Model)` (`sklmodels.py:138`) | · | molpal factory `"lgbm"` only | Gradient-boosted trees |
| `GPModel(Model)` (`sklmodels.py:232`) | · | `"gp"` only | Gaussian process |
| `RandomModel(Model)` (`random.py:8`) | · | `"random"` only | Random predictions. A baseline in MolPAL's own framework (al-molecular does its random baselines differently) |
| `NN`, `NNModel`, `NNEnsembleModel`, `NNTwoOutputModel`, `NNDropoutModel` (`nnmodels.py`) | · | none; the `nn()` factory returns `NNModelTorch` instead | Original Keras/TensorFlow versions. **Not importable here** (TensorFlow isn't installed) |
| `Transformer`, `TransformerModel`, `TransformerTwoOutputModel` (`transformermodels.py`) | · | `"transformer"` only | MolPAL-side MoLFormer fine-tuning wrapper (submits Slurm jobs). Not used: al-molecular fine-tunes MoLFormer through `BackboneFinetuner` |
| `MolCLR`, `MolCLRModel`, `MolCLRTwoOutputModel` (`molclrmodels.py`) | · | `"molclr"` only | MolCLR graph model wrapper. Not used |

---

## 3. The surrogates: `surrogates.py`

All follow the §0 contract. The first column says how you reach each one.
`mve(surrogate_type=...)` names that aren't `--surrogate` choices are factory-only (○).

| `--surrogate` / factory name | Class | One-line idea |
|---|---|---|
| `single` ★ | `SingleBackboneMVESurrogate` | one frozen backbone → small net with MVE heads |
| `lightweight` ○ | `LightweightMVESurrogate` | all backbones concatenated → one MVE net |
| `bigfusion` ○ | `BigFusionSurrogate` | one net per backbone, Borda-rank vote, sigma = 0 |
| `ensemble` ★ | `EnsembleFusionSurrogate` | one net per backbone, rank vote + rank-disagreement sigma (**method [3]**) |
| `learned` ◆ | `LearnedFusionSurrogate` | one net per backbone, RidgeCV learns the weights |
| `nonlinear` ○ | `NonlinearFusionSurrogate` | one net per backbone, small MLP combines (mu, sigma) |
| `attention` ○ | `AttentionFusionSurrogate` | one net per backbone, per-molecule softmax weights |
| `oof` ○ | `OOFFusionSurrogate` | like `learned`, but K-fold out-of-fold so no labels are held out |
| `ft_molformer` ○ | `LightweightMoLFormerScheduleSurrogate` | concatenated MVE net; MoLFormer fine-tuned after round 3 |
| `ft_molformer_single` ★, `ft_grover` ○, `ft_unimol` ○ | `SingleBackboneFinetuneScheduleSurrogate` | one backbone, fine-tuned from round 3 on (**method [2]**) |
| `ft_fusion` ◆ | `FTFusionSurrogate` | GROVER + MoLFormer fine-tuned, UniMol frozen |
| `ltall` ★ | `LTAllSurrogate` | all backbones, learned per-source weights, one shared MLP, 5-member ensemble (**current default**) |

### `SingleBackboneMVESurrogate` ★ (line 196)

**The basic unit.** Several fusion surrogates are built out of these.

* **`__init__(in_dim, spearman_weight=0.1, lr=3e-4, dropout=0.25)`:** builds `self._model = _DualMVEModel(in_dim)` once and `self._loss_fn = CombinedLoss(...)`.
* **`fit(X, y, epochs=50, batch=256)`:** standardizes `y` (stores `_ym`, `_ys`), then calls `_train_model(...)`. **Warm start:** the same network keeps training from round to round.
* **`predict(X)`:** `_predict_model` → returns `(mu_n * _ys + _ym, sig_n)`.
  * ⚠ `mu` is converted back to score units, but `sigma` stays in **standardized** units. With `--acq ucb` this mixes units, unless the score's standard deviation happens to be about 1. `LTAllSurrogate` rescales both.
* **Module-level helpers it uses:**
  * `_train_model(model, X, y_norm, loss_fn, epochs, batch, lr)`: AdamW, mixed precision (bf16/fp16) on GPU, gradient clipping at 1.0, data uploaded to the GPU once.
  * `_predict_model(model, X)`: eval mode, `no_grad`, batches of 1024, returns `(mu, sqrt(var))`.

### `LightweightMVESurrogate` ○ (line 238)

Same as above, but `X` is all backbones glued together (`in_dim` = sum of dims). It is the
paper's "Lightweight" architecture. Simple and cheap, but one net has to learn from very
different embedding spaces at once.

### `BigFusionSurrogate` ○ (line 282)

* **Holds:** `self._surrogates = {backbone: SingleBackboneMVESurrogate}`, optional per-backbone `weights`.
* **`_split(X)`:** cuts the wide concatenated `X` back into per-backbone blocks, using `dims` (backbone order matters!).
* **`fit`:** trains each backbone's surrogate.
* **`predict_components(X)`:** per-backbone means, shape `(n, n_backbones)`. Pointwise, so it is safe to call chunk by chunk.
* **`combine(comps)`:** Borda count over the **whole** pool: `mu = −(weighted sum of ranks)`, **`sigma = 0`**.
* **`predict = combine(predict_components(X))`.**
* With sigma = 0, UCB equals greedy for this class. That's why method [3] uses `EnsembleFusionSurrogate` instead.

### `EnsembleFusionSurrogate` ★ (line 393): method [3], "our fusion"

* **Holds:** one `SingleBackboneMVESurrogate` per backbone, `beta = 0.2`, default `epochs = 50`, `batch = 256`.
* **`fit`:** `_split`, then trains each backbone's surrogate (prints a timing line per backbone).
* **`n_components`** (`@property`): the number of backbones.
* **`predict_components`:** per-backbone means.
* **`combine(comps)`:**
  1. per backbone, `_desc_ranks` (1 = best, **ties averaged**), divided by n
  2. `mu = −mean(rank)`
  3. `sigma = (beta / 2) · std(rank)`, so molpal's `ucb` (`mu + 2·sigma`) equals exactly `−mean_rank + beta·std_rank`
* **Why `predict_components` and `combine` are separate methods:** ranks must be computed over the whole candidate pool, but prediction happens in 50k chunks. `_chunked_get_means_and_vars` checks `hasattr(surrogate, "combine")`, predicts the components chunk by chunk, then calls `combine` **once**.
* **History:** three bugs fixed on 2026-10-10. sigma was 0 (so UCB = greedy), ranks were computed per chunk, and ties favored the end of the pool. Use `runs/fusionfix_*`, not `runs/fusion_*`. Fixed results on EnamineHTS: 0.788 / 0.906 / 0.961 recall at 0.1 / 0.2 / 0.4% batches. Tested in `tests/test_rank_fusion.py`.

### `LearnedFusionSurrogate` ◆ (line 494)

Combines **raw** per-backbone means with a `sklearn` `RidgeCV` meta-learner:
`mu = w_g·mu_g + w_m·mu_m + w_u·mu_u + b`.

* `fit`: 80/20 split → train the backbone surrogates on 80% → predict the held-out 20% → fit RidgeCV on those honest predictions. The backbone surrogates stay trained on 80%.
* `predict` returns `(mu_meta, zeros)`, so use it with greedy.
* From `reports/figure4_and_fusion_comparison.html` (2026-08-10, before the fusion fix): 96.7% vs `ensemble`'s 95.0% at the largest batch.

### `NonlinearFusionSurrogate` ○ (line 622)

Like `learned`, but the meta-learner is `_FusionMLP` on six features
`[mu_g, sigma_g, mu_m, sigma_m, mu_u, sigma_u]`, so it can learn rules like "trust UniMol
more when MoLFormer is unsure". `_backbone_features` builds that 6-column input.
Returns `(mu, zeros)`.

### `AttentionFusionSurrogate` ○ (line 769)

Same 6 features, but `_AttentionModule` outputs **softmax weights per molecule**, and
`mu = Σ w_k · mu_k`. Returns `(mu, zeros)`.

### `OOFFusionSurrogate` ○ (line 931)

Fixes `learned`'s cost of holding back 20% of the labels: K-fold out-of-fold stacking.
Each molecule gets a prediction from fold models that never saw it. RidgeCV is fit on all
of those, then the main surrogates are retrained on 100% of the labels. Fold models train
for a third of the epochs (`fold_epoch_frac`) to keep the runtime similar.

### `LightweightMoLFormerScheduleSurrogate` ○ (line 1063)

Two phases. For the first `n_lt_rounds` (3) calls to `fit`, it trains a
`LightweightMVESurrogate` (held as `self._lightweight`) on the frozen concatenated embeddings. After that it **fine-tunes MoLFormer** on the labels
(`BackboneFinetuner`), re-embeds the whole pool, writes the new MoLFormer columns into the
shared `emb_dict` **in place**, sets `embeddings_refreshed = True`, and keeps the head
**frozen**. Sets `needs_smiles = True`, so the wrapper also passes SMILES.

### `SingleBackboneFinetuneScheduleSurrogate` ★ (line 1205): method [2]

* For the first `n_ft_delay` (2) calls: an MVE head on frozen embeddings.
* Afterwards: fine-tune the backbone (`BackboneFinetuner`, backbone lr 1e-5, head lr 1e-4), re-extract the **whole pool's** embeddings, update `emb_dict` in place, and continue training the head (warm start).
* `needs_smiles = True`.
* `mve()` hands it and the wrapper the **same** single-backbone dict, so the wrapper's `_get_X` sees the refreshed embeddings. Sharing one mutable object between two classes is deliberate here.
* About 95–98% recall at the largest batch, at 100–1000× the cost of frozen fusion.

### `FTFusionSurrogate` ◆ (line 1358)

GROVER and MoLFormer fine-tuned jointly, **UniMol frozen**. UniMol's conformer cache can't
be rebuilt every round: that would take about 2.4 days for the pool, or about 300 GB of RAM
to hold it. The head is a `_DualMVEModel` on the concatenation. 97.3% at the largest batch
(`reports/figure4_and_fusion_comparison.html`).

### `LTAllSurrogate` ★ (line 1595): the current default

A reproduction of the LT-All paper's fusion: all backbones in **one** network, trained
jointly.

* **`__init__(dims, lr=5e-4, weight_decay=1e-5, dropout=0.2, epochs=200, batch=8192, tau=1.0, M=5, top_weight=0.0)`**
* **`_train_one(...)`:** builds a **fresh** `_LTAllMLP` every call. There is deliberately no warm start: the paper retrains each member independently every round. AdamW, mixed precision, gradient clipping. With `top_weight > 0`, the loss weights each molecule by `exp(top_weight · y_std)`, i.e. "fit the good tail better". That is the project's one *reliable* improvement (`docs/adaptive_acq_results.md` §2).
* **`fit`:** standardizes `y`, trains `M = 5` members, and prints the effective sample size of the weights when `top_weight > 0`.
* **`predict`:** `mu` = mean over the 5 members, `sigma` = their standard deviation, both rescaled to score units.
* No `combine`: each molecule is scored independently, so it never had the per-chunk ranking bug.
* ENHITS, 5 backbones, 0.1% batches: 0.876 recall (greedy).

### The PyTorch building blocks (Layer 4)

All are `nn.Module`s. You define layers in `__init__` and the computation in `forward`.
Calling `module(x)` runs `forward` (via `nn.Module.__call__`).

| Class (line) | Used by | Shape / job |
|---|---|---|
| `_MVEHead` (42) | `_DualMVEModel` | `h → (mu, var)`: two little MLPs, `h_dim → 64 → 1`. `var = softplus(·) + 0.01`. (The docstring says `+ 1e-6`; the code adds `0.01`.) |
| `_LightweightBackbone` (66) | `_DualMVEModel` | `in_dim → 256 → 128` with ReLU, BatchNorm, Dropout |
| `_DualMVEModel` (92) | `SingleBackboneMVESurrogate`, `LightweightMVESurrogate`, `SingleBackboneFinetuneScheduleSurrogate`, `FTFusionSurrogate` | backbone → **two** independent MVE heads → averages their mu and var |
| `_FusionMLP` (605) | `NonlinearFusionSurrogate` | `6 → 32 → 16 → 1`, GELU |
| `_AttentionModule` (741) | `AttentionFusionSurrogate` | `6 → 32 → 16 → 3`, then softmax over the backbones |
| `_SourceWeightedFusion` (1546) | `_LTAllMLP` | learnable softmax weight per backbone; scales each block, then concatenates. With equal starting weights it equals plain concatenation |
| `_LTAllMLP` (1567) | `LTAllSurrogate` | `_SourceWeightedFusion` → `D → 1024 → 512` (LayerNorm, ReLU, Dropout 0.2) → `512 → 256 → 1`. A deterministic mean only, no MVE head |

The leading underscore marks them as internal: nothing outside `surrogates.py` should
build them.

---

## 4. Losses: `molpal/models/losses.py`

### `CombinedLoss(nn.Module)` ★ (line 92)

`loss = mve_loss + spearman_weight · spearman_loss` (weight 0.1). Used by every
`_DualMVEModel` surrogate.

* `mve_loss`: Gaussian negative log-likelihood, `½[log σ² + (μ − y)² / σ²]`. It trains the mean and the variance together. A big σ² excuses a big error, but costs `log σ²`.
* `spearman_loss`: a differentiable rank correlation (via `_soft_rank`). It pushes the *ordering* to be right, which is what top-k recall rewards.

`LTAllSurrogate` doesn't use it (plain or top-weighted MSE).

---

## 5. Featurizers: `molpal/featurizer.py`

### `EmbeddingFeaturizer` ★ (line 127)

A `@dataclass` (fields `embed_dir`, `backbones`, `smiles_source`; `__init__` is generated
for you). Its single method `load()` returns `(emb_dict, pool_smiles)`:

* reads `<backbone>_embeddings.npy` **memory-mapped** when present (O(rows touched) memory), otherwise `.npz`
* takes the SMILES order from the `.npz`, or from `smiles_source` for bare `.npy` files (AmpC, D4)
* checks that all backbones agree on the row order

Called by `run_experiment.load_embeddings()`.

### `Featurizer` ★ (line 30)

A `@dataclass` for fingerprints (`fingerprint="pair"`, `radius=2`, `length=2048`).

* `__post_init__` fixes the MACCS size at 167.
* `__len__` returns the length.
* `__call__(smi)` returns a fingerprint array, or `None` if RDKit can't parse the SMILES.

Used by `MolPALExplorer` for RF/NN, and by the MPN path.

---

## 6. In-loop fine-tuning: `backbone_finetuner.py`

### `BackboneFinetuner` ★ (line 134)

Wraps one pretrained backbone (`"grover" | "unimol" | "molformer"`) with a regression head
so it can be fine-tuned during AL. Used by the three schedule surrogates and `FTFusionSurrogate`.

* **Lazy setup:** `_setup_grover` / `_setup_unimol` / `_setup_molformer` load weights on first use. `_grover_instance` builds the GROVER model.
* **`finetune(labeled_smiles, labeled_scores, n_epochs=10, batch_size=32, lr_backbone=1e-5, lr_head=1e-4)`:** a small backbone learning rate, to avoid wiping out what pretraining learned. It dispatches to `_ft_muben` (GROVER), `_ft_unimol` or `_ft_molformer`.
* **`extract_pool_embeddings(batch_size=64)`:** re-embeds the whole pool in `pool_smiles` order. This is the expensive step. `_emb_grover` / `_emb_unimol` / `_emb_molformer` do the per-backbone forward pass and pooling; `_load_unimol_pool_dataset` loads UniMol's conformers.
* **Monkey-patching:** a module-level `_inject_smiles` replaces MUBen's `Dataset.read_csv` so MUBen trains on *our* SMILES list instead of a CSV file.

### `_MubenConfig` ★ (line 69)

A plain "attribute bag": an object whose only job is to carry the settings that MUBen's
dataset and model constructors read (`model_name`, `data_dir`, `checkpoint_path`, ...).
A tiny class is used instead of a dict because MUBen reads `config.x` (attributes), not
`config["x"]`.

---

## 7. The follow-up project: `adaptive_acq/`

Question: can a small controller that picks UCB's β each round beat a fixed β? (Answer:
no. Greedy, β = 0, is as good as anything, so that axis is closed. See
`docs/adaptive_acq_results.md`.)

### `AdaptiveMVEExplorer(MVEExplorer)` ★ (`explorer.py:22`)

A subclass of the main loop. `__init__(controller, beta_scale, diversity_lambda,
prefilter_mult, top_weight_schedule, ...)` calls `super().__init__(acq="greedy", **kwargs)`.
`acq` is unused, because acquisition moves into `Campaign`.

* **`_fit_predict(labeled_idx, y_hb, cand_idx)`:** adapter. Applies this round's `top_weight` if a schedule was given, trains the inherited `self.model`, predicts with the same `_chunked_get_means_and_vars` as the main loop, and returns `(mu, sigma)` over the candidates. It stops with an error if every sigma is 0 while the controller wants β > 0.
* **`_score(idx)`:** adapter, the oracle lookup.
* **`run()`:** builds a `Campaign` around those two adapters plus the controller, computes the hit threshold, restores the shortlist on resume, and runs.
* **`_on_round`:** per-round logging and checkpointing (also the controller state and the last batch).
* **`_restore`:** reloads the controller and `LastBatch` on `--resume`.
* **Design point:** it replays `MVEExplorer`'s seeded random draw, so the hit threshold always comes from the uniform random initial set, even when resuming.

### `Campaign` ★ (`campaign.py:18`)

One AL campaign, written so that the **simulator and the real 2.1M-molecule run share the
exact same logic**. It knows nothing about molecules: it gets `fit_predict` and `score_fn`
as **functions** (functions as values again).

* `step()` = one round: fit/predict → build the 10-feature context → `controller.plan(...)` → `select_batch` / `select_diverse` → score → `controller.update(...)` with the per-molecule "hit" rewards.
* `run()` loops `step()`.
* `recall()` grades against `true_top_mask`, which is **never** shown to the controller.
* `budget()` (a plain method) is the total the campaign *will* label: init + one batch per round.
* `_candidates()` respects the candidate mask and the shortlist.
* `_set_shortlist()` implements the **prefilter**: score everything once, then keep only the top `prefilter_mult × budget()` molecules. It's a cost lever, not a recall lever: it cuts pool-scoring cost without losing recall. Combined with 20 rounds (`r20pf`) it gave +0.014 recall at 0.71× the compute.

### `LastBatch` ★ (`state.py:35`)

A `@dataclass` that remembers what last round predicted for the molecules it picked
(`mu`, `sigma`), what docking returned (`y`), the reward, β, and the best-k mean before.
Next round turns it into feedback features (`last_spearman`, `last_sigma_err_corr`).

### The β controllers (`controllers.py`)

All share one small interface. A controller's `plan(ctx, rnd, n_rounds, batch_size)`
returns a list of `(arm, beta, count)` entries whose counts add up to the batch size, and
`update(...)` learns from the hits.

| Class (line) | Idea |
|---|---|
| `Controller` (33) | Base class. `plan` raises `NotImplementedError`, so subclasses must override it (a home-made interface, not an `ABC`). Default no-op `update`, `state_dict`/`load_state_dict` (for checkpoints), `describe` (for logs). Class attributes `name`, `arms` |
| `FixedBeta(Controller)` (56) | The same β every round. `FixedBeta(0)` is greedy, the control arm |
| `Schedule(Controller)` (70) | Hand-designed β over rounds: `linear_decay`, `linear_increase`, `alternating`, `explore_then_exploit`, `pulse` (high β for one round only, to measure the most that adapting could be worth). `beta_at(rnd, n_rounds)` |
| `BatchSplitThompson(Controller)` (109) | Splits each batch across all β arms according to Thompson-sampling estimates of each arm's hit rate (`prob_best`), with a 5% floor per arm. It discounts old evidence and caps each round's evidence at 50 pseudo-observations, because molecules in one batch aren't independent |
| `LinUCB(Controller)` (172) | Contextual bandit: one linear model per arm on the 10 campaign features. Meant to be fit offline in the simulator (`train_linucb.py`) and loaded for the real run; `save` writes JSON |
| `PolicyController(Controller)` (230) | Wraps `BetaPolicy`. `mode="sample"` draws actions and records `(ctx, action, logp)` for GRPO training; `mode="greedy"` takes the most likely action |

### `BetaPolicy(nn.Module)` ★ (`policy.py:18`)

A tiny MLP, `10 → 64 → 64 → n_arms` (Tanh), returning log-softmax. The last layer starts at
zero, so all arms are equally likely at first. `probs(ctx)` returns NumPy probabilities.
`save` stores the arms and feature names with the weights. `load` is a `@classmethod`:
you call it on the class (`BetaPolicy.load(path)`) and it builds an instance, refusing a
checkpoint trained on a different feature set. Its input is campaign statistics, **never
molecules**. That's the design difference from GLARE-style RL policies.

### Simulator classes (`sim_env.py`)

| Class (line) | Job |
|---|---|
| `SimTask` (50) | One cached simulator task: a PCA-reduced random subsample of a real pool (`X`, `y`, `name`, `source`). `load(path)` is a `@classmethod` constructor. `__len__` is the number of molecules |
| `BootstrapRidge` (124) | Cheap surrogate: M ridge regressions on bootstrap resamples. `__call__(X, labeled_idx, y_hb, cand_idx) -> (mu, sigma)`, the same shape as `Campaign`'s `fit_predict` |
| `MLPEnsemble` (144) | Cheap stand-in for `LTAllSurrogate`: M small MLPs, a fresh start each round, sigma = member spread. Also callable |

Both simulator surrogates are **callable objects** (`__call__`) rather than `fit`/`predict`
classes, because `Campaign` expects one function.

---

## 8. Making embeddings: `backbones/` and `embed/`

### `backbones/base.py`: a unified backbone interface △

Written to collapse 2–3 copies of "forward pass + pooling" per backbone into one. **No
subclasses exist yet** (`GroverBackbone`, `MoLFormerBackbone`, ... are named in the
docstring but have no files), and nothing imports this module. Read it as a design, not as
running code.

| Class (line) | Python feature | Job |
|---|---|---|
| `MoleculeFeatures` (42) | `@dataclass` | One molecule's preprocessed input, opaque outside its backbone |
| `MoleculeBatch` (54) | `@dataclass` | A collated batch, plus `to(device)` |
| `ConformerStore` (72) | `typing.Protocol` | "Anything with `__getitem__(i)` returning `(atoms, coords)`". A Protocol is duck typing written down: no inheritance needed, type checkers verify the shape |
| `Backbone` (82) | `abc.ABC` | Abstract methods `featurize(smi)`, `collate(features)`, `forward_and_pool(batch)`, and `load(...)` (an abstract `@classmethod`). A concrete `resolve_device` (`@staticmethod`) every subclass inherits |
| `ConformerStagedBackbone(Backbone)` (141) | mixin (adds one ability) | Adds `attach_conformer_store(store)`, for UniMol, which needs pre-generated 3D conformers |

`backbones/cli.py` (functions only) is the shared command-line scaffold for the chunked
extraction scripts.

### `embed/`

| Class (file:line) | Job |
|---|---|
| `_GroverConfig` (`embed/compute/compute_grover_embeddings_chunk.py:79`) | Attribute bag for building GROVER (hidden size 128, 4 heads, ...), the same values as `_MubenConfig` |
| `MubenRuntimeConfig` (`embed/extract_embeddings.py:108`) | Attribute bag for MUBen in the original EnamineHTS extraction script |
| `MubenRuntimeConfig` (`embed/compute/generate_grover_embeddings_chunk.py:104`) | A separate copy of the same idea for the chunked GROVER generator. Two classes with the same name in different files is legal (each lives in its own module), but you have to check which one an import refers to |
| `_BlockReader` (`embed/stitch/pca_reduce_embeddings.py:53`) | Reads row ranges of a `.npy` with plain `seek` + `read` instead of a memory-map slice: same bytes, about 2× faster on `/N/project`, and no stalls. `__init__` parses the `.npy` header to find where the data starts; `read(start, end)` returns a NumPy array view of the bytes |

---

## 9. Tests: `tests/test_rank_fusion.py`

### `_FakeModel` (line 24)

A **test double**: just the two attributes `_chunked_get_means_and_vars` touches,
`.surrogate` and `._get_X(xs)`, so the fusion fix can be tested without real embeddings.
Duck typing makes this possible: the function never checks that it got a real
`EmbeddingMVEModel`.

---

## 10. MolPAL internals that al-molecular does not use

al-molecular vendors the whole MolPAL library but replaces its top-level machinery with
the simpler explorers in §1. These classes are only imported within `molpal/` itself.
They're worth knowing so you can ignore them when they come up in a search.

| Class (file:line) | What it is in MolPAL | al-molecular's replacement |
|---|---|---|
| `Explorer` (`molpal/explorer.py:20`) · | MolPAL's full Bayesian-optimization loop (config files, checkpoints, objectives, pools) | `MVEExplorer`, `MolPALExplorer` |
| `Acquirer` (`acquirer/acquirer.py:17`) · | Initial/batch acquisition with heaps, temperature, decay | `get_metric` functions + `argsort` in `run()` |
| `MoleculePool(Sequence)` (`pools/base.py:29`) · | The library as a sequence; fingerprints, clustering, pruning. Inherits `collections.abc.Sequence`, so it supports `len`, iteration, `in` and indexing | a list of SMILES + `emb_dict` |
| `EagerMoleculePool(MoleculePool)` (`pools/base.py:643`) · | An alias (no new methods) | none |
| `LazyMoleculePool(MoleculePool)` (`pools/lazypool.py:12`) · | Computes fingerprints on demand | none |
| `Objective(ABC)` (`objectives/base.py:7`) · | Interface: `__call__` → `forward` | `DictObjective`, or direct oracle lookups |
| `LookupObjective(Objective)` (`objectives/lookup.py:13`) · | Scores looked up from a CSV | `load_oracle()` |
| `AutoName(Enum)` (`utils.py:6`) · | Enum helper (values = member names, `from_str`) | none |
| `InvalidExplorationError`, `IncompatibilityError` (`exceptions.py`) · | Custom exception classes (`class X(Exception): pass`) | none |
| `UncertaintyType(Enum)` (`chemprop/utils.py:32`) · | Chemprop's uncertainty-mode enum | none |

**Used indirectly, through the MPN path** (you rarely need to open these):

| Class | Job |
|---|---|
| `MoleculeDatapoint`, `MoleculeDataset(Dataset)`, `MoleculeSampler(Sampler)`, `MoleculeDataLoader(DataLoader)` (`chemprop/data/data.py`) | Chemprop's data pipeline: SMILES → molecule objects → batches. Subclasses of PyTorch's `Dataset`/`Sampler`/`DataLoader`. Used by `MPNN` and `probe_mpn_training.py` |
| `StandardScaler` (`chemprop/data/scaler.py`) | Target normalization (`fit` / `transform` / `inverse_transform`). Copies also exist in `transformer/utils.py` and `molclr/utils.py` |
| `MolGraph`, `BatchMolGraph` (`chemprop/features/featurization.py`) | One molecule / a batch as atom and bond feature graphs |
| `MPNEncoder(nn.Module)`, `MPN(nn.Module)` (`chemprop/models/mpn.py`) | The message-passing encoder, and a wrapper that featurizes if needed |
| `MoleculeModel(nn.Module)` (`mpnn/model.py:17`) | Encoder + feed-forward head |
| `EvaluationDropout(nn.Dropout)` (`mpnn/model.py:12`) | Dropout that stays on at prediction time (for MC dropout) |
| `LitMPNN(pl.LightningModule)` (`mpnn/ptl/model.py:15`) | The PyTorch Lightning training wrapper `MPNN.train` uses |
| `EpochAndStepProgressBar` (`mpnn/ptl/callbacks.py`) | Lightning progress bar (commented out in `MPNN.train`) |
| `EarlyStoppingCallback`, `PrintingCallback`, `TqdmCallback` (`mpnn/ray/callbacks.py`) | Ray Train callbacks, only for the DDP path, which is never enabled here |
| `NoamLR(_LRScheduler)` (`chemprop/nn_utils.py:145`) | Warm-up-then-decay learning-rate schedule |

**MolPAL's Transformer and MolCLR internals** (all ·, not used): `LitTransformer`, `Net`,
`lm_layer`, `PropertyPredictionDataset`, `InferenceDataset`,
`PropertyPredictionDataModule` (×2, also in molclr), `CheckpointEveryNSteps`,
`ModelCheckpointAtEpochEnd`, `CustomWriter`, `RotateAttentionLayer`, `RotaryEmbedding`,
`RotateEncoderBuilder`, `MolTranBertTokenizer`, `Lamb` (an optimizer), `ListDataset` (×2),
`GINEConv`, `GINet`, `LitMolCLR`, `MolTestDataset`, `MolInferDataset`. They support the
`"transformer"` and `"molclr"` model types from §2, which al-molecular never selects.

---

## 11. Vendored backbone packages (summarized per package)

These hold the big pretrained models' code. You need them only when **making embeddings**
(`embed/`) or **fine-tuning** (`BackboneFinetuner`). The AL loop never touches them.

| Package | Classes | What it is | Used by |
|---|---|---|---|
| `muben/` | 97 | MUBen, a molecular uncertainty benchmark library: GROVER, UniMol, GIN, DNN, TorchMD-Net models, trainers, uncertainty methods (SWAG, BBP, SGLD, evidential, ...) | `BackboneFinetuner` (GROVER/UniMol fine-tuning), `embed/extract_embeddings.py`, the GROVER/UniMol chunk scripts |
| `unimol1/` | 13 | Standalone port of Uni-Mol v1 (conformer data, collation, model) | UniMol extraction |
| `unimol2/` | 14 | Standalone port of Uni-Mol2 (1.1B parameters), embedding path only, verified bit-exact against upstream | `embed/compute/compute_unimol2_embeddings_chunk.py` |
| `grover/` (git submodule) | — | Tencent's original GROVER code | `grover3400`, the "real" 3400-d GROVER extraction |
| `ibm_materials/` (git submodule) | — | IBM's foundation models: MoLFormer/SMI-TED (`smi_ted`), MHG-GED (`mhg_model`), and more | SMI-TED / MHG-GED / MoLFormer extraction |

If you do need to read one, start from its top-level model class and its `forward`. The
rest is standard transformer or graph-network layers.

---

## Python patterns you'll meet, and where

| Pattern | Examples |
|---|---|
| Interface by convention (duck typing) | the surrogate contract; `Campaign`'s `fit_predict`/`score_fn` functions; `_FakeModel` |
| Abstract base class (`ABC`, `@abstractmethod`) | `Model`, `Objective`, `Backbone` |
| `typing.Protocol` | `ConformerStore` |
| `raise NotImplementedError` base class | `Controller` |
| Inheritance + `super()` | `ParallelMVEExplorer`, `SingleBackboneEmbeddingModel`, `AdaptiveMVEExplorer`, every `nn.Module` |
| Composition | Explorer → `.model` → `.surrogate`; fusion surrogates holding per-backbone surrogates; `MPNTwoOutputModel` → `MPNN` |
| Factory functions | `mve()`, `model()`, `mpn()`, `get_metric()` |
| `@dataclass` | `Featurizer`, `EmbeddingFeaturizer`, `LastBatch`, `MoleculeFeatures`, `MoleculeBatch` |
| `@property` | `Model.provides`/`type_`, `n_components`, `DictObjective.path` |
| `@classmethod` constructors | `BetaPolicy.load`, `SimTask.load`, `Backbone.load` |
| `@staticmethod` | `Backbone.resolve_device` |
| `__call__` (callable objects) | `DictObjective`, `Model`, `Featurizer`, `BootstrapRidge`, `MLPEnsemble` |
| Attribute bags | `_MubenConfig`, `_GroverConfig`, `MubenRuntimeConfig` |
| Capability checks (`hasattr`/`getattr`) | `combine` (rank fusion), `needs_smiles`, `embeddings_refreshed` |
| Shared mutable state, on purpose | `emb_dict` updated in place by the fine-tuning surrogates and read by the wrapper |
