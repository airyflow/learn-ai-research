# 06: A guided tour of al-molecular

You've built every piece of this in modules 01–05. Now read how the real
project does it. Each stop gives you **where to look**, **what concept it
teaches** and **questions to answer**. Open the file at that line in VS Code
(links work from the Markdown preview), read for 10–20 minutes, then answer
the questions in your `notes.md`. Line numbers are as of commit `b5fed43`.

Three runnable scripts in this folder copy the trickiest patterns at small scale:

| Script | Copies | Run |
|---|---|---|
| `memmap_store.py` | `smiles_chunking.py`, `shared_embedding_store.py`, numpy views vs copies, memmap | `python 06_al_molecular_tour/memmap_store.py` |
| `mini_shard_workers.py` | `ParallelMVEExplorer` + `predict_pool_shard_worker.py` (coordination through marker files) | `python 06_al_molecular_tour/mini_shard_workers.py [--kill-shard 2]` |
| `fusion_uncertainty.py` | `EnsembleFusionSurrogate` (a case study in docstring vs code) | `python 06_al_molecular_tour/fusion_uncertainty.py` |

Recommended order: stops 1–4 after module 02, stops 5–7 after module 03,
stops 8–10 after modules 04–05, and the case study last.

---

## Stop 1. Configuration without a framework
[run_experiment.py:48](../../../al-molecular/run_experiment.py#L48) `_load_config_env`, and [the DATASETS registry at line 86](../../../al-molecular/run_experiment.py#L86)

**Concepts:** `pathlib.Path`, `str.partition`, `os.environ.setdefault`, module-level code that runs at import time, dict-of-dicts as a registry.

- Why `setdefault` and not `os.environ[key] = value`? Which one wins: `config.env` or an `export` in a Slurm script?
- `_load_config_env(...)` is called at module level (line ~65). What does that mean for `import run_experiment` from another file (stop 9 does this)?
- Adding a new dataset means adding one dict entry. Which functions read `DATASETS[DATASET][...]`?

## Stop 2. Messy real data with pandas
[run_experiment.py:302](../../../al-molecular/run_experiment.py#L302) `load_oracle`

**Concepts:** `pd.read_csv`, normalizing column names, `next(... for ...)` to find the first match, `pd.to_numeric(errors="coerce")`, `dict(zip(...))`.

- This is the real version of bug 1 in `02_debugging`. AmpC has a `"no_score"` value. What would happen downstream if those rows were kept as strings?
- `next(c for c in df.columns if "smiles" in c)`: what happens if no column matches? How would you make the error message friendlier?

## Stop 3. Small classes and Python's "dunder" methods
[run_experiment.py:349](../../../al-molecular/run_experiment.py#L349) `DictObjective`

**Concepts:** `__call__` (an object you can call like a function), `@property`, conditional expressions inside a dict comprehension.

- `self.c = -1.0 if minimize else 1.0`: MolPAL always *maximizes*, so lower-is-better docking scores get flipped. Find the same trick in `MVEExplorer` (`self._sign`).

## Stop 4. Resuming after a crash
[run_experiment.py:461](../../../al-molecular/run_experiment.py#L461) `find_resume_checkpoint` and [`_checkpoint` at line 685](../../../al-molecular/run_experiment.py#L685)

**Concepts:** `Path.glob`, `str.removeprefix`, `try/except ValueError: continue`, `pickle`, `json`.

- A round only counts as checkpointed when **both** `state.json` and `scores.pkl` exist. Why both? What if the job was killed between the two writes?
- Exercise: write `latest_complete_round(run_dir)` from memory, then compare.

## Stop 5. The AL loop itself
[run_experiment.py:496](../../../al-molecular/run_experiment.py#L496) `MVEExplorer.__init__` and [`run()` at line 638](../../../al-molecular/run_experiment.py#L638)

**Concepts:** the whole of module 03, at real scale. `np.random.default_rng(seed)`, `rng.choice(..., replace=False)`, sets for the labeled indices, `np.argsort(scores)[::-1][:k]` for top-k.

- Map every line of `run()` onto `03_pytorch/al_loop.py`: train, mask out labeled, predict, acquire, query the oracle, record.
- `top_local` holds positions in `pool_idx`, not pool indices. Find the line that converts them. (This is the same index-space trap as in `al_loop.py`.)
- Line ~480 builds the mask with a Python `for` loop over `labeled_idx`. Rewrite it as one numpy line. (`mask[list(...)] = False`)
- `true_top_k_set` ([line 457](../../../al-molecular/run_experiment.py#L457)) sorts the **whole** oracle. For AmpC that's 99.5M items. What does `heapq.nsmallest` or `np.argpartition` save?

## Stop 6. Bounding memory during pool prediction
[run_experiment.py:365](../../../al-molecular/run_experiment.py#L365) `_chunked_get_means_and_vars`

**Concepts:** module 04, part B, in production. Then run `memmap_store.py` part 2.

- Read the docstring: three ~24 GB copies came from fancy indexing. Which line of `memmap_store.py` shows the same thing in miniature?

## Stop 7. The surrogate model
[surrogates.py:42–111](../../../al-molecular/surrogates.py#L42) `_MVEHead` / `_DualMVEModel`, [`_train_model` at line 119](../../../al-molecular/surrogates.py#L119), [`_predict_model` at line 172](../../../al-molecular/surrogates.py#L172)

**Concepts:** `nn.Module` composition, predicting a mean **and** a variance (mean–variance estimation, MVE), `F.softplus` to keep the variance positive, mixed precision with `torch.autocast` + `GradScaler`, `clip_grad_norm_`, `torch.randperm` shuffling on the GPU instead of a DataLoader.

- Why `softplus(...) + 0.01` and not `exp(...)`? What breaks if the variance can reach 0?
- `_train_model`'s docstring explains why it avoids a DataLoader. When **is** a DataLoader the right choice? (Hint: data that doesn't fit on the GPU, or expensive per-item processing.)
- The order is `scaler.scale(loss).backward()` → `unscale_` → `clip_grad_norm_` → `step`. Why must `unscale_` come before clipping?

## Stop 8. Inheritance and a subtle ordering bug
[run_experiment.py:743](../../../al-molecular/run_experiment.py#L743) `ParallelMVEExplorer.__init__`

**Concepts:** subclassing, `super().__init__(*args, **kwargs)`, changing `kwargs` before passing them up, `frozenset`.

- The long comment explains why `usable_mask` had to be patched **before** `super().__init__()`. Explain the bug in two sentences. The comment says a test (`verify_exclude_shard.py`) caught it, but that test isn't in the repo. Exercise: write a 15-line version of it yourself (tiny fake pool, exclude shard 1, assert no initial pick falls inside it).

## Stop 9. A distributed system made of files
[predict_pool_shard_worker.py](../../../al-molecular/predict_pool_shard_worker.py) (whole file, 160 lines), [`ParallelMVEExplorer.run` at line 1055](../../../al-molecular/run_experiment.py#L1055), [the Slurm array script](../../../al-molecular/slurm/al_runs/ampc/submit_ampc_al_predict_workers_h100single.sh)

**Concepts:** `import run_experiment as exp` to reuse another script's functions, setting a module global from outside (`exp.DATASET = ...`), polling loops, marker files, Slurm `--array` + `$SLURM_ARRAY_TASK_ID`, `${VAR:?message}` in bash.

- First run `mini_shard_workers.py`, then with `--kill-shard 2`. Then read the real worker.
- Why is `--num-shards` passed explicitly instead of taken from `SLURM_ARRAY_TASK_COUNT`? (The Slurm script's comment tells a real story.)
- Compare with `05_distributed/ddp_al_iteration.py`: same job, done with `all_gather`. When would you choose each?

## Stop 10. Engineering for a big shared filesystem
[`_get_X_cached` at line 857](../../../al-molecular/run_experiment.py#L857), [`_snapshot_emb_cache` at line 997](../../../al-molecular/run_experiment.py#L997), [shared_embedding_store.py](../../../al-molecular/shared_embedding_store.py)

**Concepts:** sequential vs random reads on Lustre, caches that grow and how to avoid keeping 2–3 copies, atomic writes (`write tmp → rename`), defensive `raise ValueError(...)` with useful messages.

- Run `memmap_store.py` parts 3–5 first.
- The comments record real incidents (OOM at 160 GB, 800 GB disk quota exceeded). For each one, name the Python or numpy behavior that caused it.

---

## Case study: does the code do what the docstring says?

Run `fusion_uncertainty.py`, then read [surrogates.py:393–492](../../../al-molecular/surrogates.py#L393) `EnsembleFusionSurrogate`.

The class docstring and the README's description of method [3] say acquisition uses
**inter-backbone rank disagreement** as uncertainty (`-mean_rank + beta * std_rank`).
The `predict()` method returns `(-borda, zeros)`. `self._beta` is stored but never used, and no
std is computed. Every version in git history is the same. Consequences that are
worth checking in your results:

1. **`--acq ucb` = `--acq greedy` for `--surrogate ensemble`.** `ucb(mu, var) = mu + beta * sqrt(var)` and var is 0. Any ensemble UCB vs greedy difference in the results can only come from randomness or run-to-run differences, not from exploration.
2. **The Borda ranks are computed per 50,000-molecule chunk** (`POOL_PREDICT_CHUNK`), not over the whole pool. So every chunk has its own "rank 1". That's harmless if the pool order is random. But `run_experiment.py`'s own comments say the ENHITS rows follow `EnamineHTS_scores.csv.gz`, which is **sorted by docking score**. In the toy version that cuts precision from 43% to 16%. Comparing ENHITS vs ENHITS_shuffled results for the ensemble surrogate should show whether this matters in the real runs.

The skill this teaches: **read the code against its docstring and its claims, and write a tiny
replica to test your reading.** It applies to every codebase, including your own.
