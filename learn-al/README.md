# learn-al: Python, debugging, PyTorch, memory and distributed training, all through active learning

This is for someone who already programs in another language. Every module is a
small runnable script built around the same task, an **active-learning loop over
a molecule pool**, so each concept maps directly onto `al-molecular`. Read the
comments, run the script, then do the `# EXERCISE` lines at the bottom.

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/learn-al
mkdir -p logs        # Slurm output goes here
```

| # | Script | Concepts | AL piece | Run |
|---|--------|----------|----------|-----|
| ★ | `beginner_tour/README.md` + 7 notebooks | **for complete beginners:** Python, PyTorch and active learning from zero, on REAL data | a tour of the whole al-molecular repo, running its real code | open `beginner_tour/1_meet_the_data.ipynb` |
| 00 | `00_how_al_molecular_works/README.md` + 4 scripts | the big picture: pool, oracle, backbones, surrogate, greedy vs UCB, rank fusion, reading `runs/` | the whole al-molecular loop on toy data, then the repo's real surrogates and real results | `python 00_how_al_molecular_works/step1_the_loop.py` (then 2, 3, 4) |
| 01 | `01_python/idioms.py` | truthiness, slicing, comprehensions, unpacking, dict/set/Counter, `*args/**kwargs`, dataclasses, context managers, generators, the mutable-default trap | pool/labeled bookkeeping with sets and dicts | `python 01_python/idioms.py` |
| 01 | `01_python/modules_and_imports.md` | `import`, packages, `sys.path`, `__main__`, `python -m` | the knowledge you need to reorganize al-molecular | read |
| 02 | `02_debugging/buggy_al_loop.py` | tracebacks, pdb, `breakpoint()`, post-mortem, VS Code debugger, `logging`, `assert` | a pure-Python AL loop with 4 bugs (two of them silently break the experiment instead of crashing) | `python 02_debugging/buggy_al_loop.py` |
| 03 | `03_pytorch/al_loop.py` | tensors, `nn.Module`, Dataset/DataLoader, training loop, `eval()`/`no_grad` | ensemble surrogate on fingerprints, greedy/UCB/random acquisition, "% of top-1% found" | `python 03_pytorch/al_loop.py --acq ucb` |
| 04 | `04_memory/memory_probe.py` | tensor dtypes, `tracemalloc`, CUDA peak memory, `no_grad`, bf16 autocast, gradient checkpointing, `memory_summary` | pool storage, chunked pool scoring, surrogate training memory | `python …` (CPU part), `sbatch 04_memory/run_gpu.sh` |
| 05 | `05_distributed/ddp_al_iteration.py` | ranks, `torchrun`, process groups, `DistributedSampler`, DDP, `all_reduce`/`all_gather`, rank-0 I/O | DDP surrogate training, then pool scoring sharded across GPUs with a global top-k | `torchrun --nproc_per_node=2 …` (CPU), `sbatch 05_distributed/run_ddp.sh` (4×H100) |
| 06 | `06_al_molecular_tour/README.md` + 3 scripts | reading a real codebase: config, pandas cleaning, checkpoint/resume, inheritance, memmap, views vs copies, atomic writes, marker-file coordination, checking code against docstrings | a guided tour of 10 places in al-molecular, plus a bug case study | `python 06_al_molecular_tour/<script>.py` |

## Suggested order

New to Python and machine learning? Start with **`beginner_tour/`** (8 notebooks, real data) and come back here afterwards.

0. **Big picture first (day 1).** Run the four scripts in `00_how_al_molecular_works/` and read its README.
1. **Python (week 1–2).** `idioms.py`, then skim the official
   [Python tutorial](https://docs.python.org/3/tutorial/) chapters 3–9 for *how Python
   differs* from languages you know. Read `al-molecular/smiles_chunking.py` line by line.
2. **Debugging (week 2).** Fix `buggy_al_loop.py` with the debugger, not by reading the
   code. Then put a `breakpoint()` in `al-molecular/run_experiment.py` where the pool is
   updated, and step through one AL iteration of a real run.
3. **PyTorch (week 3).** `al_loop.py` with all three acquisition functions. Then
   [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html). Map each
   block of `al_loop.py` onto `al-molecular/surrogates.py`.
4. **Memory (week 4).** `memory_probe.py` on a GPU node, then
   [Understanding CUDA memory usage](https://pytorch.org/docs/stable/torch_cuda_memory.html).
   Estimate the memory of al-molecular's largest pool and embedding store.
5. **Distributed (week 5–6).** `ddp_al_iteration.py` on CPU, then on 4 GPUs. Read the
   [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and, for
   surrogates too large for one GPU (e.g. fine-tuning Uni-Mol/GROVER backbones),
   [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html). Compare with
   `al-molecular/predict_pool_shard_worker.py` and `slurm/`.
6. **Tour of al-molecular (alongside 02–05, then week 7).** Follow `06_al_molecular_tour/README.md`:
   10 stops in the real code, each matched to the module that prepares you for it, plus a bug case study.
7. **Capstone.** Reorganize al-molecular into a package (`al/data`, `al/models`,
   `al/acquisition`, `al/loop`, `scripts/`, `slurm/`), using what you learned in 01 and 02.

## Habits worth building from day one

- Run a script with `python -X dev script.py` while learning; it shows warnings that are normally hidden.
- Use `logging` rather than `print` in anything that runs on Slurm.
- Assert the AL invariants every iteration: pool ∩ labeled = ∅, and |pool| + |labeled| = |library|.
- Set seeds when you're debugging, so the bug happens every time. Always compare against random acquisition.
- Before scaling up, make the smallest possible version work (1 GPU, 1k-molecule pool, 1 iteration).
