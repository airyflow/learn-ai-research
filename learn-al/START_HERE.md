# Start here

This course teaches Python, machine learning and **active learning** through one real
research project, `al-molecular`, which searches millions of molecules for promising drugs.
You don't need to know any of it before you start. This page gives you **one path** through
everything in `learn-al/`, and says what to do when something goes wrong.

Unfamiliar word? Look it up in [GLOSSARY.md](GLOSSARY.md).

---

## The path

Work through the stages in order. Each one ends with a short check: if you can do what it
says, move on. If not, re-read the notebook's summary or look at the worked answers.

### Stage 0: Get set up and learn the very basics (day 1, about 2 hours)

1. **[beginner_tour/0_setup_check.ipynb](beginner_tour/0_setup_check.ipynb)** (about 2 minutes).
   Checks that everything works and builds the data. Fix anything it reports before going on.
2. **[00_how_al_molecular_works/python_from_zero.ipynb](00_how_al_molecular_works/python_from_zero.ipynb)**.
   Python from zero (variables, lists, dictionaries, loops, functions, a first look at NumPy),
   told through a small story about molecules. Do every *Your turn*.

> **Ready when** you can write a `for` loop that prints each key and value of a dictionary,
> and say in one sentence what active learning is.

### Stage 1: The core of the project (week 1, about 5 hours)

The beginner tour's first notebooks. Each runs the project's real code on 21,044 real molecules.

| Notebook | You'll understand |
|---|---|
| [1_meet_the_data](beginner_tour/1_meet_the_data.ipynb) | what the data is: docking scores, SMILES, embeddings |
| [2_your_first_experiment](beginner_tour/2_your_first_experiment.ipynb) | the active-learning loop, by running the real experiment |
| [3_how_the_model_learns](beginner_tour/3_how_the_model_learns.ipynb) | what "training a model" means, from zero |
| [4_choosing_molecules](beginner_tour/4_choosing_molecules.ipynb) | explore vs exploit, and combining several AI models |
| [4b_fingerprints_and_molpal](beginner_tour/4b_fingerprints_and_molpal.ipynb) | the classic approach the project started from (needs a GPU, about 25 min) |

Do the *Try it yourself* exercises. The worked answers are in
[beginner_tour/solutions/](beginner_tour/solutions/).

> **Ready when** you can explain, without notes, why active learning finds far more of the
> best molecules than random docking, and why one weak backbone hurt the "equal vote" model.

### Stage 2: How it's built, and how it scales (week 2, about 5 hours)

| Notebook | You'll understand |
|---|---|
| [5: classes_and_call_flow](00_how_al_molecular_works/classes_and_call_flow.ipynb) | classes, and which function calls which in the real code (the hardest notebook: take it slowly) |
| [6_smarter_strategies](beginner_tour/6_smarter_strategies.ipynb) | the follow-up project's controllers, and functions as values |
| [7_scaling_up](beginner_tour/7_scaling_up.ipynb) | memory, big files, splitting work across many GPUs |
| [8_tests_and_bugs](beginner_tour/8_tests_and_bugs.ipynb) | real bugs, and how tests catch them |

Keep [00_how_al_molecular_works/CLASS_GUIDE.md](00_how_al_molecular_works/CLASS_GUIDE.md)
open while reading the project's code: it explains every class in it.

> **Ready when** you can open `run_experiment.py` in the frozen copy, find `MVEExplorer.run`,
> and say what each block of it does.

### Stage 3: Going deeper (weeks 3 and later)

The numbered modules (01–06) were written for people who already program. After stages
0–2 you're one of them. Recommended order:

1. **[02_debugging](02_debugging/buggy_al_loop.py)**, the most useful next skill: find bugs with
   a debugger instead of by staring at code.
2. **[01_python](01_python/idioms.py)**: the Python habits real code is full of
   (comprehensions, unpacking, `*args`, dataclasses, generators), plus
   [modules_and_imports.md](01_python/modules_and_imports.md).
3. **[03_pytorch](03_pytorch/al_loop.py)**: a complete PyTorch active-learning loop to take apart.
4. **[06_al_molecular_tour](06_al_molecular_tour/README.md)**: ten more stops in the real code.
5. **[04_memory](04_memory/memory_probe.py)** and **[05_distributed](05_distributed/ddp_al_iteration.py)**:
   GPU memory and multi-GPU training, once you need them.

Optional: [00_how_al_molecular_works/walkthrough.ipynb](00_how_al_molecular_works/walkthrough.ipynb)
and its four `step*.py` scripts cover stage 1's ideas again on small fake data. They're a
good quick review.

---

## When something goes wrong

**Read the last line of the error first.** Python prints a long *traceback*, but the last
line names the problem (for example `KeyError: 'XYZ'`). Then find it below.

| What you see | What it means | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or `rdkit`, `pandas`...) | the notebook is using the wrong Python | pick the **py310** kernel (top-right corner of the notebook), then re-run from the top |
| `FileNotFoundError: The mini dataset isn't built yet` | the tour's data hasn't been made | run `beginner_tour/0_setup_check.ipynb`; it builds it |
| `NameError: name 'scores' is not defined` | you ran a cell before the cells it depends on | run the notebook **from the top**, in order |
| The notebook says *"Kernel died"* or *"Kernel restarted"* | Python crashed, usually out of memory | restart the kernel and run from the top. If it repeats, close other notebooks |
| `CUDA out of memory`, or *"no GPU found"* | the graphics card is full, or there isn't one | close other notebooks that use the GPU. Without a GPU, skip the MPN part of notebook 4b |
| A cell shows `[*]` for a long time | it's still working | most cells finish in seconds; the notebook warns about long ones (4b's campaign cell takes ~25 min). The **■ stop** button interrupts it |
| Red text, but the result appeared | warnings from packages, not errors | ignore it |
| Your numbers differ slightly from the text | randomness, or GPU arithmetic | normal; the text describes the pattern |
| `0_setup_check` says *"someone changed files in the frozen copy"* | the project code the tour uses was edited | run the `git ... checkout -- .` command it prints |

### Don't edit the frozen copy

The tour runs the project's code from `repos/al-molecular-tutorial`, a frozen copy, so the
lessons always match the code they quote. When you want to experiment (for example adding a
`breakpoint()` while learning to debug in stage 3), go ahead, then undo it:

```bash
git -C /N/slate/mengjing/repos/al-molecular-tutorial checkout -- .
```

### Asking for help

Include three things: which notebook and cell, the **last line** of the error, and what you
expected to happen. That's usually enough for someone to see the problem straight away.
