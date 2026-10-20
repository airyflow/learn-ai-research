# Glossary

Plain-language meanings of the words used in `learn-al`, grouped by topic. The last column
says where each one is taught, so you can go back to it. Notebook numbers refer to the
[beginner tour](beginner_tour/README.md); **5** is
[classes_and_call_flow](00_how_al_molecular_works/classes_and_call_flow.ipynb), and **P** is
[python_from_zero](00_how_al_molecular_works/python_from_zero.ipynb).

## The problem

| Word | Meaning | Taught in |
|---|---|---|
| **active learning** | a loop: test a few things, learn from the results, use what you learned to choose what to test next, repeat. Here, "test" means dock a molecule | P, 2 |
| **docking** | a computer simulation that fits a molecule into a protein's pocket and estimates how strongly it sticks. Slow: seconds to minutes per molecule | 1 |
| **docking score** | the result of docking, in kcal/mol. **More negative = sticks better.** −12 is excellent, −5 is poor | 1 |
| **pool** (or library) | all the candidate molecules. Here 21,044; in the real project 2.1 million to 1.5 billion | 1 |
| **oracle** | the thing that knows the true answer. Here, a dictionary of pre-computed docking scores, looked up instead of re-docking | 1 |
| **budget** | how many molecules we can afford to dock in total | 2 |
| **round** | one trip around the active-learning loop | 2 |
| **batch** | the molecules docked in one round | 2 |
| **top-k** (the best 100) | the k molecules with the best true scores, the ones we're trying to find | 1, 2 |
| **recall** (top-k recall) | the fraction of the true top-k that we've docked so far. 0.95 = found 95 of the best 100 | 2 |
| **random baseline** | the same budget spent on random molecules. Every method is compared with it | 2 |
| **seed** | the starting point of a random-number generator. Same seed = same "random" choices. Results are averaged over several seeds | 2 |
| **ceiling** | when every method already scores near 100%, so no difference can show | 2 |

## Molecules

| Word | Meaning | Taught in |
|---|---|---|
| **SMILES** | a molecule written as a line of text, e.g. `CCO` (ethanol) | P, 1 |
| **heavy atoms** | all atoms except hydrogen. A simple measure of a molecule's size | 1 |
| **fingerprint** | a long row of 0s and 1s: a checklist of small structural features. Computed by a fixed rule, no AI | 4b |
| **atom-pair fingerprint** | a fingerprint recording pairs of atom types and how many bonds apart they are | 4b |
| **Tanimoto similarity** | how alike two fingerprints are: of the features in *either*, the fraction in *both* | 4b |
| **molecular graph** | a molecule as dots (atoms) joined by lines (bonds) | 4b |

## Describing molecules with AI

| Word | Meaning | Taught in |
|---|---|---|
| **embedding** | a list of numbers describing a molecule, made by an AI model. Similar molecules get similar numbers | 1 |
| **backbone** | a big pretrained AI model that makes embeddings: MoLFormer (reads SMILES text), Uni-Mol2 (reads 3D shape), MHG-GED (reads the graph) | 1 |
| **pretrained** | already trained on millions of molecules before our project starts. Its knowledge comes for free | 1, 4b |
| **cosine similarity** | how alike two embeddings are (1 = pointing the same way). For Uni-Mol2 only the *order* of similarities is meaningful | 1, answers 1 |
| **PCA** | a standard way to squeeze many numbers into fewer while keeping most of the information | 6 |

## Models and learning

| Word | Meaning | Taught in |
|---|---|---|
| **model** | a program that makes a guess from an input; here, a docking score from an embedding | P, 3 |
| **surrogate** | the project's name for that model: a cheap stand-in for docking | 2, 3 |
| **parameter** (weight) | a number inside a model that training adjusts | 3 |
| **training** (fitting) | adjusting the parameters so the guesses match the known answers better | 3 |
| **loss** | a number measuring how wrong the guesses are. Training makes it smaller | 3 |
| **mean squared error (MSE)** | a common loss: the average of (guess − truth)² | 3 |
| **gradient** | which way, and how steeply, the loss changes as each parameter changes ("which way is downhill") | 3 |
| **gradient descent** | training by repeatedly taking a small step downhill | 3 |
| **learning rate** | how big each downhill step is | 3 |
| **epoch** | one pass over all the training examples | 3 |
| **mini-batch** | the small group of examples used for one training step | 3 |
| **optimizer** (Adam, AdamW) | the part of the code that takes the downhill steps | 3 |
| **neural network** | a model made of layers: `Linear` layers mix numbers, `ReLU` adds bends so it can learn curved patterns | 3 |
| **PyTorch** | the Python toolbox the project uses for neural networks | 3 |
| **tensor** | PyTorch's grid of numbers (like a NumPy array, but it can track gradients and live on a GPU) | 3 |
| **train / test split** | training on some examples and judging on others the model never saw. The only honest test | 3 |
| **overfitting** (memorizing) | getting better on the training examples but worse on new ones | 3, answers 3 |
| **weight decay** | a gentle pull of all weights toward zero, to discourage memorizing | 3 |
| **Spearman correlation** | does the model put things in the right *order*? 1 = perfect, 0 = random | 3 |
| **mu (μ)** | the model's guess | 3 |
| **sigma (σ)** | the model's uncertainty: how big its error typically is. **variance** = σ² | 3 |
| **MVE** | mean–variance estimation: a model that outputs both a guess and an uncertainty | 3 |
| **ensemble** | several models trained separately; their average is the guess, their disagreement the uncertainty | 4 |
| **warm start** | continuing training from last round's model instead of starting fresh | 3 |
| **random forest** | many small decision trees whose answers are averaged | 4b |
| **MPN** (message-passing network) | a graph neural network that learns its own description of molecules from their structure | 4b |
| **GPU** | a graphics card. It runs neural networks far faster than a normal processor (**CPU**) | 0, 7 |
| **mixed precision** | using shorter numbers (16-bit) on a GPU, for speed | 3 |

## Choosing what to dock

| Word | Meaning | Taught in |
|---|---|---|
| **acquisition function** | the rule that turns guesses (and uncertainties) into "dock these next" | 4 |
| **greedy** | dock the molecules with the best guesses. Pure exploitation | 4 |
| **UCB** (upper confidence bound) | rank by guess + β × uncertainty: "how good could it be?" | 4 |
| **β** (beta) | how much UCB values uncertainty. β = 0 is greedy | 4, 6 |
| **exploit / explore** | use what you know / try something uncertain that might be better | 4 |
| **rank fusion** (equal vote) | combining several backbones' models by averaging their rankings | 4 |
| **learned weights** (RidgeCV) | combining backbones with weights learned from data | 4 |
| **LT-All** | the project's default model: all embeddings into one network, 5 copies for uncertainty | 4 |
| **controller** | a small decision-maker that picks β each round (the `adaptive_acq` project) | 6 |
| **bandit** (Thompson sampling) | a controller that learns by trying options and keeping score | 6 |
| **hit** | a docked molecule in the estimated top 1%: what a controller is rewarded for | 6 |
| **prefilter / shortlist** | score the whole pool once, then only ever consider the most promising part | 6 |
| **top-weighted loss** | training that counts good molecules more. The project's one reliable improvement | 6 |

## Working at scale

| Word | Meaning | Taught in |
|---|---|---|
| **memory** (RAM) | where a running program keeps its data. A laptop has about 16 GB | 7 |
| **memory map** | treating a file on disk as if it were in memory, reading only the parts you touch | 7 |
| **contiguous vs scattered reads** | reading one stretch of a file vs jumping around in it. Scattered is tens to hundreds of times slower | 7 |
| **view vs copy** | a slice (`a[2:5]`) is a window onto the same data; an index list (`a[[2, 3, 4]]`) makes new data | 7 |
| **chunk** | a piece of a big job done on its own (e.g. 50,000 molecules at a time) | 7 |
| **shard** | a fixed slice of the pool handled by one GPU job | 7 |
| **marker file** | an empty file whose existence signals "I'm done" between jobs | 7 |
| **Slurm**, **job**, `sbatch` | the cluster's scheduler; a job is a script you ask it to run; `sbatch` submits it | 7 |
| **row-keyed** | identifying molecules by row number instead of by SMILES, for billion-molecule pools | 7 |

## The project's code

| Word | Meaning | Taught in |
|---|---|---|
| `run_experiment.py` | the project's main script: one active-learning experiment per run | 2 |
| `main()`, `parse_args()` | the function a command runs, and the one that reads its options | 2, 5 |
| `MVEExplorer` | the class that runs the loop (`run()` is the loop itself) | 2, 5 |
| **run folder** | where an experiment saves its results: `history.json`, `all_explored_final.csv`, `iter_N/` | 2 |
| **checkpoint** | saved progress (`iter_N/`) so a crashed run can continue (`--resume`) | 2, 5 |
| **frozen copy** | `repos/al-molecular-tutorial`: the version of the code this course teaches. Don't edit it | 0 |
| **test** | a small function that checks one thing that must be true | 8 |
| **silent bug** | a bug that gives wrong numbers without crashing. The dangerous kind | 8 |
| **monkeypatch** | temporarily replacing part of the code, for a test | 8 |

## Python

| Word | Meaning | Example | Taught in |
|---|---|---|---|
| **notebook, cell, kernel** | a notebook holds cells; the kernel is the Python running them | Shift + Enter runs a cell | 0, P |
| **variable** | a name for a value | `score = -8.1` | P |
| **string** | text, in quotes | `"CCO"` | P |
| **f-string** | text with values filled in | `f"score {score}"` | P |
| **list** | values in order; counting starts at **0** | `mols[0]` | P |
| **dictionary** (dict) | key → value pairs | `oracle["CCO"]` | P |
| **set** | a collection with no duplicates; fast "is it in here?" | `best & docked` | 2, 4 |
| **`for` loop** | do something for each item | `for s in mols:` | P |
| **`if`** | do something only when a condition is true | `if score < -10:` | P |
| **function** (`def`, `return`) | a named recipe | `def dock(smi): return oracle[smi]` | P |
| **`import`**, module, package | load code someone else wrote | `import numpy as np` | P, 1 |
| **list comprehension** | a one-line loop that builds a list | `[size(s) for s in mols]` | P, 1 |
| **NumPy array** | a fast grid of numbers; `.shape` is its size | `emb.shape` | P, 1 |
| **pandas DataFrame** | a table with named columns | `scores["score"].min()` | 1 |
| **boolean mask** | an array of True/False used to pick items | `y[y < -10]` | 6, 7 |
| **`lambda`** | a short, nameless function | `lambda x: x * 2` | 6 |
| **class, object, method, attribute** | a class bundles data (attributes) with functions (methods); an object is one instance | `model.fit(X, y)` | 5 |
| **`self`** | inside a method, the object it was called on | `self.mean` | 5 |
| **inheritance** | a class built on another, reusing its methods | `class B(A):` | 5, 6 |
| **exception**, `try` / `except` | an error, and catching it so the program continues | `except KeyError:` | 8 |
| **traceback** | the error report; read its **last line** first | `KeyError: 'XYZ'` | 8 |
| **`assert`** | stop loudly if something that must be true isn't | `assert len(batch) == 210` | 8 |
