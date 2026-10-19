"""A pure-Python active-learning loop with 4 planted bugs. Find each one WITH THE DEBUGGER.

The loop: start with a few labeled molecules, then repeatedly
  (1) "train" a surrogate (here just the mean of the neighbors' scores),
  (2) score the unlabeled pool,
  (3) acquire the top-k by predicted score,
  (4) look up their true labels (the "oracle"), move them pool -> labeled.

Step 1 - read the traceback:
    python 02_debugging/buggy_al_loop.py
  Read it bottom-up: the last line is the error, the line above it is where
  it happened, and the lines higher up are how execution got there.

Step 2 - post-mortem debugging (open a debugger at the crash):
    python -m pdb -c continue 02_debugging/buggy_al_loop.py
  At the (Pdb) prompt:
    p <expr>   print an expression     pp <expr>  pretty-print
    l / ll     list source             w          where am I (stack)
    u / d      move up/down the stack  n / s      next line / step into
    c          continue                b 42       breakpoint at line 42
    interact   full Python REPL with the local variables   q  quit

Step 3 - put `breakpoint()` inside the AL loop and step through ONE iteration.
  Watch len(pool) and len(labeled) change.

Step 4 - the same in VS Code: click left of a line number to add a red dot,
  then Run > Start Debugging > "Python File". Hover variables, use the
  Variables pane, and type expressions in the Debug Console.

Some bugs do NOT crash; they quietly break the AL experiment (the dangerous
kind). Asserts in main() catch them. Answers are at the bottom.
"""
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("al")


def read_csv_lines(n=200, seed=0):
    """Fake screening results file: 'id,feature,score' lines, some scores missing."""
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        x = rng.random()
        score = "n/a" if i % 37 == 0 else f"{(x - 0.7) ** 2 * -10 + rng.gauss(0, 0.05):.3f}"
        lines.append(f"mol{i:03d},{x:.4f},{score}")
    return lines


def load_library():
    """id -> (feature, true score)."""
    lib = {}
    for line in read_csv_lines():
        mol_id, x, score = line.split(",")
        lib[mol_id] = (float(x), float(score))                     # BUG 1
    return lib


def oracle(lib, mol_id):
    """The 'experiment': return the true score."""
    return lib[mol_id][1]


def surrogate_predict(labeled, lib, mol_id, k=5):
    """Predict = mean true score of the k labeled molecules with the closest feature."""
    x = lib[mol_id][0]
    nearest = sorted(labeled, key=lambda m: abs(lib[m][0] - x))[:k]
    return sum(labeled[m] for m in nearest) / len(nearest)


def acquire(pool, preds, batch_size):
    ranked = sorted(pool, key=lambda m: preds[m])                  # BUG 2 (no crash!)
    return ranked[:batch_size]


def run_al(lib, n_init=10, batch_size=10, n_iters=5, seed=0):
    rng = random.Random(seed)
    ids = sorted(lib)
    init = rng.sample(ids, n_init)
    labeled = {m: oracle(lib, m) for m in init}                    # id -> true score
    pool = [m for m in ids if m not in labeled]

    history = []
    for it in range(n_iters):
        preds = {m: surrogate_predict(labeled, lib, m) for m in pool}
        picked = acquire(pool, preds, batch_size)
        for m in picked:
            labeled[m] = oracle(lib, m)
        # BUG 3 is around here
        best = max(labeled.values())
        history = log_iteration(it, best, history)
        log.info("iter %d  labeled=%d  pool=%d  best=%.3f", it, len(labeled), len(pool), best)
    return labeled, pool, history


def log_iteration(it, best, history=[]):                          # BUG 4
    history.append((it, round(best, 3)))
    return history


def main():
    lib = load_library()
    labeled, pool, history = run_al(lib)

    assert len(labeled) + len(pool) == len(lib), "pool/labeled must partition the library"
    assert not set(labeled) & set(pool), "acquired molecules must leave the pool"
    bests = [b for _, b in history]
    assert bests == sorted(bests), "best-so-far can never decrease"
    true_best = max(s for _, s in lib.values())
    log.info("found best %.3f (library best %.3f)", bests[-1], true_best)
    assert bests[-1] > true_best - 0.1, "greedy AL should get close to the best molecule"
    print("All checks passed - now try the EXERCISES.")


if __name__ == "__main__":
    main()

# EXERCISES (after fixing):
# 1. Call log_iteration(0, 1.0) twice in a row and explain the output.
# 2. Replace greedy acquisition with random acquisition. How much worse is it after 5 iters?
# 3. Add an "exploration" bonus: pred + beta * (distance to the nearest labeled molecule).
#    That is the idea behind UCB acquisition in 03_pytorch.
# 4. Open al-molecular/run_experiment.py, find where the pool and labeled set are
#    updated, and set a breakpoint() there.



















# ANSWERS
# 1. float("n/a") -> ValueError. Real screening files have missing labels, so decide
#    on a policy: skip those rows (try/except ValueError: continue) or treat them as failures.
# 2. sorted() is ascending; for maximisation acquisition needs reverse=True
#    (or -preds[m]). This kind of bug "works" but makes AL look worse than random.
# 3. Acquired molecules are never removed from the pool, so the same ones get
#    picked every iteration. Add: picked_set = set(picked); pool = [m for m in pool if m not in picked_set]
# 4. Mutable default argument. It's harmless here because history is passed in, but
#    calling log_iteration without history shares one list across calls. Use history=None.
