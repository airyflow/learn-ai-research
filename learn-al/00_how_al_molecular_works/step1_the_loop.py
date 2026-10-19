"""Step 1: the active-learning loop, with nothing else in the way.

Run:   python 00_how_al_molecular_works/step1_the_loop.py      (CPU, ~1 second)

The story al-molecular tells, in one paragraph:
    We have a huge list of molecules (the POOL). For each one, a docking
    program could compute a SCORE (lower = binds better). Docking is slow, so we
    pretend we can only afford to dock a few. The goal: find as many of the
    true top-k molecules as possible while docking as few as we can.

    So we dock a small random handful, train a cheap model (the SURROGATE)
    to guess scores, ask it "which undocked molecules look best?", dock
    those, retrain, and repeat. That repeat-loop is active learning.

Everything here is fake and tiny (5,000 molecules instead of 2.1 million), but
the variable names and steps are the same as MVEExplorer.run() in
al-molecular/run_experiment.py.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. The data.  In al-molecular:
#      pool_smiles  <- molpal/libraries/EnamineHTS.csv.gz   (2.1M SMILES strings)
#      oracle       <- data/EnamineHTS_scores.csv.gz        ({smiles: docking score})
#      embeddings   <- one (N, D) array per backbone, e.g. MoLFormer is (N, 768)
# ---------------------------------------------------------------------------
N_POOL = 5_000
pool_smiles = np.array([f"mol_{i}" for i in range(N_POOL)])        # stand-ins for SMILES

# An "embedding" is just a row of numbers that describes a molecule.
embeddings = rng.normal(size=(N_POOL, 8)).astype(np.float32)

# The hidden truth: a docking score, lower = better (like kcal/mol, e.g. -12.7).
# It's not perfectly linear, so a simple model can't get it exactly right.
e = embeddings
true_scores = -8 + 1.2 * e[:, 0] - 0.8 * e[:, 1] + 0.6 * e[:, 2] * e[:, 3] + rng.normal(scale=0.4, size=N_POOL)
oracle = dict(zip(pool_smiles, true_scores))                       # same shape as load_oracle()

# ---------------------------------------------------------------------------
# 2. The metric.  "Of the true top-k molecules, what fraction have we docked?"
#    Same as true_top_k_set() and MVEExplorer._recall() in run_experiment.py.
# ---------------------------------------------------------------------------
TOPK = 50
true_top_k = {s for s, _ in sorted(oracle.items(), key=lambda x: x[1])[:TOPK]}   # lowest scores = best


def recall(labeled_scores):
    return sum(1 for s in labeled_scores if s in true_top_k) / TOPK


# ---------------------------------------------------------------------------
# 3. The surrogate.  A cheap model: input = embedding, output = guessed score.
#    Here it's plain linear regression (least squares).  In al-molecular it's
#    a small neural net (SingleBackboneMVESurrogate in surrogates.py), but it
#    has the same two methods: fit(X, y) and predict(X).
# ---------------------------------------------------------------------------
class LinearSurrogate:
    def fit(self, X, y):
        X1 = np.c_[X, np.ones(len(X))]                  # add a column of 1s for the intercept
        self.w, *_ = np.linalg.lstsq(X1, y, rcond=None)

    def predict(self, X):
        return np.c_[X, np.ones(len(X))] @ self.w


# ---------------------------------------------------------------------------
# 4. The loop.  Read this side by side with MVEExplorer.run().
# ---------------------------------------------------------------------------
INIT, BATCH, N_ROUNDS = 100, 100, 5


def run(strategy, seed=42):
    r = np.random.default_rng(seed)

    # Round 0: dock a random handful.  (MVEExplorer.__init__: rng.choice(...))
    init_idx = r.choice(N_POOL, INIT, replace=False)
    labeled_idx = set(init_idx.tolist())
    labeled_scores = {pool_smiles[i]: oracle[pool_smiles[i]] for i in init_idx}

    model = LinearSurrogate()
    print(f"\n--- {strategy} ---   init: {INIT} random molecules, recall={recall(labeled_scores):.0%}")

    for rnd in range(N_ROUNDS):
        # (a) training data = everything docked so far.
        #     SIGN FLIP: the code multiplies scores by -1 (self._sign = -1.0) so that
        #     "higher prediction = better molecule".  Every acquisition function
        #     then just picks the HIGHEST numbers.
        idx = list(labeled_idx)
        X_train = embeddings[idx]
        y_train = -1.0 * np.array([labeled_scores[pool_smiles[i]] for i in idx])
        model.fit(X_train, y_train)

        # (b) the candidates = every molecule not docked yet
        mask = np.ones(N_POOL, bool)
        mask[idx] = False
        pool_idx = np.where(mask)[0]

        # (c) score the candidates and pick the best BATCH of them
        if strategy == "random":
            selected = r.choice(pool_idx, BATCH, replace=False)
        else:  # "greedy": trust the model's guess
            mu = model.predict(embeddings[pool_idx])
            top_local = np.argsort(mu)[::-1][:BATCH]      # [::-1] reverses: biggest first
            selected = pool_idx[top_local]                # local position -> pool position

        # (d) "dock" them: look up the real score in the oracle
        for i in selected:
            labeled_scores[pool_smiles[i]] = oracle[pool_smiles[i]]
            labeled_idx.add(int(i))

        print(f"  Round {rnd + 1}/{N_ROUNDS}  labeled={len(labeled_idx):,}  "
              f"best={min(labeled_scores.values()):.2f}  top-{TOPK} recall={recall(labeled_scores):.0%}")
    return recall(labeled_scores)


r_rand = run("random")
r_greedy = run("greedy")
print(f"\nSame budget ({INIT + N_ROUNDS * BATCH} of {N_POOL} docked): "
      f"random found {r_rand:.0%} of the top-{TOPK}, active learning found {r_greedy:.0%}.")
print("That gap is the whole point of the repo.  The real experiment asks how to make it bigger.")

# EXERCISE 1: change BATCH to 50 and N_ROUNDS to 10 (same total budget). Does recall go up?
#             (al-molecular studies the same question with batch = 0.1% / 0.2% / 0.4% of the pool.)
# EXERCISE 2: delete the sign flip (use +1.0). What happens to greedy, and why?
