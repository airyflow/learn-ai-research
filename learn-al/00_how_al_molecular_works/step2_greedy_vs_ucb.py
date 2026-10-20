"""Step 2: uncertainty, and the two acquisition functions the repo compares.

Run:   python 00_how_al_molecular_works/step2_greedy_vs_ucb.py      (CPU, ~3 seconds)

Step 1's surrogate returned one number per molecule: its guess (mu).
al-molecular's surrogates return TWO numbers:
    mu    = "I think the score is about this"
    sigma = "...and I'm this unsure about it"
Then an ACQUISITION FUNCTION turns (mu, sigma) into one number to sort by:
    greedy:  score = mu                     -> pick what looks best
    ucb:     score = mu + beta * sigma      -> pick what COULD be best (optimism)
These are the real functions from al-molecular/molpal/acquirer/metrics.py,
imported below, which are what `--acq greedy` and `--acq ucb` call.
"""
import sys
from pathlib import Path

import numpy as np

AL_REPO = Path(__file__).resolve().parents[3] / "al-molecular-tutorial"
sys.path.insert(0, str(AL_REPO))
from molpal.acquirer.metrics import greedy, ucb   # noqa: E402  the repo's own code

# ---------------------------------------------------------------------------
# Part A. Four molecules, by hand.  (Remember: after the sign flip, higher = better.)
# ---------------------------------------------------------------------------
names = ["A", "B", "C", "D"]
mu = np.array([10.0, 9.5, 9.0, 6.0])
sigma = np.array([0.1, 0.2, 1.5, 0.3])
var = sigma ** 2      # metrics.ucb wants VARIANCE; EmbeddingMVEModel.get_means_and_vars squares sigma for it

print("Part A: pick 1 molecule out of 4")
print("  mol   mu   sigma   greedy   ucb(beta=2)")
for n, m, s, g, u in zip(names, mu, sigma, greedy(mu), ucb(mu, var)):
    print(f"   {n}  {m:5.1f}  {s:5.1f}   {g:6.1f}   {u:6.1f}")
print(f"  greedy picks {names[np.argmax(greedy(mu))]} (best guess);  "
      f"ucb picks {names[np.argmax(ucb(mu, var))]} (could be 9 + 2*1.5 = 12, the model just doesn't know yet)")

# ---------------------------------------------------------------------------
# Part B. Where does sigma come from?  One common trick: train several models
# and see how much they DISAGREE.  Here: 5 linear models, each trained on a
# random resample of the training data ("bootstrap").  Where they agree, sigma
# is small; where they disagree, sigma is big.
#   al-molecular has several versions of this idea:
#     MVE head      (SingleBackboneMVESurrogate) - the net predicts sigma directly
#     ensemble      (LTAllSurrogate)             - M=5 nets, sigma = their std
#     rank disagreement (EnsembleFusionSurrogate)- see step 3
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
N_POOL, TOPK, INIT, BATCH, N_ROUNDS = 5_000, 50, 100, 100, 5
emb = rng.normal(size=(N_POOL, 8)).astype(np.float32)
true_scores = -8 + 1.2 * emb[:, 0] - 0.8 * emb[:, 1] + 0.6 * emb[:, 2] * emb[:, 3] + rng.normal(scale=0.4, size=N_POOL)
true_top_k = set(np.argsort(true_scores)[:TOPK].tolist())


class BootstrapEnsemble:
    def __init__(self, n_models=5, seed=0):
        self.n_models, self.rng = n_models, np.random.default_rng(seed)

    def fit(self, X, y):
        X1 = np.c_[X, np.ones(len(X))]
        self.ws = []
        for _ in range(self.n_models):
            pick = self.rng.integers(0, len(X), len(X))         # resample rows with replacement
            w, *_ = np.linalg.lstsq(X1[pick], y[pick], rcond=None)
            self.ws.append(w)

    def predict(self, X):                                       # -> (mu, sigma), like every surrogate
        preds = np.stack([np.c_[X, np.ones(len(X))] @ w for w in self.ws])   # shape (n_models, n)
        return preds.mean(axis=0), preds.std(axis=0)


def run(acq, seed):
    r = np.random.default_rng(seed)
    labeled = set(r.choice(N_POOL, INIT, replace=False).tolist())
    model = BootstrapEnsemble(seed=seed)
    curve = []
    for _ in range(N_ROUNDS):
        idx = np.array(sorted(labeled))
        model.fit(emb[idx], -true_scores[idx])                  # sign flip, as in step 1
        pool_idx = np.array([i for i in range(N_POOL) if i not in labeled])
        mu, sigma = model.predict(emb[pool_idx])
        scores = ucb(mu, sigma ** 2) if acq == "ucb" else greedy(mu)
        labeled.update(pool_idx[np.argsort(scores)[::-1][:BATCH]].tolist())
        curve.append(len(labeled & true_top_k) / TOPK)
    return curve


print("\nPart B: same loop as step 1, 5 seeds each, top-50 recall per round")
for acq in ["greedy", "ucb"]:
    curves = np.array([run(acq, s) for s in range(42, 47)])
    means = "  ".join(f"{m:.0%}" for m in curves.mean(axis=0))
    print(f"  {acq:6s}  rounds 1-5: {means}   final {curves[:, -1].mean():.0%} +- {curves[:, -1].std():.0%}")

print("""
How to read this: on this toy, greedy and UCB end up about tied -- the gap
is no bigger than the +- spread between seeds, so it's not a real difference.
That matches what the real repo found on EnamineHTS: after the 2026-10-10
fusion fix, UCB ties greedy within seed noise (README table), and
adaptive_acq/ found a fixed greedy beta=0 is as good as any learned beta.
Always compare over several seeds -- a single run can mislead you.""")

# EXERCISE 1: try beta = 0.5 and beta = 5 (ucb(mu, sigma**2, beta=...)). Which extreme looks like random search?
# EXERCISE 2: set n_models=1. What is sigma now, and what does UCB turn into?
#             (That was exactly the fusion bug: sigma = 0 made "ucb" runs identical to greedy.)
