"""A tiny active-learning campaign, written as an RL environment.

IN PLAIN WORDS: a fake screening campaign that runs in 0.2 s. 10,000 "molecules", a small
surrogate that predicts mu and sigma, and 6 rounds where you choose beta and see what you
found. Two kinds of library: 'smooth' (greedy wins) and 'hidden' (a little exploring wins).
Every lesson tests its controller here. Walkthrough: LESSONS.md, Lesson 0.

Every module in this course (bandits, MDPs, LinUCB, GRPO, the LLM agent) uses
this environment, and it deliberately copies al-molecular/adaptive_acq:

  adaptive_acq                     here
  ------------------------------   ----------------------------------------------
  2.1M ENHITS molecules            n (default 10k) points in R^3, Gaussian clusters
  docking score (lower = better)   y (HIGHER = better here, to keep the maths simple)
  LT-All / ridge ensemble          Bayesian linear regression on random Fourier
                                   features (a cheap Gaussian-process stand-in)
  UCB = mu + beta_eff * sigma      the same, with the same "normalized" beta
  state.featurize() 10 features    observe() -> the same 10 feature names
  observable reward (hit rate)     the same: fraction of the batch above a cutoff
                                   estimated once from the random initial sample
  true top-k recall                info["recall"]; never part of the observation

Gym-style API:
    env = ToyCampaign(regime="smooth")
    obs = env.reset(seed=0)                     # np.ndarray, shape (10,)
    obs, reward, done, info = env.step(beta)    # one AL round with this beta
    env.peek(beta)                              # counterfactual reward (sim only)

Regimes (the "task" a controller faces; a real campaign doesn't know which one it's in):
    smooth  -- the good molecules sit in the clusters the random init already
               covers. Greedy is best; exploring wastes batch. (ENHITS behaves like this.)
    hidden  -- a small, far-away cluster (1% of the library) holds most of the
               top 1%. Greedy rarely finds it; exploration through sigma does.
               Greedy still has the HIGHER per-round hit rate here -- a trap for
               any controller that only looks at this round's reward.
    mixed   -- each reset() picks smooth or hidden at random (for contextual
               controllers: can they tell which one they're in?).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

torch.set_num_threads(min(4, torch.get_num_threads()))  # be polite on shared login nodes

FEATURE_NAMES = (
    "bias",
    "progress",             # round / n_rounds
    "is_first",             # 1 on the first round (no feedback yet)
    "last_reward",          # previous round's hit fraction
    "last_spearman",        # rank corr(predicted mu, observed y) on last batch
    "last_sigma_err_corr",  # rank corr(sigma, |error|) on last batch: is sigma informative?
    "headroom",             # tanh(mean sigma of greedy top-B / std(mu))
    "ucb_greedy_jaccard",   # overlap of greedy top-B and beta_max top-B
    "improvement",          # tanh(change in mean of best-k labeled, in labeled-std units)
    "last_beta",            # previous beta / beta_max
)
FEATURE_DIM = len(FEATURE_NAMES)
ARMS = (0.0, 0.25, 0.5, 1.0, 2.0)  # normalized beta values; 0 == greedy
EPS = 1e-12


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b):
    if len(a) < 3:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    if ra.std() < EPS or rb.std() < EPS:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def top_k(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def ucb(mu, sigma, beta):
    """Normalized beta, as in adaptive_acq.acquisition.effective_beta: beta=1
    means an average uncertainty bonus is worth one std of predicted score."""
    beta_eff = beta * mu.std() / (sigma.mean() + EPS)
    return mu + beta_eff * sigma


# ── the library ───────────────────────────────────────────────────────────────

def make_library(regime, n, d=3, seed=0):
    """Returns X (n, d) and y (n,), higher y = better."""
    rng = np.random.default_rng(seed)
    n_hidden = int(0.01 * n) if regime == "hidden" else 0
    n_main = n - n_hidden
    centers = rng.normal(0, 2.0, size=(6, d))
    assign = rng.integers(0, len(centers), n_main)
    X = centers[assign] + rng.normal(0, 1.0, size=(n_main, d))

    # Smooth objective over the main clusters: a linear trend plus a few bumps.
    w = rng.normal(0, 1, d) / np.sqrt(d)
    bumps = centers[rng.choice(len(centers), 3, replace=False)] + rng.normal(0, 0.5, (3, d))

    def f(Z):
        out = 0.6 * Z @ w
        for c in bumps:
            out += 1.5 * np.exp(-0.5 * ((Z - c) ** 2).sum(1) / 1.5 ** 2)
        return out

    y = f(X)
    if n_hidden:
        # A compact cluster far from everything, with uniformly high scores.
        far = rng.normal(0, 1, d)
        far = 6.0 * far / np.linalg.norm(far)
        Xh = far + rng.normal(0, 0.6, size=(n_hidden, d))
        yh = np.quantile(y, 0.995) + 0.3 * y.std() * rng.standard_normal(n_hidden)
        X, y = np.vstack([X, Xh]), np.concatenate([y, yh])
    y = y + rng.normal(0, 0.05 * y.std(), len(y))          # oracle noise
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ── the surrogate: Bayesian linear regression on random Fourier features ─────

class RFFSurrogate:
    """mu, sigma over the whole library, retrained from scratch each round.
    sigma grows away from labeled data, which is what makes UCB meaningful."""

    def __init__(self, X, n_features=256, lengthscale=1.5, noise=0.1, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1 / lengthscale, size=(X.shape[1], n_features))
        b = rng.uniform(0, 2 * np.pi, n_features)
        Phi = np.sqrt(2.0 / n_features) * np.cos(X @ W + b)        # (n, D), computed once
        # The matrix products run in torch: numpy in this conda env is linked
        # against the slow reference BLAS, torch ships MKL (about 15x faster here).
        self.Phi = torch.from_numpy(Phi)
        self.noise = noise

    def fit_predict(self, labeled_idx, y_labeled):
        m, s = y_labeled.mean(), y_labeled.std() + EPS
        t = (y_labeled - m) / s
        P = self.Phi[torch.from_numpy(np.asarray(labeled_idx))]
        A = P.T @ P / self.noise ** 2 + torch.eye(P.shape[1], dtype=P.dtype)
        Ainv = torch.linalg.inv(A)                                  # (D, D): small, so invert once
        w = Ainv @ (P.T @ torch.from_numpy(t)) / self.noise ** 2    # posterior mean weights
        mu = (self.Phi @ w).numpy()
        var = ((self.Phi @ Ainv) * self.Phi).sum(1).clamp_min(0.0)  # phi^T A^-1 phi per molecule
        return mu * s + m, var.sqrt().numpy() * s


# ── the environment ───────────────────────────────────────────────────────────

@dataclass
class _Last:
    mu: np.ndarray
    sigma: np.ndarray
    y: np.ndarray
    reward: float
    beta: float
    best_k_before: float


class ToyCampaign:
    def __init__(self, regime="smooth", n=10_000, init=100, batch=50, n_rounds=6,
                 topk_frac=0.01, hit_q=0.95, arms=ARMS):
        assert regime in ("smooth", "hidden", "mixed")
        self.regime_arg = regime
        self.n, self.init, self.batch, self.n_rounds = n, init, batch, n_rounds
        self.topk = max(1, int(round(topk_frac * n)))
        self.hit_q = hit_q
        self.arms = tuple(arms)
        self.beta_max = max(self.arms) if max(self.arms) > 0 else 1.0
        self._cache = {}

    def _library(self, regime, lib_seed):
        key = (regime, lib_seed)
        if key not in self._cache:
            X, y = make_library(regime, self.n, seed=lib_seed)
            self._cache[key] = (X, y, RFFSurrogate(X, seed=lib_seed))
        return self._cache[key]

    def reset(self, seed=0, lib_seed=None):
        """seed fixes the initial random sample (and the regime when "mixed").
        lib_seed picks the library; default = a small set of 4 libraries so
        repeated episodes reuse cached features."""
        rng = np.random.default_rng(seed)
        self.regime = self.regime_arg if self.regime_arg != "mixed" else ("smooth", "hidden")[rng.integers(2)]
        lib_seed = int(rng.integers(4)) if lib_seed is None else lib_seed
        self.X, self.y, self.surr = self._library(self.regime, lib_seed)
        self.true_top = np.zeros(self.n, dtype=bool)
        self.true_top[top_k(self.y, self.topk)] = True

        self.labeled = np.zeros(self.n, dtype=bool)
        init_idx = rng.choice(self.n, self.init, replace=False)
        self.labeled[init_idx] = True
        self.labeled_idx = init_idx
        # The hit cutoff is estimated ONCE from the random initial sample and
        # then frozen (adaptive_acq/state.py explains why moving cutoffs fail).
        self.hit_thr = float(np.quantile(self.y[init_idx], self.hit_q))
        self.rnd, self.last = 0, None
        self._predict()
        return self.observe()

    def _predict(self):
        mu, sigma = self.surr.fit_predict(self.labeled_idx, self.y[self.labeled_idx])
        self.cand = np.flatnonzero(~self.labeled)
        self.mu, self.sigma = mu[self.cand], sigma[self.cand]

    def _best_k_mean(self):
        yl = self.y[self.labeled_idx]
        k = min(self.topk, len(yl))
        return float(np.sort(yl)[-k:].mean())

    def _pick(self, beta):
        return top_k(ucb(self.mu, self.sigma, beta), self.batch)

    def observe(self):
        """The controller's context: only things a real campaign could compute."""
        g = self._pick(0.0)
        e = self._pick(self.beta_max)
        jacc = len(np.intersect1d(g, e)) / len(np.union1d(g, e))
        headroom = np.tanh(self.sigma[g].mean() / (self.mu.std() + EPS))
        if self.last is None:
            is_first, lr, rho, rho_sig, imp, lb = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            L = self.last
            is_first, lr, lb = 0.0, L.reward, L.beta / self.beta_max
            rho = spearman(L.mu, L.y)
            rho_sig = spearman(L.sigma, np.abs(L.mu - L.y))
            yl = self.y[self.labeled_idx]
            imp = np.tanh((self._best_k_mean() - L.best_k_before) / (yl.std() + EPS))
        return np.array([1.0, self.rnd / self.n_rounds, is_first, lr, rho, rho_sig,
                         headroom, jacc, imp, lb])

    def recall(self):
        return float(self.true_top[self.labeled_idx].sum() / self.topk)

    def peek(self, beta):
        """Reward had this round used `beta`. Needs the oracle on molecules we
        did NOT acquire, so only a simulator can do it (train_linucb.py uses
        exactly this 'full-information' trick)."""
        new = self.cand[self._pick(beta)]
        return float((self.y[new] > self.hit_thr).mean())

    def snapshot(self):
        """Everything step() changes. restore(snapshot()) rewinds the campaign, so
        a simulator can ask "what if I had played a different beta?" (module 04)."""
        return (self.labeled.copy(), self.labeled_idx.copy(), self.rnd, self.last,
                self.cand, self.mu, self.sigma)

    def restore(self, snap):
        labeled, labeled_idx, self.rnd, self.last, self.cand, self.mu, self.sigma = snap
        self.labeled, self.labeled_idx = labeled.copy(), labeled_idx.copy()

    def step(self, beta):
        assert self.rnd < self.n_rounds, "episode is over; call reset()"
        sel = self._pick(float(beta))
        new = self.cand[sel]
        new_y = self.y[new]
        reward = float((new_y > self.hit_thr).mean())
        self.last = _Last(self.mu[sel], self.sigma[sel], new_y, reward, float(beta), self._best_k_mean())

        self.labeled[new] = True
        self.labeled_idx = np.concatenate([self.labeled_idx, new])
        self.rnd += 1
        done = self.rnd == self.n_rounds
        if not done:
            self._predict()
        info = {"recall": self.recall(), "regime": self.regime, "round": self.rnd}
        return self.observe() if not done else np.zeros(FEATURE_DIM), reward, done, info


def run_fixed(env, beta, seed):
    """Convenience: one episode at a fixed beta -> (rewards, recalls)."""
    env.reset(seed)
    rewards, recalls, done = [], [], False
    while not done:
        _, r, done, info = env.step(beta)
        rewards.append(r)
        recalls.append(info["recall"])
    return rewards, recalls


if __name__ == "__main__":
    import time
    for regime in ("smooth", "hidden"):
        env = ToyCampaign(regime)
        for beta in ARMS:
            t0 = time.perf_counter()
            res = [run_fixed(env, beta, s) for s in range(8)]
            rew = np.mean([np.mean(r) for r, _ in res])
            fin = np.mean([c[-1] for _, c in res])
            print(f"{regime:6s} beta={beta:4.2f}  mean hit-rate={rew:.3f}  final recall={fin:.3f}"
                  f"  ({(time.perf_counter() - t0) / 8:.2f}s/episode)")
