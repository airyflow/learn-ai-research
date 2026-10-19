"""Policy gradient: REINFORCE -> REINFORCE with a baseline -> GRPO, on the toy campaign.

IN PLAIN WORDS: a small network maps the situation to probabilities over beta. Play whole
campaigns; make the choices from better-than-expected campaigns more likely. Plain REINFORCE
learns nothing here, adding a baseline fixes it, GRPO (compare 8 campaigns from the same start)
learns fastest -- and discovers "explore early" on its own.
Core exercises: 1 and 3. Walkthrough: LESSONS.md, Lesson 5.

Run:   python 05_policy_gradient/grpo.py --algo grpo          (~1-2 min on 8 CPU workers)
       python 05_policy_gradient/grpo.py --algo reinforce
       python 05_policy_gradient/grpo.py --algo baseline

Modules 01 and 04 learned a VALUE per arm (a hit rate, theta_a . x) and acted greedily on it.
Policy gradient skips the value: it parameterizes the policy directly,

    pi_theta(a | s) = softmax(MLP_theta(s))            (policy.py:BetaPolicy is the same network)

and climbs the expected RETURN of whole episodes. The return can be anything a simulator can
score -- here, final recall -- so the delayed value of exploration (module 03) is in the signal.

The policy gradient theorem (Sutton & Barto ch. 13; Spinning Up "Intro to Policy Optimization"):

    grad J(theta) = E[ sum_t grad log pi_theta(a_t | s_t) * A_t ]

  REINFORCE   A_t = R (the episode's return). Unbiased, very noisy: every action in a good
              episode is pushed up, including the bad ones.
  baseline    A_t = R - b, with b = mean return of the batch. Same expectation, less variance
              (subtracting a constant doesn't change E[grad log pi] = 0).
  GRPO        A_t = (R_i - mean_G R) / std_G R over a GROUP of G rollouts from the SAME start
              (same initial labeled set): "was this rollout better than its siblings?". This
              cancels the luck of the start, which dominates the return here. Then a PPO-style
              update: several epochs on the same batch, the probability ratio clipped to
              [1 - eps, 1 + eps], plus a KL penalty to the initial policy. No value network
              (that is the "G" -- group-relative -- in GRPO; Shao et al. 2024, DeepSeekMath).

Task: regime='hidden', where the best fixed beta (0.25) reaches ~0.93 final recall and greedy
only ~0.64 (common/toy_campaign.py). A policy that starts uniform over ARMS should learn to
put its mass on the betas that explore a little.
"""
import argparse
import copy
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.toy_campaign import ARMS, FEATURE_DIM, ToyCampaign  # noqa: E402


class Policy(nn.Module):
    """A copy of adaptive_acq/policy.py:BetaPolicy: 10 features in, log-probs over arms out."""

    def __init__(self, n_arms=len(ARMS), hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_arms))
        nn.init.zeros_(self.net[-1].weight)       # start exactly uniform over arms
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.log_softmax(self.net(x), dim=-1)


# ── rollouts (run in worker processes; each holds its own env) ────────────────

_ENV = None


def _init_worker(regime):
    global _ENV
    torch.set_num_threads(1)
    _ENV = ToyCampaign(regime)


def rollout(job):
    """job = (policy state_dict, start seed, sampling seed, mode). Returns the trajectory
    [(obs, action, logp_old)] and the return R = final recall."""
    state, seed, rseed, mode = job
    pol = Policy()
    pol.load_state_dict(state)
    rng = np.random.default_rng(rseed)
    obs, done, traj = _ENV.reset(seed=seed), False, []
    while not done:
        with torch.no_grad():
            logp = pol(torch.as_tensor(obs, dtype=torch.float32)).numpy().astype(np.float64)
        p = np.exp(logp) / np.exp(logp).sum()
        a = int(rng.choice(len(ARMS), p=p)) if mode == "sample" else int(np.argmax(p))
        traj.append((obs, a, float(logp[a])))
        obs, _, done, info = _ENV.step(ARMS[a])
    return {"seed": seed, "traj": traj, "R": info["recall"], "betas": [ARMS[a] for _, a, _ in traj]}


# ── the three updates ─────────────────────────────────────────────────────────

def batch_tensors(rows):
    X = torch.as_tensor(np.stack([r[0] for r in rows]), dtype=torch.float32)
    a = torch.as_tensor([r[1] for r in rows])
    logp_old = torch.as_tensor([r[2] for r in rows], dtype=torch.float32)
    adv = torch.as_tensor([r[3] for r in rows], dtype=torch.float32)
    return X, a, logp_old, adv


def advantages(results, algo, group_size):
    """One advantage per episode, then copied to every step of that episode."""
    R = np.array([r["R"] for r in results])
    if algo == "reinforce":
        return R
    if algo == "baseline":
        return R - R.mean()
    A = np.empty_like(R)                                   # grpo: normalize within each group
    for g in range(0, len(R), group_size):
        Rg = R[g:g + group_size]
        A[g:g + group_size] = (Rg - Rg.mean()) / (Rg.std() + 1e-6)
    return A


def update(policy, ref, opt, rows, algo, epochs, clip, kl_coef, ent_coef):
    X, a, logp_old, adv = batch_tensors(rows)
    n_epochs = epochs if algo == "grpo" else 1             # vanilla PG: one step per batch (on-policy)
    with torch.no_grad():
        logp_ref = ref(X).gather(1, a[:, None]).squeeze(1)
    for _ in range(n_epochs):
        logp_all = policy(X)
        logp = logp_all.gather(1, a[:, None]).squeeze(1)
        ent = -(logp_all.exp() * logp_all).sum(1).mean()
        if algo == "grpo":
            ratio = (logp - logp_old).exp()                 # pi_new / pi_old for the taken action
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            d = logp_ref - logp
            kl = (d.exp() - d - 1).mean()                  # k3 estimator of KL(pi || pi_ref), >= 0
            loss = pg + kl_coef * kl - ent_coef * ent
        else:
            kl = torch.tensor(0.0)
            loss = -(logp * adv).mean() - ent_coef * ent
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return float(ent), float(kl)


def state_of(policy):
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["reinforce", "baseline", "grpo"], default="grpo")
    p.add_argument("--regime", default="hidden", choices=["smooth", "hidden", "mixed"])
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--groups", type=int, default=4, help="start seeds per iteration")
    p.add_argument("--group-size", type=int, default=8, help="rollouts per start seed (G)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--test", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = Policy()
    ref = copy.deepcopy(policy).eval()                     # the KL anchor: the uniform start
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    test_seeds = list(range(10_000, 10_000 + args.test))   # never used for training

    with get_context("fork").Pool(args.workers, initializer=_init_worker, initargs=(args.regime,)) as pool:
        def evaluate():
            s = state_of(policy)
            res = pool.map(rollout, [(s, sd, 0, "greedy") for sd in test_seeds])
            return np.mean([r["R"] for r in res]), res

        R0, _ = evaluate()
        print(f"[{args.algo}] regime={args.regime}  greedy-decoded test recall before training: {R0:.3f}")
        for it in range(args.iters):
            t0 = time.perf_counter()
            s = state_of(policy)
            starts = [int(rng.integers(1 << 30)) for _ in range(args.groups)]
            jobs = [(s, sd, int(rng.integers(1 << 30)), "sample") for sd in starts for _ in range(args.group_size)]
            results = pool.map(rollout, jobs)              # ordered: group g = results[g*G:(g+1)*G]
            A = advantages(results, args.algo, args.group_size)
            rows = [(o, a, lp, float(adv)) for r, adv in zip(results, A) for o, a, lp in r["traj"]]
            ent, kl = update(policy, ref, opt, rows, args.algo, args.epochs, args.clip, args.kl_coef, args.ent_coef)
            beta_hist = np.bincount([a for r in results for _, a, _ in r["traj"]], minlength=len(ARMS))
            print(f"  it {it:3d}  mean R={np.mean([r['R'] for r in results]):.3f}  entropy={ent:.2f}  "
                  f"kl={kl:.3f}  arm share={np.round(beta_hist / beta_hist.sum(), 2)}  "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

        R1, res = evaluate()
        print(f"\n[{args.algo}] greedy-decoded test recall after training: {R1:.3f}  (was {R0:.3f})")
        print("  betas chosen by round, first 5 test campaigns:")
        for r in res[:5]:
            print(f"    seed {r['seed']}: {r['betas']}  recall={r['R']:.2f}")
        print("  reference (common/toy_campaign.py, 'hidden'): greedy ~0.64, fixed beta=0.25 ~0.93")


if __name__ == "__main__":
    main()

# EXERCISE 1: Run all three --algo values with the same --seed. What I got (seed 0, 800 episodes
#   each): reinforce 0.692 -> 0.692 (learned nothing), baseline 0.692 -> 0.927, grpo 0.692 ->
#   0.933, and grpo's arm shares moved fastest. Why does plain REINFORCE fail? Every R here is
#   ~0.6-1.0, all positive, so EVERY sampled action gets pushed up; the signal is the small
#   difference between episodes, buried under that common push. Run each with 2 more seeds:
#   which is most consistent? (Variance is the whole story of policy gradient.)
#   Also look at what GRPO learned: beta=1 in round 0, then 0.5. Compare module 03.
# EXERCISE 2: In advantages(), why can GRPO divide by the group's std, but REINFORCE can't use
#   R directly without exploding/vanishing steps? What happens to a group where all 8 rollouts
#   get the same R? (Answer: zero advantage -> zero gradient. Is that a bug or a feature?)
# EXERCISE 3: Set --kl-coef 0 --ent-coef 0 and train longer (--iters 60). Watch the entropy.
#   What does the policy collapse to, and does the test recall go up or down? This is why GRPO
#   (and RLHF) keep a KL anchor to a reference policy.
# EXERCISE 4: --regime smooth. The best policy is greedy. Does GRPO find it? Then --regime mixed:
#   can it do better than one fixed beta, i.e. does it USE the observation? Check the chosen
#   betas per test campaign against its regime.
# EXERCISE 5: Test seeds reuse the same 4 libraries as training (ToyCampaign.reset picks
#   lib_seed in 0..3). So "test" here only means "new initial sample". Change evaluate() to
#   use lib_seed=7. Does recall hold up? This is why adaptive_acq trains on D4, selects on
#   AmpC and reports on ENHITS.
# EXERCISE 6: Map this file onto adaptive_acq/train_grpo.py: _rollout, grpo_update, the group
#   construction, the val_starts / policy_best.pt logic. What does train_grpo.py have that this
#   file lacks? (Validation-based checkpoint selection; fixed-beta baselines on the val starts.)
