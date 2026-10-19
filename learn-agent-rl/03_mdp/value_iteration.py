"""MDPs, the Bellman equation, value iteration and Q-learning -- on a 4-state cartoon of AL.

IN PLAIN WORDS: a cartoon campaign small enough to solve exactly. Work backwards from the last
round ("exploring then is useless") to the first, and you get the best plan: explore early,
exploit late. Q-learning finds the same plan by trial and error. Short campaigns -> no exploring.
Core exercises: 1 and 2 (and read 3). Walkthrough: LESSONS.md, Lesson 3.

Run:   python 03_mdp/value_iteration.py           (a few seconds)

Module 02 showed that on the 'hidden' regime the policy with the highest per-round reward
(greedy) has the worst final recall. This module strips the problem down to a tiny MDP small
enough to solve EXACTLY, so you can see why: exploration has DELAYED value.

The cartoon:
  state   (t, q):  t = round (0..T-1), q = surrogate quality, 0 (knows nothing) .. 3 (mapped
                   the whole landscape, including the hidden cluster)
  actions explore (UCB with beta>0) or exploit (greedy)
  exploit: hits this round = EXPLOIT_HITS[q]; q stays the same (greedy only samples where the
           model is already confident, so it learns little that is new)
  explore: hits this round = EXPLORE_HITS (fewer); with prob P_LEARN, q -> q + 1
Reward = hits this round. Return = sum of rewards over T rounds (gamma = 1: finite horizon).

The Bellman optimality equation (finite horizon, Sutton & Barto eq. 3.20 / 4.1):
    Q_t(q, a) = E[ r(q, a) + V_{t+1}(q') ]
    V_t(q)    = max_a Q_t(q, a),        V_T(q) = 0
Solving it backwards from t = T-1 to 0 is "backward induction" = value iteration for a
finite horizon. It needs the model (EXPLOIT_HITS, P_LEARN). Q-learning (part 2) learns the
same Q from sampled episodes without knowing the model.
"""
import numpy as np

T = 6
Q_LEVELS = 4
EXPLOIT_HITS = np.array([0.30, 0.50, 0.70, 0.90])   # better model -> better greedy picks
EXPLORE_HITS = 0.10
P_LEARN = 0.8
ACTIONS = ("exploit", "explore")


def step_model(q, a):
    """Returns a list of (prob, reward, next_q): the full transition model."""
    if a == 0:
        return [(1.0, EXPLOIT_HITS[q], q)]
    nq = min(q + 1, Q_LEVELS - 1)
    return [(P_LEARN, EXPLORE_HITS, nq), (1 - P_LEARN, EXPLORE_HITS, q)]


# ── 1. value iteration (backward induction) ──────────────────────────────────

def value_iteration(T=T, gamma=1.0):
    V = np.zeros((T + 1, Q_LEVELS))                    # V[T] = 0: no rounds left
    Q = np.zeros((T, Q_LEVELS, 2))
    for t in reversed(range(T)):
        for q in range(Q_LEVELS):
            for a in range(2):
                Q[t, q, a] = sum(p * (r + gamma * V[t + 1, nq]) for p, r, nq in step_model(q, a))
            V[t, q] = Q[t, q].max()
    return V, Q


def evaluate(policy, T=T, gamma=1.0):
    """Exact expected return of a given policy(t, q) -> a, starting at q=0."""
    V = np.zeros((T + 1, Q_LEVELS))
    for t in reversed(range(T)):
        for q in range(Q_LEVELS):
            a = policy(t, q)
            V[t, q] = sum(p * (r + gamma * V[t + 1, nq]) for p, r, nq in step_model(q, a))
    return V[0, 0]


# ── 2. tabular Q-learning (model-free) ────────────────────────────────────────

def sample(q, a, rng):
    outs = step_model(q, a)
    i = rng.choice(len(outs), p=[o[0] for o in outs])
    return outs[i][1], outs[i][2]


def q_learning(episodes=20000, eps=0.3, seed=0):
    """Q(t,q,a) <- Q + alpha * (r + max_a' Q(t+1,q',a') - Q). The target uses the max over
    next actions (off-policy), while behaviour is eps-greedy (module 01!).
    alpha = 1 / (visits to (t,q,a)): a running average, which converges here."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((T + 1, Q_LEVELS, 2))
    N = np.zeros((T, Q_LEVELS, 2))
    for _ in range(episodes):
        q = 0
        for t in range(T):
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[t, q]))
            r, nq = sample(q, a, rng)
            target = r + Q[t + 1, nq].max()            # Q[T] stays 0
            N[t, q, a] += 1
            Q[t, q, a] += (target - Q[t, q, a]) / N[t, q, a]
            q = nq
    return Q[:T]


def show_policy(Q, title):
    print(f"\n{title}  (rows: q = surrogate quality, cols: round t; E = explore, . = exploit)")
    for q in range(Q_LEVELS):
        # q > t can't happen when starting from q=0 (q rises at most 1 per round): shown as blank
        print(f"  q={q}  " + " ".join(" " if q > t else ("E" if Q[t, q, 1] > Q[t, q, 0] else ".")
                                      for t in range(Q.shape[0])))


if __name__ == "__main__":
    V, Q = value_iteration()
    show_policy(Q, "Optimal policy from value iteration")
    print(f"\nV_0(q=0), optimal:             {V[0, 0]:.3f}")
    print(f"always exploit (myopic/greedy): {evaluate(lambda t, q: 0):.3f}")
    print(f"always explore:                 {evaluate(lambda t, q: 1):.3f}")
    print(f"explore 2 rounds, then exploit: {evaluate(lambda t, q: 1 if t < 2 else 0):.3f}")
    # The myopic policy picks argmax of the IMMEDIATE reward -- exactly what the bandit in
    # module 01 Part C learned. At q=0, exploit gives 0.30 > 0.25, so it never explores.

    Qhat = q_learning()
    show_policy(Qhat, "Policy learned by Q-learning (20000 sampled episodes, no model)")
    reach = np.array([[q <= t for q in range(Q_LEVELS)] for t in range(T)])
    print(f"max |Q_learned - Q_true| over reachable states = {np.abs(Qhat - Q)[reach].max():.3f}")

    for horizon in (2, 3, 4, 6, 10):
        _, Qh = value_iteration(T=horizon)
        n_explore = int((Qh[:, 0, 1] > Qh[:, 0, 0]).sum())
        print(f"horizon T={horizon:2d}: optimal plan from q=0 explores in round(s) "
              f"{[t for t in range(horizon) if Qh[t, 0, 1] > Qh[t, 0, 0]]}  ({n_explore} at q=0)")

# EXERCISE 1: Verify one Bellman backup by hand: Q_{T-1}(q, explore) and Q_{T-1}(q, exploit) for
#   q=0. Why is the last round ALWAYS exploit, whatever q is? (No future left to benefit.)
# EXERCISE 2: The horizon table: with T=2, exploring never pays. adaptive_acq campaigns have 5
#   rounds and very good surrogates from round 1 (high q at t=0). Use this to explain, in two
#   sentences, Stage 1's "greedy is best on ENHITS" result.
# EXERCISE 3: Set gamma=0.0 in value_iteration (pass it through). What policy do you get? Explain
#   why "a contextual bandit = an MDP with gamma = 0" -- the most useful sentence in this course.
# EXERCISE 4: Q-learning with eps=0.0. Does it still find the optimal policy? Why not? (Module 01.)
# EXERCISE 5: Make P_LEARN depend on q (e.g. harder to improve a good model). How does the
#   optimal policy's shape change? This is the "beta_t = f(s_t)" the brief talks about: here
#   you can compute f exactly; in modules 04-06 you have to learn it.
