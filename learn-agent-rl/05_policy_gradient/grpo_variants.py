"""GRPO and its 2025-26 variants (Dr. GRPO, DAPO, GSPO), on one group of four answers.

Run:   python 05_policy_gradient/grpo_variants.py        (instant, no GPU, no training)

Every variant keeps GRPO's core idea: sample a GROUP of answers to the same question,
score them, and push up the ones that beat the group average. They differ only in small
bookkeeping choices. Each choice turned out to matter in real LLM training. This script
prints, for one toy group, how much each answer gets pushed under each variant.

Toy setup: one question, 4 sampled answers. reward 1 = correct, 0 = wrong.
Answer D is wrong AND very long (rambling), the classic failure GRPO was criticised for.
"""
import numpy as np

answers = ["A", "B", "C", "D"]
reward = np.array([1.0, 1.0, 0.0, 0.0])
length = np.array([50, 60, 40, 400])          # tokens per answer

np.set_printoptions(precision=4, suppress=True)


def show(title, per_token, note):
    print(f"\n{title}")
    for a, r, L, w in zip(answers, reward, length, per_token):
        print(f"  {a}: reward {r:.0f}, {L:4d} tokens -> push per token {w:+.4f}   total push {w * L:+.2f}")
    print("  ->", note)


# ---- 1. GRPO (DeepSeekMath, 2024) -------------------------------------------------
# advantage = (reward - group mean) / group std, then each answer's loss is AVERAGED over its
# own tokens (divide by its length).
adv = (reward - reward.mean()) / (reward.std() + 1e-6)
show("1. GRPO: advantage / std, loss averaged over each answer's own length",
     adv / length,
     "D's 400 wrong tokens are punished 10x more weakly per token than C's 40. Long wrong "
     "answers are cheap, so models learn to ramble.")

# ---- 2. Dr. GRPO (Liu et al., 2025) --------------------------------------------------
# Fix: no division by the answer's length (use one constant for everyone), and no division
# by the group std.
adv_dr = reward - reward.mean()
show("2. Dr. GRPO: no std, no per-answer length division",
     adv_dr / length.max(),
     "every wrong token is now punished the same, however long the answer.")

# ---- 3. DAPO (ByteDance, 2025): token-level averaging over the whole group -------------
adv_dapo = (reward - reward.mean()) / (reward.std() + 1e-6)
show("3. DAPO token-level loss: average over ALL tokens in the group",
     adv_dapo / length.sum(),
     "also length-fair: one token counts the same wherever it comes from.")

print("\nDAPO adds three more tricks:")
easy = np.array([1.0, 1.0, 1.0, 1.0])
print(f"  * dynamic sampling: a group where every answer gets the same reward, e.g. {easy}, has "
      f"advantage {easy - easy.mean()} -> zero gradient. DAPO drops such groups and samples more.")
print("  * clip-higher: the ratio pi_new/pi_old is clipped to [1-0.2, 1+0.28] instead of "
      "[0.8, 1.2], so rare good tokens can grow faster (keeps exploring).")
print("  * overlong shaping: answers past a length limit get a small extra penalty.")

# ---- 4. GSPO (Qwen, 2025): one importance ratio per ANSWER, not per token -------------
rng = np.random.default_rng(0)
token_ratios = np.exp(rng.normal(0, 0.15, size=8))          # pi_new/pi_old for 8 tokens of one answer
token_ratios[3] = 1.9                                        # one noisy token
seq_ratio = np.exp(np.mean(np.log(token_ratios)))            # geometric mean = length-normalised
print("\n4. GSPO: clip one sequence-level ratio instead of every token's ratio")
print("  token ratios     :", token_ratios)
print(f"  clipped per token: {np.clip(token_ratios, 0.8, 1.2)}  <- GRPO/PPO: token 3 is clipped alone")
print(f"  sequence ratio   : {seq_ratio:.4f}  <- GSPO: one number for the whole answer, clipped as a unit")
print("  -> steadier for long answers and mixture-of-experts models, where single-token ratios are noisy.")

# EXERCISE 1: set reward = [1, 0, 0, 0] (a hard question). How does dividing by std change the
#   size of A's push compared with the easy case [1, 1, 1, 0]? Dr. GRPO argues this re-weights
#   questions by difficulty in a way nobody asked for.
# EXERCISE 2: in 05_policy_gradient/grpo.py, update() takes .mean() over ALL steps in the batch.
#   Which variant's averaging is that? Why doesn't length bias matter much in the toy campaign?
#   (Hint: how many steps does every episode have?)
# EXERCISE 3: add a --variant flag to grpo.py's advantages(): grpo | drgrpo (no std).
#   Compare final recall on the hidden library with 5 seeds each.
