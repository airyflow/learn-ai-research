# Glossary

Each term is defined in plain words, followed by what it means in your campaign.

## The basics

| Term | Plain meaning | In your campaign |
|---|---|---|
| **agent / controller** | whatever makes the decisions | the thing that picks β each round |
| **environment** | everything the agent acts on | the library, the surrogate, the oracle |
| **state** | the full situation right now | labeled set, μ and σ over the pool, round number |
| **observation** | the part of the state the agent can see | the 10 features from `observe()` |
| **action** | one decision | this round's β (or retrain / stop, later) |
| **reward** | the score for one step | fraction of this batch that were hits |
| **episode** | one full run from start to finish | one campaign (6 rounds in the toy, 5 on ENHITS) |
| **return** | the score for a whole episode | final recall, or the sum of rewards |
| **policy** | the rule mapping observation → action | "β = 0.25 always", a bandit, a network, an LLM |
| **horizon** | how many steps are left | rounds remaining |

## Exploring

| Term | Plain meaning | In your campaign |
|---|---|---|
| **explore / exploit** | try something uncertain / use what looks best | β > 0 / β = 0 |
| **greedy** | always pick the best-looking option | β = 0: top μ |
| **β (beta)** | how much uncertainty counts | score = μ + β·σ |
| **recall** | fraction of the true best you have found | top-1% (toy) or top-1000 (ENHITS) found |
| **regime** | the kind of problem you're in | smooth (greedy wins) or hidden (exploring wins) |
| **myopic / short-sighted** | only cares about this step's reward | a bandit rewarded by this round's hits |
| **delayed value** | a payoff that comes later | exploring now → better model → more hits later |

## Bandits (lessons 1 and 4)

| Term | Plain meaning |
|---|---|
| **multi-armed bandit** | repeatedly choose among options with unknown payoffs; no state, no future |
| **arm** | one option (here, one β value) |
| **regret** | how much reward you lost by not knowing the best arm from the start |
| **ε-greedy** | best-looking arm most of the time, random arm with probability ε |
| **UCB1** | pick the arm with the highest average + bonus for being rarely tried |
| **Thompson sampling** | sample a plausible hit rate per arm from your beliefs; pick the highest sample |
| **contextual bandit** | a bandit that looks at the current situation before choosing |
| **LinUCB** | contextual bandit with one linear regression per arm, plus an uncertainty bonus |
| **full-information training** | in a simulator, you can see what *every* arm would have earned, not just the one you picked |

## Planning (lessons 2 and 3)

| Term | Plain meaning |
|---|---|
| **MDP** | Markov decision process: states, actions, rewards, and the rules for what happens next |
| **POMDP** | an MDP where you only see part of the state (always true in practice) |
| **value V(s)** | the best total score you can still get from state s |
| **Q-value Q(s, a)** | the best total score if you take action a now, then play perfectly |
| **Bellman equation** | value now = reward now + value of where you end up |
| **value iteration / backward induction** | solve the Bellman equation starting from the last step and going backwards |
| **Q-learning** | learn Q-values from experience, without knowing the rules |
| **discount γ (gamma)** | how much future rewards count; γ = 0 makes an MDP into a bandit |

## Learning a policy (lesson 5)

| Term | Plain meaning |
|---|---|
| **policy network** | a neural net from observation to probabilities over actions |
| **policy gradient** | make actions from good episodes more likely, from bad ones less likely |
| **REINFORCE** | the simplest policy gradient: push by the episode's return |
| **baseline** | subtract the average return, so only better-than-average episodes push up |
| **advantage** | how much better this episode (or action) was than expected |
| **GRPO** | group-relative policy optimization: compare each rollout to siblings from the same start; no value network |
| **PPO clipping** | limit how far one update can move the policy |
| **KL penalty** | a cost for drifting too far from a reference policy |
| **entropy** | how spread out the policy's choices are; 0 means it always picks the same thing |
| **rollout** | playing one episode with the current policy |
| **RLHF / RLVR** | RL from human feedback / from *verifiable* rewards (an automatic checker, like your simulator's recall) |

## Agents (lesson 6)

| Term | Plain meaning |
|---|---|
| **LLM agent** | an LLM in a loop: look (tools) → decide → act → see the result |
| **tool / function calling** | the LLM asks your code to run a named function with JSON arguments |
| **tool schema** | the JSON description of a tool's inputs, which the LLM must follow |
| **validation** | checking the LLM's action before running it (schema first, then rules) |
| **repair loop** | sending the error back to the LLM so it can fix its own action |
| **fallback** | the safe default when the agent fails (greedy), always logged |
| **MCP** | Model Context Protocol: a standard way to publish tools so any AI client can call them |
| **mock backend** | a scripted fake LLM, for testing the plumbing for free |
| **vLLM** | software that serves an open-weight LLM on your own GPU |
| **trajectory** | the full record of one agent run: every tool call, result and decision |

## Environments and graders (lesson 7)

| Term | Plain meaning |
|---|---|
| **environment contract** | the two calls every environment offers: `reset(seed)` and `step(action)` |
| **deterministic** | same seed + same actions → exactly the same result, every time |
| **trajectory capture / rollout logging** | saving every step of every episode, to evaluate or train on later |
| **proxy reward** | an easy-to-measure stand-in for what you really want (hits this round instead of final recall) |
| **grader** | code that scores a finished episode, outside the agent's reach |
| **verifier** | a grader that checks against the ground truth automatically (a test suite, a proof checker, your simulator) |
| **verifiable reward** | a reward produced by a verifier; the "VR" in RLVR |
| **reward hacking** | scoring high on the proxy without achieving the goal |

## Evaluating agents (lessons 6+ and 8)

| Term | Plain meaning |
|---|---|
| **prompt injection** | text inside a tool result that tries to give the agent orders |
| **output validation** | checking what a *tool* returns before the model reads it (not just what the model sends) |
| **failure taxonomy** | a named list of the ways an agent goes wrong, built by reading runs |
| **task success rate** | fraction of episodes that pass the grader's threshold |
| **outcome metric** | did it achieve the goal? (success rate, recall) |
| **process metric** | did it behave reliably? (valid first try, repairs, fallbacks, cost, time) |
| **bootstrap** | resample your episodes many times to see how much a number could move by luck |
| **confidence interval (CI)** | the range a number would plausibly fall in if you re-ran; if a difference's CI includes 0, you can't claim it |
| **paired comparison** | compare two agents episode by episode on the same seeds, which removes seed luck |
| **LLM-as-judge** | using an LLM to grade text; must itself be checked against human labels |
| **precision / recall of a grader** | of what it flagged, how much was really wrong / of what was really wrong, how much it flagged |
