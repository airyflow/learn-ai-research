# learn-agent-rl

A hands-on course in **reinforcement learning and LLM agents**, taught through one problem you
already know: *in an active-learning campaign, when should you explore and when should you
exploit?* It is the sequel to `learn-al`: small Python scripts, each runnable in minutes on a
CPU, each ending with exercises that point back into `al-molecular`.

**Start here: [LESSONS.md](LESSONS.md).** It walks through every lesson in plain words:
the idea, what to run, what you should see, and which exercises matter.
Unfamiliar word? Check [GLOSSARY.md](GLOSSARY.md).
Which of these skills do employers want right now? See [TRENDING_SKILLS.md](TRENDING_SKILLS.md).

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/learn-agent-rl
export OMP_NUM_THREADS=4           # be polite on shared nodes
python common/toy_campaign.py      # the playground; should print a table in ~20 s
```

## The lessons

Lessons 1–6 each answer "when to explore?" in a smarter way than the one before. Lessons 7–8
are about building and judging those deciders, which is what most job postings ask for.

| # | Script | The idea in one line | Time to run | Status |
|---|---|---|---|---|
| 0 | `common/toy_campaign.py` | a fake campaign to practise on; no single β wins everywhere | 20 s | ✅ |
| 1 | `01_bandits/bandits.py` | try options, keep the best average, and why that picks greedy | 2 min | ✅ |
| 2 | `02_environment/campaign_as_mdp.py` | the parts of the loop; most hits per round ≠ best result | 30 s | ✅ |
| 3 | `03_mdp/value_iteration.py` | plan backwards from the end: explore early, exploit late | 1 s | ✅ |
| 4 | `04_contextual_bandits/linucb.py` | a bandit that reads the situation first | 6–8 min | ✅ |
| 5 | `05_policy_gradient/grpo.py` | learn a whole strategy by trial and error (GRPO) | 1 min | ✅ |
| 5+ | `05_policy_gradient/grpo_variants.py` | Dr. GRPO, DAPO, GSPO: the 2025–26 fixes, on one toy group | instant | ✅ |
| 6 | `06_agents/agent_controller.py` | an LLM with tools, and every action checked before it runs | 30 s (mock) | ✅ |
| 6+ | `06_agents/injection_test.py` | prompt injection: when a tool result gives the agent orders | 10 s | ✅ |
| 7 | `07_environments/step1…step4` | environments, trajectories, graders, reward hacking | 3–15 s each | ✅ |
| 8 | `08_agent_evals/step1…step5` | evaluating agents: outcome vs process, CIs, checking reasons | seconds each | ✅ |
| 9 | `09_adaptive_acq_tour/` | run your real code on the toy, then a 10-stop reading tour | 10 s + 3 evenings | ✅ |

Lesson 6 has three interchangeable "brains": a free scripted **mock**, a free **local model**
(gpt-oss-20b via vLLM on a GPU: `sbatch 06_agents/run_vllm_agent.slurm` from a Quartz login
node), and **Claude** (paid API, saved for the final comparison table).

## The results you should reproduce

Final recall on the toy campaign (mean over seeds):

| | greedy (β = 0) | β = 0.25 | GRPO policy (lesson 5) | mock agent (lesson 6) |
|---|---|---|---|---|
| smooth library | **0.99** | 0.97 | — | 0.998 |
| hidden library | 0.64 | **0.93** | 0.93 | 0.958 |

The thread through the whole course: **greedy gets the most hits every round**, so anything
rewarded per round (lesson 1's bandit, lesson 4's LinUCB with `--reward hit`) learns greedy,
and fails on `hidden`. Methods that see the *final* result (lessons 3, 5 and 4 with
`--reward rollout`) learn to explore early.

## How this connects to your project

- Your real benchmark (ENHITS) behaves like the **smooth** library. Your β experiments
  confirmed it: no β beats greedy by a meaningful margin (no-go, 2026-10-12).
- So in al-molecular, the decisions worth automating are about **cost**: retrain or not,
  cheap model or expensive model, stop early. See [ROADMAP.md](ROADMAP.md) for the project
  plan and [`../../career-plan/PLAN.md`](../../career-plan/PLAN.md) for the daily schedule.

## Habits

- Compare every controller to greedy, using the same seeds.
- Make it work in the toy or the simulator before spending GPU time.
- Log every agent decision with its inputs, so you can audit it later.
- Write 3 sentences in `notes.md` after each lesson.
