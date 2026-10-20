# learn-ai-research

Three hands-on courses, each taught through a real research project rather than toy
examples. Every lesson is a small script or notebook that runs in seconds to minutes on a
CPU and ends with exercises.

| # | Course | What you learn | Taught through |
|---|---|---|---|
| 1 | [learn-al](learn-al/README.md) | Python, debugging, PyTorch, memory, distributed training | `al-molecular`: active learning over millions of molecules |
| 2 | [learn-agent-rl](learn-agent-rl/README.md) | reinforcement learning and LLM agents | when an active-learning campaign should explore vs exploit (`al-molecular/adaptive_acq`) |
| 3 | [learn-llm-eval](learn-llm-eval/README.md) | evaluating LLM systems: metrics, calibration, bootstrap, LLM judges | the NeuroPathology report-extraction project, on **fake data only** |

New to Python and machine learning? Start with **[learn-al/START_HERE.md](learn-al/START_HERE.md)**.
It takes you from zero: a setup check (which also builds the small real dataset into
`data/`, not in git), Python basics, then the
[beginner tour](learn-al/beginner_tour/README.md), 9 notebooks that walk through the whole
`al-molecular` project on real data, with worked answers.

## Setup

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/<course>
```

## What this repo depends on

The courses teach a **frozen copy** of the `al-molecular` research code, not the live repo.
Research code changes all the time (on 2026-10-19 one commit moved most of
`run_experiment.py` down by 130 lines), and the lessons quote it line by line. So they're
pinned to one version:

```
repos/
├── learn-ai-research/        (this repo)
├── al-molecular/             the live research repo: change it freely, the courses won't break
├── al-molecular-tutorial/    a git worktree of al-molecular at tag `tutorial-2026-10-20`
│   └── runs/                 the few experiment results the lessons read (copied in, git-ignored)
└── NeuroPathlogy/            linked from learn-llm-eval's last lesson
```

* **Set it up** (once, from `repos/al-molecular`):
  `git worktree add --detach ../al-molecular-tutorial tutorial-2026-10-20`, then copy
  `history.json` and `all_explored_final.csv` from `runs/fusionfix_*`,
  `runs/fusion_greedy_frac0.001` and `runs/fusion_ucb_frac0.001` into the copy's `runs/`.
* **Move the courses to newer code:** in the copy, `git checkout <new tag>`, then re-run every
  notebook and fix any line references that moved.
* Scripts find the copy relative to themselves; notebooks use the absolute path
  `/N/slate/mengjing/repos/al-molecular-tutorial` (`AL_REPO`). On another machine, edit those.
* The big data (the 2.1M-molecule embeddings on `/N/project`) isn't copied. Lessons read it
  in place, so the beginner tour only runs on the IU cluster.
* `learn-llm-eval` uses only generated fake reports (`learn-llm-eval/common/fake_reports.py`).
  **Never copy real NACC reports or values into this repo.**
