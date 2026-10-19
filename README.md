# learn-ai-research

Three hands-on courses, each taught through a real research project rather than toy
examples. Every lesson is a small script or notebook that runs in seconds to minutes on a
CPU and ends with exercises.

| # | Course | What you learn | Taught through |
|---|---|---|---|
| 1 | [learn-al](learn-al/README.md) | Python, debugging, PyTorch, memory, distributed training | `al-molecular`: active learning over millions of molecules |
| 2 | [learn-agent-rl](learn-agent-rl/README.md) | reinforcement learning and LLM agents | when an active-learning campaign should explore vs exploit (`al-molecular/adaptive_acq`) |
| 3 | [learn-llm-eval](learn-llm-eval/README.md) | evaluating LLM systems: metrics, calibration, bootstrap, LLM judges | the NeuroPathology report-extraction project, on **fake data only** |

New to Python? Start with
[learn-al/00_how_al_molecular_works/python_from_zero.ipynb](learn-al/00_how_al_molecular_works/python_from_zero.ipynb).

## Setup

```bash
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /N/slate/mengjing/repos/learn-ai-research/<course>
```

## What this repo depends on

The courses expect the projects they teach to sit **next to this repo**, in the same
parent folder:

```
repos/
├── learn-ai-research/   (this repo)
├── al-molecular/        used by learn-al and learn-agent-rl
└── NeuroPathlogy/       linked from learn-llm-eval's last lesson
```

* The `learn-al` scripts find `al-molecular` relative to themselves, and the notebooks use
  the absolute path `/N/slate/mengjing/repos/al-molecular`. On another machine, edit
  `AL_REPO` at the top of each notebook.
* A few lessons read experiment results from `al-molecular/runs/`. That folder isn't in
  al-molecular's git repo, so those cells only work where the runs exist (the IU cluster).
* `learn-llm-eval` uses only generated fake reports (`learn-llm-eval/common/fake_reports.py`).
  **Never copy real NACC reports or values into this repo.**
