#!/bin/bash
# Module 06 with a local LLM: serve gpt-oss-20b with vLLM on one H100, then run the agent
# controller against it. Free (no API cost); uses about 1 GPU-hour of your allocation.
# Every environment trick below is copied from NeuroPathlogy/slurm/run_neuro_vllm.slurm, where
# it was debugged -- see the comments there for why each one is needed.
#
#   cd /N/slate/mengjing/repos/learn-ai-research/learn-agent-rl && mkdir -p logs && sbatch 06_agents/run_vllm_agent.slurm
#
#SBATCH -J vllm_agent
#SBATCH -o /N/slate/mengjing/repos/learn-ai-research/learn-agent-rl/logs/vllm_agent_%j.out
#SBATCH -e /N/slate/mengjing/repos/learn-ai-research/learn-agent-rl/logs/vllm_agent_%j.err
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -A r00939

COURSE=/N/slate/mengjing/repos/learn-ai-research/learn-agent-rl
NEURO=/N/slate/mengjing/repos/NeuroPathlogy
MODEL="openai/gpt-oss-20b"
SEEDS=${SEEDS:-5}                   # campaigns per regime: sbatch --export=ALL,SEEDS=10 ...
EFFORT=${EFFORT:-low}               # gpt-oss reasoning effort: low | medium | high
MODE=${MODE:-native}                # native tool calls | json (tools described in the prompt)

set -euo pipefail
set +u                              # conda's activate hooks are not `set -u` safe
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate neuro                # has vllm, openai, torch, numpy, pydantic
set -u

# Copy the env to RAM: importing torch/vLLM from Lustre can take 30-90 min otherwise.
SHM_ENV=/dev/shm/${USER}_envs/neuro
if [ ! -d "$SHM_ENV" ]; then
    mkdir -p "$(dirname "$SHM_ENV")"
    cp -a "$(dirname "$(dirname "$(which python3.12)")")" "$SHM_ENV"
fi
PYBIN=$SHM_ENV/bin/python3.12
VLLM_BIN=$SHM_ENV/bin/vllm
export LD_LIBRARY_PATH=$SHM_ENV/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/N/project/ADRD/neuropathoroot/cache_hf HF_HUB_OFFLINE=1   # cached gpt-oss-20b weights
export VLLM_USE_FLASHINFER_SAMPLER=0

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "[vllm] starting at $(date) on port $PORT"
# --enable-auto-tool-choice --tool-call-parser openai: lets gpt-oss make real tool calls.
# Without them (first run, job 11915188) every reply came back empty and the agent fell back
# to greedy in 60 of 60 rounds.
"$PYBIN" "$VLLM_BIN" serve "$MODEL" --config "$NEURO/config/GPT-OSS_Hopper.yaml" --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser openai \
    > "$COURSE/logs/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 720); do           # cold starts have taken 10-45 min on this cluster
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[vllm] ready after $((i * 5)) s"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[error] vLLM died during startup"; tail -50 "$COURSE/logs/vllm_server_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 5
done

cd "$COURSE"
"$PYBIN" 06_agents/agent_controller.py --backend vllm --vllm-url "http://localhost:$PORT" \
    --vllm-model "$MODEL" --vllm-mode "$MODE" --effort "$EFFORT" --seeds "$SEEDS" \
    --log 06_agents/decisions_vllm.jsonl
# Check the run's reliability, not just its recall (lesson 8):
"$PYBIN" 08_agent_evals/step3_basic_metrics.py 06_agents/decisions_vllm.jsonl
echo "[done] $(date)"
