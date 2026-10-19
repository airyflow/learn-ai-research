#!/bin/bash
#SBATCH -J mem_probe
#SBATCH -p h100-multi
#SBATCH --gpus-per-node h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH -A r00939
#SBATCH -o logs/mem_probe_%j.txt
#SBATCH -e logs/mem_probe_%j.err

# Usage (from learn-al/):  mkdir -p logs && sbatch 04_memory/run_gpu.sh
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
nvidia-smi
python 04_memory/memory_probe.py
