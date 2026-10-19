#!/bin/bash
#SBATCH -J ddp_al
#SBATCH -p h100-multi
#SBATCH --gpus-per-node h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH -A r00939
#SBATCH -o logs/ddp_al_%j.txt
#SBATCH -e logs/ddp_al_%j.err

# Usage (from learn-al/):  mkdir -p logs && sbatch 05_distributed/run_ddp.sh
# One Slurm task; torchrun then starts 4 processes, one per GPU.
# For multi-node: --nodes=2, then run torchrun under srun with
#   --nnodes=$SLURM_NNODES --rdzv_backend=c10d --rdzv_endpoint=$(hostname):29500
# (al-molecular/slurm/test_multinode_gpu.sh checks whether this cluster allows it).
source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 05_distributed/ddp_al_iteration.py
