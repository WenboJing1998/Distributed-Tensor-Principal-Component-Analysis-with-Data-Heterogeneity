#!/bin/bash
#SBATCH --job-name=ten_het
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=50:00:00
#SBATCH --mem=50GB
#SBATCH --array=1-120
#SBATCH --output=/scratch/wj2093/tensor/hetero.out
#SBATCH --error=/scratch/wj2093/tensor/hetero.err


module purge
module load python/intel/3.8.6

cd /scratch/wj2093/tensor

python hetero_hpc_same_core.py