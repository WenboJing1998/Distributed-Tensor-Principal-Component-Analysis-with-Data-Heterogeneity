#!/bin/bash
#SBATCH --job-name=ten_hom
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=50:00:00
#SBATCH --mem=50GB
#SBATCH --array=1-120
#SBATCH --output=/scratch/wj2093/tensor/homo.out
#SBATCH --error=/scratch/wj2093/tensor/homo.err


module purge
module load python/intel/3.8.6

cd /scratch/wj2093/tensor

python homo_hpc.py