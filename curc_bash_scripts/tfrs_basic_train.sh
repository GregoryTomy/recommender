#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=2:00:00
#SBATCH --partition=aa100
#SBATCH --output=tfrs_basic.%j.out
#SBATCH --job-name=tfrs_basic

module purge
module load anaconda
conda activate rec
python -m src.4_tfrs_basic