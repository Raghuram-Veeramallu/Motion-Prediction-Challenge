#!/bin/bash -l        
#SBATCH --time=10:00:00
#SBATCH --ntasks=24
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --gres=gpu:v100:6
#SBATCH -p v100
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=veera047@umn.edu
cd /home/csci5525/veera047/baseline
module load conda
conda activate waymo
python train.py configs/run_config.yaml
