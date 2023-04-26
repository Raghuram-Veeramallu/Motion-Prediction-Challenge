#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH --ntasks=24
#SBATCH --mem=15g
#SBATCH --tmp=15g
#SBATCH --gres=gpu:v100:6
#SBATCH -p v100
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=veera047@umn.edu
cd ~/baseline
module load conda
conda activate waymo
python train.py configs/run_config.yaml
