#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --ntasks=16
#SBATCH --mem=10g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=veera047@umn.edu 
cd ~/baseline
module load conda
conda activate waymo
python prerender/prerender.py \
    --data-path ../../shared/data/2023_data/testing/ \
    --output-path ../../shared/data/hari_prerenders/new/testing/ \
    --n-jobs 16 \
    --n-shards 4 \
    --shard-id 0 \
    --config configs/prerender.yaml
