python prerender/prerender.py \
    --data-path ../../shared/data/2023_data/validation/ \
    --output-path ../../shared/data/hari_prerenders/validation/ \
    --n-jobs 24 \
    --n-shards 8 \
    --shard-id 0 \
    --config configs/prerender.yaml