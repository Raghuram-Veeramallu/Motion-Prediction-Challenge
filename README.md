# Motion Prediction for Self Driving cars

To replicate this:
```
1. Pull the code
2. Create a conda environment with the `requirements.txt` file.
3. Download the dataset from https://waymo.com/open/data/motion/
4. Run the prerendering script
5. Run the training script and prediction script
```

### Prerendering
```
python prerender/prerender.py \
    --data-path <path-to-data-files> \
    --output-path <path-to-store-output-renders> \
    --n-jobs 16 \
    --n-shards 4 \
    --shard-id 0 \
    --config configs/prerender.yaml
```

### Training
```
python train.py configs/run_config.yaml
```

### Predict/Submit
```
python predict_v3_1.py configs/run_config.yaml
```
