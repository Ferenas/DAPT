#!/bin/bash

#cd ../..

# custom config
DATA="/home/ubuntu/Data_file/few_shot_data"   #Your data path
TRAINER=MaPLe

DATASET=$1    #10 dataset from, like "stanford_cars"
CFG=vit_b16_t
MODE=dapt-g

#seed = ("0" "1" "2")
#method "Uniform"    "Uncertainty" "Herding" "Submodular" "Glister" "GraNd" "Craig" "Cal"    "Forgetting"
#sample_rate = 0.05 0.1 0.2 0.3 0.5 1.0  #"Uncertainty" "Herding"
# Normally method == "Uniform" and sample_rate = "1.0", which means no data selection process

for seed in 1 2 3; do
  for rate in 1.0; do   #
    for shot in 1 2 4 8 16; do
      for method in "Uniform"; do
        echo "Running with seed =${seed} and sample rate=${rate} and Method=${method}"
        CUDA_VISIBLE_DEVICES=1 python train.py \
        --root ${DATA} \
        --seed ${seed} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir output_few \
        --mode ${MODE} \
        DATASET.NUM_SHOTS ${shot} \
        DATASET.SUBSAMPLE_CLASSES all \
        DATASET.SELECTION_RATIO ${rate} \
        DATASET.SELECTION_METHOD ${method}
        done
    done
  done
done
