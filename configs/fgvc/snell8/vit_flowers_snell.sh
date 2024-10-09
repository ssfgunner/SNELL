#!/usr/bin/env bash         

# hyperparameters
DATASET=oxfordflower
SEED=0
batch_size=32
LR=0.001
WEIGHT_DECAY=0.01
tuning_model=snell
low_rank_dim=8
init_thres=0.6


exp_name=fgvc_vit_supervised_${LR}_${init_thres}_${WEIGHT_DECAY}_${low_rank_dim}_${batch_size}

python train.py --data-path=./data/fgvc/${DATASET} --init_thres=${init_thres} \
 --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_snell --resume=checkpoints/ViT-B_16.npz \
 --output_dir=./saves_fgvc/${tuning_model}/${DATASET}/${exp_name} \
 --batch-size=${batch_size} --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY} --mixup=0 --cutmix=0 \
 --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=10  --opt=adamw --low_rank_dim=${low_rank_dim} \
 --exp_name=${exp_name} --seed=0 \
 --test --block=BlockSNELLParallel  --tuning_model=${tuning_model} --freeze_stage