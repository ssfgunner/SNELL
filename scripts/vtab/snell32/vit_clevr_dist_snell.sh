#!/usr/bin/env bash         

# hyperparameters
DATASET=clevr_dist
SEED=0
batch_size=16
LR=0.001
WEIGHT_DECAY=0.001
tuning_model=snell
low_rank_dim=32
init_thres=0.74

exp_name=vtab_vit_supervised_${LR}_${init_thres}_${WEIGHT_DECAY}_${low_rank_dim}_${batch_size}

python train.py --data-path=./data/vtab-1k/${DATASET} --init_thres=${init_thres} \
 --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_snell --resume=checkpoints/ViT-B_16.npz \
 --output_dir=./saves_vtab/${tuning_model}/${DATASET}/${exp_name} \
 --batch-size=${batch_size} --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 --direct_resize \
 --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=10  --opt=adamw --low_rank_dim=${low_rank_dim} \
 --exp_name=${exp_name} --seed=0 \
 --test --block=BlockSNELLParallel  --tuning_model=${tuning_model} --freeze_stage