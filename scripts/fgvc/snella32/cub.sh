#!/usr/bin/env bash         

# hyperparameters
DATASET=cub
SEED=0
batch_size=16
LR=0.001
WEIGHT_DECAY=0.05
tuning_model=snella
low_rank_dim=32
init_thres=0
target_ratio=0.3
init_warmup=0
final_warmup=0
mask_interval=1
beta1=0.85
beta2=0.85


exp_name=fgvc_vit_supervised_${LR}_${init_thres}_${WEIGHT_DECAY}_${low_rank_dim}_${batch_size}

python train_alloc_ema.py --data-path=/data/shufan/shufan/SPT/data/fgvc/${DATASET} --init_thres=${init_thres} \
 --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_snell --resume=checkpoints/ViT-B_16.npz \
 --output_dir=./saves_snella_release/${tuning_model}/${DATASET}/${exp_name} \
 --batch-size=${batch_size} --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY} --mixup=0 --cutmix=0 \
 --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=10  --opt=adamw --low_rank_dim=${low_rank_dim} \
 --exp_name=${exp_name} --seed=0 \
 --test --block=BlockSNELLParallel  --tuning_model=${tuning_model} --freeze_stage \
 --init_warmup=${init_warmup} --final_warmup=${final_warmup} --mask_interval=${mask_interval} \
 --beta1=${beta1} --beta2=${beta2} --use_sparse_allocator --metric ipt --model-ema --model-ema-decay=0.9999 \
