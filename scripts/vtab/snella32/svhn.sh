# hyperparameters
SEED=0
batch_size=16
LR=0.001
WEIGHT_DECAY=0.0001
tuning_model=mixk
low_rank_dim=32

DATASET=svhn

exp_name=vtab_vit_supervised_${LR}_${init_thres}_${WEIGHT_DECAY}_${low_rank_dim}_${batch_size}


python train.py --data-path=/data/shufan/shufan/SPT/data/vtab-1k/${DATASET} \
 --data-set=${DATASET} --model_name=vit_base_patch16_224_in21k_snell --resume=checkpoints/ViT-B_16.npz \
 --output_dir=./saves_vtab_snella_release/${tuning_model}/${DATASET}/${exp_name} \
 --batch-size=${batch_size} --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 --direct_resize \
 --smoothing=0 --launcher="none" --seed=${SEED} --val_interval=10  --opt=adamw --low_rank_dim=${low_rank_dim} \
 --exp_name=${exp_name} --seed=0 \
 --test --block=BlockSNELLParallel  --tuning_model=${tuning_model} --freeze_stage