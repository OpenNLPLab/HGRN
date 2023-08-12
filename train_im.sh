#!/usr/bin/env bash

export NCCL_LL_THRESHOLD=0

GPUS=$1
BATCH_SIZE=$2
ARCH=$3
PORT=$(( $RANDOM + 2000 ))
export MASTER_PORT=${MASTER_PORT:-$PORT}

OUTPUT_DIR=./checkpoints/$4
RESUME=./checkpoints/$4/checkpoint_best.pth

LR=$5
DATASET=$6
WEIGHT_DECAY=$7
WARMUP_EPOCHS=$8
CLIP_GRAD=$9

PROG=${10}
DATA=${11}

echo GPUS $GPUS
echo BATCH_SIZE $BATCH_SIZE
echo ARCH $ARCH
echo OUTPUT_NAME $4
echo LR $LR
echo DATASET $DATASET
echo WEIGHT_DECAY $WEIGHT_DECAY
echo WARMUP_EPOCHS $WARMUP_EPOCHS
echo CLIP_GRAD $CLIP_GRAD
echo PROG $PROG
echo DATA $DATA

mkdir -p log

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env $PROG \
    --data-set $DATASET --data-path $DATA \
    --batch-size $BATCH_SIZE --dist-eval --output_dir $OUTPUT_DIR \
    --resume $RESUME --model $ARCH --epochs 300 --fp32-resume --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-epochs $WARMUP_EPOCHS \
    --clip-grad $CLIP_GRAD \
    --broadcast_buffers \
    2>&1 | tee log/${ARCH}.log
