#!/bin/bash

BATCH_SIZE=8
TOKENS_PER_SAMPLE=512
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
prefix=lm
MAX_UPDATE=100000
WARM_UP=8000
PORT=$(( $RANDOM + 2000 ))
LR=0.0005
CLIP_NORM=0
GPUS=$1
ARCH=$2
DATA_DIR=$3
UPDATE_FREQ=$(( 128 / $BATCH_SIZE / $GPUS ))

echo $PORT

fairseq-train --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints/$prefix/$ARCH \
    --distributed-world-size $GPUS  \
    --arch $ARCH --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm $CLIP_NORM \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-07 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
    --batch-size $BATCH_SIZE \
    --max-update $MAX_UPDATE --log-interval 1  2>&1 | tee $ARCH.log