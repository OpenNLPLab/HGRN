#!/usr/bin/env bash

export HYDRA_FULL_ERROR=1
export DATA_PATH=PATH_TO_LRA_DATA
program_path=PATH_TO_LRA_DIR

TASK=$1
ARCH=$2
BS=$3
lr=$4
N_LAYERS=$5
D_MODEL=$6
NORM=$7
PRENORM=$8
use_softmax=$9
act_fun=${10}
expand_ratio_ffn=${11}
num_heads=${12}
cards=${13}

python ${program_path}/train.py wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.gpus=$cards \
loader.batch_size=${BS} \
optimizer.lr=${lr} \
model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} \
model.norm=${NORM} model.prenorm=${PRENORM} train.seed=2222 \
model.use_softmax=${use_softmax} \
model.act_fun=${act_fun} \
model.expand_ratio_ffn=${expand_ratio_ffn} \
model.lg_local_heads=${num_heads}