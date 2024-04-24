#! /usr/bin/env bash

gpus=2
batch_size=$(( 2048 / $gpus ))
# batch_size=32 # for test
arch=hgrn1d_vit_tiny
# arch=hgrn1d_vit_small
output_name=${arch}
lr=0.0005
dataset=IMNET
dataset=CIFAR # for test
weight_decay=0.05
warmup_epochs=10
clip_grad=5
code_dir=im/classification/main.py
data_dir=path_to_data_dir

echo $code_dir

bash train_im.sh $gpus $batch_size $arch $output_name $lr $dataset $weight_decay $warmup_epochs $clip_grad $code_dir $data_dir