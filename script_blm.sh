#! /usr/bin/bash

arch=roberta_transnormer_t1
# arch=roberta_transnormer_t2
# change to your data dir
data_dir=path_to_bin_data

bash train_blm.sh 8 $arch $data_dir