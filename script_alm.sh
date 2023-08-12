#! /usr/bin/bash

arch=transnormer_t1
# arch=transnormer_t2
# change to your data dir
data_dir=/cpfs01/user/zhongyiran/data/qinzhen/qinzhen_nlp_exp_data/lm-wikitext-103

bash train_alm.sh 2 $arch $data_dir