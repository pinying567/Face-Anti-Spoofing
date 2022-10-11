#!/bin/bash
# setting training config here
batch_size=64
epochs=30
lr=1e-3
step=10
gamma=0.1
save_dir="checkpoint/cnn_task1"
checkpoint=""

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --step ${step} --gamma ${gamma} --save_dir ${save_dir}"
#config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 main_cnn_task1.py ${config}"

echo "${run}"
${run}

