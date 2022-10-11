#!/bin/bash
# setting training config here
batch_size=8
epochs=90
lr=1e-4
step=30
gamma=0.1
save_dir="checkpoint/cnn-rnn_task2"
checkpoint="checkpoint/cnn_task2/last_checkpoint.pkl"

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --step ${step} --gamma ${gamma} --save_dir ${save_dir}"
config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 main_task2.py ${config}"

echo "${run}"
${run}

