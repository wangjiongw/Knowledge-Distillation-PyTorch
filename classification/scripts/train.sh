#!/bin/bash
model=resnet
depth=110
dataset=cifar10
epochs=220
lr=1e-1

exp=${model}${depth}_${dataset}_${epochs}e_${lr}
path=${dataset}/${model}
EXP_DIR=/mnt/lustre21/wangjiong/classification_school/playground/exp
mkdir -p ${EXP_DIR}/${path}/${exp}/model
now=$(date +"%Y%m%d_%H%M%S")

part=Pixel
numGPU=1
nodeGPU=1
pg-run -rn ${path}/${exp} -c "
srun -p ${part} --job-name=${exp} --gres=gpu:${nodeGPU} -n ${numGPU} --ntasks-per-node=${nodeGPU} \
python -u cifar.py \
  --dataset ${dataset} \
  --data_dir /mnt/lustre21/wangjiong/Data_t1/datasets/cifar/${dataset} \
  --workers 4 \
  --reset \
  --epochs ${epochs} \
  --start-epoch 0 \
  --train-batch 256 \
  --test-batch 200 \
  \
  --lr ${lr} \
  --drop 0 \
  --schedule 100 150 \
  --gamma 0.1 \
  --weight-decay 1e-4 \
  --checkpoint ${EXP_DIR}/${path}/${exp}/model \
  \
  --arch ${model} \
  --depth ${depth} \
  \
  --gpu-id 0 \
  2>&1 | tee -a ${EXP_DIR}/${path}/${exp}/train-${now}.log \
  & \
  "
