#!/bin/bash
model=resnet_vision
depth=18
dataset=cifar10
epochs=50
lr=1e-1

exp=${model}${depth}_${dataset}_${epochs}e_${lr}
path=${dataset}/${model}
EXP_DIR=/mnt/lustre21/wangjiong/classification_school/playground/exp
# mkdir -p ${EXP_DIR}/${path}/${exp}/model
mkdir -p ./model
now=$(date +"%Y%m%d_%H%M%S")

part=Pixel
numGPU=1
nodeGPU=1
# pg-run -rn ${exp} -c "
# srun -p ${part} --job-name=${exp} --gres=gpu:${nodeGPU} -n ${numGPU} --ntasks-per-node=${nodeGPU} \
python -u cifar.py \
  --dataset ${dataset} \
  --data_dir ./data \
  --workers 4 \
  --epochs 5 \
  --start-epoch 0 \
  --train-batch 256 \
  --test-batch 200 \
  \
  --lr ${lr} \
  --drop 0 \
  --schedule 10 20 \
  --weight-decay 1e-4 \
  --checkpoint ./model \
  \
  --arch ${model} \
  --depth ${depth} \
  --cardinality 32 \
  --widen-factor 4 \
  \
  --manualSeed 345 \
  --gpu-id 0 \
  2>&1 | tee -a ./model/train-${now}.log \
  & \
#   "
