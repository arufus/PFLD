#!/usr/bin/env bash
save_model=./models2/model4
pre_model=
logs=./models2/log4.txt
lr=0.001

CUDA_VISIBLE_DEVICES='0' \
nohup python -u train_model.py --model_dir=${save_model} \
                               --pretrained_model=${pre_model} \
                               --learning_rate=${lr} \
                               --level=L1 \
                               --debug=False \
                               --image_size=112 \
                               --batch_size=2 \
                               > ${logs} 2>&1 &
tail -f ${logs}
