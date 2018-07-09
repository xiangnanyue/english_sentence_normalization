#!/usr/bin/env bash

#MODEL_NAME="context2vec.ukwac.model.package/context2vec.ukwac.model"
MODEL_NAME=$1
TRAIN_DATA=$2

python2.7 normalize.py --head=30 --output_dir=./output.txt --model_dir=$MODEL_NAME.params --train_dir=$TRAIN_DATA
