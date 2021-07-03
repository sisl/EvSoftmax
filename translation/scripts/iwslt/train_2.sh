#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py --config config/train/en_de/iwslt/sparsemax.yml --seed 1236
CUDA_VISIBLE_DEVICES=2 python train.py --config config/train/en_de/iwslt/tsallis15.yml --seed 1236
CUDA_VISIBLE_DEVICES=2 python train.py --config config/train/en_de/iwslt/softmax.yml --seed 1236
CUDA_VISIBLE_DEVICES=2 python train.py --config config/train/en_de/iwslt/evsoftmax.yml --seed 1236
