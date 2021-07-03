#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python translate.py --config config/config-translate-softmax-en-de.yml
spm_decode --model /scratch/$USER/checkpoints/en_de/sentencepiece.model < preds/en_de/softmax_sp.txt > preds/en_de/softmax.txt

sacrebleu data/wmt_en_de/test.de < preds/en_de/softmax.txt
