#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python translate.py --config config/config-translate-sparsemax-en-de.yml
spm_decode --model /scratch/$USER/checkpoints/en_de/sentencepiece.model < preds/en_de/sparsemax_sp.txt > preds/en_de/sparsemax.txt

sacrebleu data/wmt_en_de/test.de < preds/en_de/sparsemax.txt
