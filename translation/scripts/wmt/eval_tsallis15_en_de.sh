#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python translate.py --config config/config-translate-tsallis15-en-de.yml
spm_decode --model /scratch/$USER/checkpoints/en_de/sentencepiece.model < preds/en_de/tsallis15_sp.txt > preds/en_de/tsallis15.txt

sacrebleu data/wmt_en_de/test.de < preds/en_de/tsallis15.txt
