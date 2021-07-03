#!/bin/bash

CONFIG=config/translate/en_de/iwslt/softmax.yml
SP=/scratch/$USER/checkpoints/en_de/sentencepiece.model
LC=mosesdecoder/scripts/tokenizer/lowercase.perl

PRED=preds/en_de/iwslt/softmax.txt
PRED_SP=preds/en_de/iwslt/softmax_sp.txt

CUDA_VISIBLE_DEVICES=2 python translate.py --config $CONFIG
spm_decode --model $SP < $PRED_SP | $LC > $PRED

sacrebleu data/iwslt14.tokenized.de-en/test.de < $PRED
