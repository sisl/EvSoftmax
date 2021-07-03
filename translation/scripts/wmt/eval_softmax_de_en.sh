#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python translate.py --config config/config-translate-iwslt-softmax-en-de.yml

sacrebleu data/iwslt14.tokenized.de-en/test.en < preds/de_en/softmax.txt
