#!/bin/bash

N=1559

# CUDA_VISIBLE_DEVICES=2 python translate.py --config config/translate/en_de/iwslt/softmax.yml
python plot_attn.py -i preds/en_de/iwslt/attn_out/softmax_de_en.pt -o softmax_attn.png -n $N --title Softmax

# CUDA_VISIBLE_DEVICES=2 python translate.py --config config/translate/en_de/iwslt/evsoftmax.yml
python plot_attn.py -i preds/en_de/iwslt/attn_out/evsoftmax_de_en.pt -o evsoftmax_attn.png -n $N --title Ev-softmax

# CUDA_VISIBLE_DEVICES=2 python translate.py --config config/translate/en_de/iwslt/sparsemax.yml
python plot_attn.py -i preds/en_de/iwslt/attn_out/sparsemax_de_en.pt -o sparsemax_attn.png -n $N --title Sparsemax

# CUDA_VISIBLE_DEVICES=2 python translate.py --config config/translate/en_de/iwslt/tsallis15.yml
python plot_attn.py -i preds/en_de/iwslt/attn_out/tsallis15_de_en.pt -o tsallis_attn.png -n $N --title Entmax-1.5