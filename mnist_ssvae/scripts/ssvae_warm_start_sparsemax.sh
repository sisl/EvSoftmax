#!/bin/bash

# python33  experiments/semi_supervised-vae/train.py \
#             --n_epochs 100 \
#             --lr 1e-3 \
#             --labeled_only \
#             --normalizer entmax \
#             --batch_size 64

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 42 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 43 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 44 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 45 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 46 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 47 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 48 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 49 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 50 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt

python3 experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 51 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path checkpoints/ssvae/warm_start/entmax15/lr-0.001_baseline-runavg/version_0/checkpoints/epoch=89.ckpt
