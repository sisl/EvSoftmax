#!/bin/bash

cd vqvae

python VQVAE/vqvae.py --data-folder data/tiny-imagenet-200 --output-folder vqvae --dataset tinyimagenet --device cuda

python VQVAE/pixelcnn_prior.py --data-folder data/tiny-imagenet-200 --model VQVAE/models/vqvae/best.pt --output-folder pixelcnn_prior --dataset tinyimagenet --device cuda --batch-size 32 --num-layers 20 --hidden-size-prior 128

python VQVAE/generated_dataset.py --data-folder data/tiny-imagenet-200 --model VQVAE/models/vqvae/best.pt --output-folder pixelcnn_prior_2 --regularization 1e-4 --lr 1e-5 --dataset tinyimagenet --device cuda --batch-size 32 --num-layers 10 --hidden-size-prior 128 --normalization esoftmax --prior models/pixelcnn_prior_1_esoftmax_reg_0.0001_lr_1e-05/prior.pt

python VQVAE/generated_dataset.py --data-folder data/tiny-imagenet-200 --model VQVAE/models/vqvae/best.pt --output-folder pixelcnn_prior_2 --regularization 1e-4 --lr 1e-5 --dataset tinyimagenet --device cuda --batch-size 32 --num-layers 10 --hidden-size-prior 128 --normalization softmax --prior models/pixelcnn_prior_softmax_reg_0.0001_lr_1e-05/prior.pt

python VQVAE/generated_dataset.py --data-folder data/tiny-imagenet-200 --model VQVAE/models/vqvae/best.pt --output-folder pixelcnn_prior_2 --regularization 1e-4 --lr 1e-6 --dataset tinyimagenet --device cuda --batch-size 32 --num-layers 10 --hidden-size-prior 128 --normalization sparsemax --prior models/pixelcnn_prior_1_sparsemax_reg_0.0001_lr_1e-06/prior.pt

python VQVAE/generated_dataset.py --data-folder data/tiny-imagenet-200 --model VQVAE/models/vqvae/best.pt --output-folder pixelcnn_prior_2 --regularization 1e-4 --lr 1e-6 --dataset tinyimagenet --device cuda --batch-size 32 --num-layers 10 --hidden-size-prior 128 --normalization tsallis15 --prior models/pixelcnn_prior_1_tsallis15_reg_0.0001_lr_1e-06/prior.pt

python classifier/test.py --data_dir data/pixelcnn_prior_2_softmax_reg_0.0001_lr_1e-05/dataset/

python classifier/test.py --data_dir data/pixelcnn_prior_2_evsoftmax_reg_0.0001_lr_1e-05/dataset/

python classifier/test.py --data_dir data/pixelcnn_prior_2_tsallis15_reg_0.0001_lr_1e-06/dataset/

python classifier/test.py --data_dir data/pixelcnn_prior_2_sparsemax_reg_0.0001_lr_1e-06/dataset/


