# Evidential Softmax for Sparse Multimodal Distributions in Deep Generative Models

## Abstract
Many applications of generative models rely on the marginalization of their high-dimensional output probability distributions. Normalization functions that yield sparse probability distributions can make exact marginalization more computationally tractable.  However, sparse normalization functions usually require alternative loss functions for training because the log-likelihood can be undefined for sparse probability distributions. Furthermore, many sparse normalization functions often collapse the multimodality of distributions.  In this work, we present *ev-softmax*, a sparse normalization function that preserves the multimodality of probability distributions. We derive its properties, including its gradient in closed-form, and introduce a continuous family of approximations to *ev-softmax* that have full support and can thus be trained with probabilistic loss functions such as negative log-likelihood and Kullback-Leibler divergence.  We evaluate our method on a variety of generative models, including variational autoencoders and auto-regressive models. Our method outperforms existing dense and sparse normalization techniques in distributional accuracy and classification performance. We demonstrate that *ev-softmax* successfully reduces the dimensionality of output probability distributions while maintaining multimodality.

## Setup
Required packages are listed in `requirements.txt`.

## Running
The implementation for the ev-softmax function and its loss function can be found in `evsoftmax.py`.

The MNIST CVAE and VQ-VAE experiments can be run using `run_mnist_cvae.sh` and `run_vqvae.sh`, respectively. Instructions for the SSVAE experiment can be found in `mnist_ssvae/README.md`, and scripts used for preprocessing, training, and evaluating can be found in `mnist_ssvae/scripts`. Instructions for the translation experiment can be found in `translation/README.md`, and scripts used for preprocessing, training, and evaluating can be found in `translation/scripts/iwslt`.
