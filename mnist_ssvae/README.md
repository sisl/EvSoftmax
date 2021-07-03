Based on code obtained from [https://github.com/deep-spin/sparse-marginalization-lvm](https://github.com/deep-spin/sparse-marginalization-lvm)

## Python requirements and installation

This code was tested on `Python 3.7.1`. To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen) repository to your home: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/deep-spin/lp-sparsemap) repository to your home, and follow the installation instructions found there
4. Install PyTorch: `pip install torch` (we used version 1.6.0)
5. Install the requirements: `pip install -r requirements.txt`
6. Install the `lvm-helpers` package: `pip install .` (or in editable mode if you want to make changes: `pip install -e .`)

## Datasets

MNIST should be downloaded automatically by running the training commands for the first time on the semi-supervised VAE. 

## Running

**Training**:

To get a warm start for the semi-supervised VAE experiment (use `softmax` normalizer for all experiments that do not use sparsemax):

```
python  experiments/semi_supervised-vae/train.py \
    --n_epochs 100 \
    --lr 1e-3 \
    --labeled_only \
    --normalizer sparsemax \
    --batch_size 64
```

To train with sparsemax on the semi-supervised VAE experiment (after getting a warm start checkpoint):

```
python experiments/semi_supervised-vae/train.py \
    --mode marg \
    --normalizer sparsemax \
    --random_seed 42 \
    --lr 5e-4 \
    --batch_size 64 \
    --n_epochs 200 \
    --latent_size 10 \
    --warm_start_path /path/to/warm_start/
```

**Evaluating**:

To evaluate any trained network against one of the test sets, run:

```
python experiments/semi_supervised-vae/test.py /path/to/checkpoint/ /path/to/hparams.yaml
```

Checkpoints should be found in the appropriate folder inside the automatically generated `checkpoints` directory, and the `yaml` file should be found in the model's automatically generated directory inside `logs`.

The evaluation results should match the paper.
