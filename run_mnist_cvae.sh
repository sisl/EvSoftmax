#!/bin/bash

cd MNIST

python3 train.py --conditional
python3 train_classifier.py
python3 plot_bars.py
python3 plot_dists.py
