"""This file computes the qualitative observations for the bar plots."""

import numpy as np
import matplotlib.pyplot as plt

import tikzplotlib

import torch

from dst_utils import *
from dst import *

NORMALIZERS = ["softmax", "sparsemax", "tsallis15", "evsoftmax"]
COLORS = ["orange", "red", "blue", "green"]
IND = [
    [2, 8, 1, 4, 6, 0, 3, 9, 5, 7],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 9, 7, 8, 6],
    [0, 6, 8, 7, 2, 4, 3, 1, 5, 9],
]

# From: https://github.com/sebastianruder/learn-to-select-data
def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = -1.0 * np.log(np.sum([np.sqrt(p * q)]))
    assert not np.isnan(sim), "Error: Similarity is nan."
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim


# Load the data and perform the DST transformation
import pdb
def main():

    random = 1235

    tracker_global_train = torch.load("tracker_mnist_SGD_random_" + str(random) + ".pt")
    K = 10

    # pdb.set_trace()
    probs_even = [
        tracker_global_train[normalizer]["alpha_p"].data.cpu().numpy()[0, :, -2]
        for normalizer in NORMALIZERS
    ]
    probs_odd = [
        tracker_global_train[normalizer]["alpha_p"].data.cpu().numpy()[1, :, -2]
        for normalizer in NORMALIZERS
    ]

    width = 0.15
    for num in range(2):
        if num == 1:
            plt.figure()
            for i in range(4):
                plt.bar(
                    np.arange(K) + width * (i - 1.5),
                    probs_odd[i][IND[i]],
                    width,
                    color=COLORS[i],
                    alpha=0.5,
                )

            plt.xlabel("Z")
            plt.ylim(0, 1.0)
            plt.title("Odd")
            plt.xticks([])
            plt.savefig(f"odd_random_{random}.png", dpi=600)
            tikzplotlib.save(f"odd_random_{random}.tex", textsize=5.0, axis_width="6cm", strict=True)
            plt.close()

        if num == 0:
            plt.figure()
            for i in range(4):
                plt.bar(
                    np.arange(K) + width * (i - 1.5),
                    probs_even[i][IND[i]],
                    width,
                    color=COLORS[i],
                    alpha=0.5,
                )

            plt.xlabel("Z")
            plt.ylabel("Probability")
            plt.ylim(0, 1.0)
            plt.title("Even")
            plt.xticks([])
            plt.savefig(f"even_random_{random}.png", dpi=600)
            tikzplotlib.save(f"even_random_{random}.tex", textsize=5.0, axis_width="6cm", strict=True)
            plt.close()


if __name__ == "__main__":
    main()
