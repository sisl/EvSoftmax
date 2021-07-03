#!/usr/bin/env python

import argparse
import glob
from os.path import basename
import torch
import numpy as np
import tikzplotlib

# matplotlib.use('pgf')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.colors as colors


# matplotlib.rcParams["figure.constrained_layout.use"] = True

# plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# matplotlib.rcParams["text.latex.preamble"] = [
#     r"\usepackage[T1]{fontenc}",
#     r"\usepackage{times}",
#     r"\usepackage{amsmath}",
#     r"\usepackage{amssymb}",
# ]

# matplotlib.rcParams['font.family'] = 'serif'

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def config(path):
    return tuple(basename(path).split("-")[:2])


def load(path):
    with open(path) as f:
        return np.array([line.strip().split() for line in f], dtype=float)


def draw_square(ax, i, j, **kwargs):
    ax.add_patch(patches.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, **kwargs))


def draw_all_squares(ax, M):
    for ii in range(M.shape[0]):
        for jj in range(M.shape[1]):
            if M[ii, jj] > 0:
                draw_square(ax, jj, ii, color="#aaaaaa", lw=1, alpha=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--sentence_number", "-n", type=int, default=12)
    parser.add_argument("--fontsize", "-fontsize", type=int, default=10)
    parser.add_argument("--title", "-t", type=str, default=None)
    opt = parser.parse_args()

    attns_list = torch.load(opt.input)
    src = attns_list[opt.sentence_number]['src']
    pred = attns_list[opt.sentence_number]['pred']
    attns = attns_list[opt.sentence_number]['attns']
    attns = attns[:len(pred), :len(src)]
    # print(attns[0, 0])
    src = ["does", "that", "make", "you", "curious", "?"]
    # pred = ["Ist", "Machen", "macht"]
    pred = ["Softmax", "Sparsemax", "Entmax-1.5", "Evsoftmax"]
    attns = np.array(
    [
        [0.0021, 0.0022, 0.5265, 0.0218, 0.3208, 0.1266],
        [0.0000, 0.0000, 0.6952, 0.0000, 0.3048, 0.0000],
        [0.0000, 0.0000, 0.8271, 0.0000, 0.1711, 0.0017],
        [0.0031, 0.0052, 0.5473, 0.0194, 0.3403, 0.0847],
    ]
)
    # attns = np.power(attns, 0.25)

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    cmap = plt.cm.PuOr_r  # OrRd
    cax = ax.matshow(
        attns,
        cmap=cmap,
        clim=(-1, 1),
        norm=MidpointNormalize(midpoint=0, vmin=1, vmax=1),
    )
    # draw_all_squares(ax, attns)
    # fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(
        [""] + list(src), rotation=45, fontsize=opt.fontsize, horizontalalignment="left"
    )
    ax.set_yticklabels([""] + list(pred), fontsize=opt.fontsize)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.savefig("seachange.pdf")
    if opt.title:
        plt.title(opt.title, y=-0.15)
    plt.savefig(opt.output, bbox_inches='tight')
    # tikzplotlib.save(opt.output[:-3] + "tex", axis_width="5cm", textsize=4.0)


if __name__ == "__main__":
    main()
