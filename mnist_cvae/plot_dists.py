import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import scipy.stats
import torch

import tikzplotlib
from models import CNN

# From: https://github.com/sebastianruder/learn-to-select-data
def bhattacharyya_distance(p, q):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = -1.0 * np.log(np.sum([np.sqrt(p * q)]))
    assert not np.isnan(sim), "Error: Similarity is nan."
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim

DIST_DICT = {
    # "KL divergence": scipy.special.kl_div,
    "Wasserstein distance": scipy.stats.wasserstein_distance,
    # "Bhattacharyya distance": bhattacharyya_distance,
}

GROUND_TRUTH = np.array(
    [
        [0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0],
        [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2],
    ]
)
NORMALIZERS = ["softmax", "sparsemax", "tsallis15", "evsoftmax"]
COLORS = ["orange", "red", "blue", "green"]

def calculate_divergences(model, tracker_global_train):
    data = {}
    for divergence, div_func in DIST_DICT.items():
        data[divergence] = {}
        for normalizer in NORMALIZERS:
            tracker = tracker_global_train[normalizer]
            x = 256 * tracker["x"] # K x B x HW
            alpha_p = tracker["alpha_p"].unsqueeze(-1) # 2 x K x B x 1
            x = x.transpose(1, 2).reshape(-1, 1, 28, 28) # KB x 1 x H x W
            logits = model(x)
            probs = torch.exp(logits).reshape(1, 10, -1, 10)
            marg = (probs * alpha_p).sum(axis=1)
            marg_np = np.array(marg.cpu())
            data[divergence][normalizer] = []
            for i in range(2):
                data[divergence][normalizer].append([
                    div_func(marg_np[i, j], GROUND_TRUTH[i])
                    for j in range(110)
                ])
            data[divergence][normalizer] = np.array([data[divergence][normalizer]])
    return data

CUTOFF = -1
def plot_divergences(data, fig_name):
    for divergence, data_dict in data.items():
        for y in range(2):
            plt.figure()
            for color, normalizer in zip(COLORS, NORMALIZERS):
                v = data_dict[normalizer]
                cutoff = CUTOFF if CUTOFF != -1 else v.shape[-1]
                mean = np.mean(v[:, y, :cutoff], axis=0)
                std = np.std(v[:, y, :cutoff], axis=0) / np.sqrt(len(v))
                steps = list(range(0, 100 * cutoff, 100))
                plt.plot(steps, mean, color=color)
                plt.fill_between(steps, mean - std, mean + std, alpha=0.5, color=color)
            oddeven = "Even" if y == 0 else "Odd"
            plt.xlabel("Iteration")
            if y == 0:
                plt.ylabel(divergence)
                plt.yticks([0, 0.05, 0.1, 0.15])
            else:
                plt.yticks([])
            plt.xticks([0, 3000, 6000])
            plt.savefig(f"{fig_name}_{divergence}_{oddeven}.png", dpi=600)
            tikzplotlib.save(f"{fig_name}_{divergence}_{oddeven}.tex", textsize=5.0, axis_width="6cm", strict=True)
            plt.close()


SEEDS = [1234]
def main():
    parser = argparse.ArgumentParser(description='Plot divergences from training')
    parser.add_argument('--model_file', type=str, default='./mnist_cnn_1epoch.pt')
    parser.add_argument('--fig_name', type=str, default='mnist_divergences')
    args = parser.parse_args()

    model = CNN()
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    model = model.to(torch.device("cuda"))

    tracker_files = [f"./tracker_mnist_SGD_random_{seed}.pt" for seed in SEEDS]
    data = None
    for tracker_file in tracker_files:
        tracker_global_train = torch.load(tracker_file)

        with torch.no_grad():
            result = calculate_divergences(model, tracker_global_train)
        
        if data is None:
            data = result
        else:
            for div in result:
                for norm in result[div]:
                    data[div][norm] = np.append(data[div][norm], result[div][norm], axis=0)
    plot_divergences(data, args.fig_name)

if __name__ == '__main__':
    main()