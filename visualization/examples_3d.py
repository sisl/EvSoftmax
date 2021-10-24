import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

EQUILATERAL_TRIANGLE = np.array(
    [[0, 1], [-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5]]
)


def sparsemax(z):
    # step 1
    z_sorted = z[np.argsort(-z)]

    # step 2
    col_range = np.arange(len(z)) + 1
    ks = (1 + col_range * z_sorted) > np.cumsum(z_sorted)
    k_z = np.max(col_range * ks)

    # step 3
    tau_z = (z_sorted[:k_z].sum() - 1) / k_z

    # step 4
    return np.maximum(z - tau_z, 0)


def evsoftmax(z):
    mask = z >= z.mean()
    items = np.exp(z) * mask
    return items / items.sum()


def softmax(z):
    y = np.exp(z)
    return y / y.sum()


def identity(z):
    return z


def project(z):
    return z @ EQUILATERAL_TRIANGLE


def plot_triangle():
    z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    z_proj = project(z)
    plt.plot(z_proj[:, 0], z_proj[:, 1], "k")


def plot_setup():
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.text(EQUILATERAL_TRIANGLE[0, 0], EQUILATERAL_TRIANGLE[0, 1], "(1, 0, 0)")
    plt.text(EQUILATERAL_TRIANGLE[1, 0], EQUILATERAL_TRIANGLE[1, 1], "(0, 1, 0)")
    plt.text(EQUILATERAL_TRIANGLE[2, 0], EQUILATERAL_TRIANGLE[2, 1], "(0, 0, 1)")
    plt.axis("off")


def plot_points(z):
    method_names = ["Input", "Softmax", "Sparsemax", "Ev-softmax"]
    functions = [identity, softmax, sparsemax, evsoftmax]
    plot_options = ["ko", "yp", "r^", "gs"]
    for method, fn, po in zip(method_names, functions, plot_options):
        res = project(fn(z))
        plt.plot(res[0], res[1], po, label=method, markersize=12)


def main():
    z1 = np.array([1.3, 0.37, -0.67])
    plt.figure(figsize=(5, 5))
    plt.subplot(121)
    plot_setup()
    plot_triangle()
    plot_points(z1)

    z2 = np.array([0.4, 1.4, -0.8])
    plt.subplot(122)
    plot_setup()
    plot_triangle()
    plot_points(z2)
    plt.legend()

    tikzplotlib.save("example.tex")


if __name__ == "__main__":
    main()
