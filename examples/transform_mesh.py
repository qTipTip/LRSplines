import matplotlib.pyplot as plt
import numpy as np

from LRSplines import init_tensor_product_LR_spline


def affine(z):
    z[:, 0] = z[:, 0] + z[:, 1] ** 2  # transforming the x coordinates
    z[:, 1] = 1.5 * z[:, 1] - z[:, 0] * 0.8 + z[:, 0]  # transforming the y coordinates
    return z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    k = np.array([-5] * 2 + list(np.linspace(-5, 5, 10)) + [5] * 2)
    d = 2
    LR = init_tensor_product_LR_spline(d, d, k, k)

    LR.refine(0.5, lambda e: np.random.random() if e.area < 0.001 else e.area)

    meshlines = LR.mesh_to_array(50)
    transformed_lines = sigmoid(affine(meshlines))

    for m in transformed_lines:
        plt.plot(m[0], m[1], linewidth=1, color="black")
    plt.show()
