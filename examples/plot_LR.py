import random
from timeit import default_timer as timer

import matplotlib.patches as plp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.integrate as integrate

from LRSplines.lr_spline import init_tensor_product_LR_spline


def f(x, y):
    return np.sin(2 * np.pi * x) * np.sin(np.pi * y) * np.exp(x * y)


def visualize_mesh(LR) -> None:
    """
    Plots the LR-mesh.
    """
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    for m in LR.meshlines:
        x = (m.start, m.stop)
        y = (m.constant_value, m.constant_value)
        if m.axis == 0:
            axs.plot(y, x, color='black')
        else:
            axs.plot(x, y, color='black')
        axs.text(m.midpoint[0], m.midpoint[1], '{}'.format(m.multiplicity), bbox=dict(facecolor='white', alpha=1))
    for m in LR.M:
        w = m.u_max - m.u_min
        h = m.v_max - m.v_min

        if m.is_overloaded():
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='red', alpha=0.2))
        else:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='green', alpha=0.2))

        axs.text(m.midpoint[0], m.midpoint[1], '{}'.format(len(m.supported_b_splines)))
    plt.title('dim(S) = {}'.format(len(LR.S)))


if __name__ == '__main__':
    for N in [10, 20, 30, 40]:
        d1, d2 = 2, 2
        ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
        LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

        x = np.linspace(0, 1, N, endpoint=False)
        np.random.seed(42)
        for k in range(12):
            m = LR.get_minimal_span_meshline(np.random.choice(LR.M), axis=k % 2)
            LR.insert_line(m)
        visualize_mesh(LR)
        z = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                z[i, j] = LR(x[i], x[j])
        expected = np.ones((N, N))
        #np.testing.assert_array_almost_equal(z, expected)
        X, Y = np.meshgrid(x, x)

        fig = plt.figure()
        axs1 = Axes3D(fig)
        axs1.plot_wireframe(X, Y, z, label='LR')
        plt.legend()
        plt.show()