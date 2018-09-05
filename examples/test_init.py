from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from lr_spline import init_tensor_product_LR_spline
from src.b_spline import BSpline


def f(e):
    x, y, area = e.midpoint[0], e.midpoint[1], e.area
    b = BSpline(1, 1, [0, 0.5, 1], [0, 0.5, 1])
    return b(x, y)


def visualize_mesh(LR) -> None:
    """
    Plots the LR-mesh.
    """

    for m in LR.meshlines:
        x = (m.start, m.stop)
        y = (m.constant_value, m.constant_value)
        if m.axis == 0:
            plt.plot(y, x)
        else:
            plt.plot(x, y)
        plt.text(m.midpoint[0], m.midpoint[1], '{}'.format(m.multiplicity), bbox=dict(facecolor='white', alpha=1))
    for m in LR.M:
        plt.text(m.midpoint[0], m.midpoint[1], '{}'.format(len(m.supported_b_splines)))
    plt.title('dim(S) = {}'.format(len(LR.S)))


if __name__ == '__main__':

    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.25, 0.75, 1, 1, 1]
    LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

    for i in range(0):
        data = list(map(f, LR.M))
        i = np.argmax(data)

        m = LR.get_minimal_span_meshline(LR.M[i], axis=i % 2)
        LR.insert_line(m)
    visualize_mesh(LR)
    plt.show()

    for b in LR.S:
        b.coefficient = np.random.random()
    x = np.linspace(0, 1, 100)
    z = np.zeros((100, 100))

    start = timer()
    for i in range(100):
        for j in range(100):
            z[i, j] = LR(x[i], x[j])
    print('Eval took {} seconds', timer() - start)

    X, Y = np.meshgrid(x, x)

    fig = plt.figure()
    axs = Axes3D(fig)

    axs.plot_wireframe(X, Y, z)
    plt.show()
