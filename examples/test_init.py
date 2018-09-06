from timeit import default_timer as timer

import matplotlib.patches as plp
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
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    for m in LR.meshlines:
        x = (m.start, m.stop)
        y = (m.constant_value, m.constant_value)
        if m.axis == 0:
            axs.plot(y, x)
        else:
            axs.plot(x, y)
        axs.text(m.midpoint[0], m.midpoint[1], '{}'.format(m.multiplicity), bbox=dict(facecolor='white', alpha=1))
    for m in LR.M:
        w = m.u_max - m.u_min
        h = m.v_max - m.v_min

        if m.is_overloaded():
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='red', alpha=0.5))
        else:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='green', alpha=0.5))

        axs.text(m.midpoint[0], m.midpoint[1], '{}'.format(len(m.supported_b_splines)))
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
    LR.M[3].add_supported_b_spline(LR.S[2])
    visualize_mesh(LR)
    plt.show()

    for b in LR.S:
        b.coefficient = np.random.random()
    N = 30
    x = np.linspace(0, 1, N)
    z = np.zeros((N, N))

    start = timer()
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], x[j])
    print('Eval took {} seconds', timer() - start)

    X, Y = np.meshgrid(x, x)

    fig = plt.figure()
    axs = Axes3D(fig)

    axs.plot_wireframe(X, Y, z)
    plt.show()
