import random
from timeit import default_timer as timer

import matplotlib.patches as plp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

from lr_spline import init_tensor_product_LR_spline


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


def assemble_mass_and_load(LR, f):
    n = len(LR.S)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for j, bj in enumerate(LR.S):
        for i, bi in enumerate(LR.S):
            A[i, j] = integrate.dblquad(lambda x, y: bi(x, y) * bj(x, y), 0, 1, lambda x: 0, lambda x: 1)[0]
        b[j] = integrate.dblquad(lambda x, y: bj(x, y) * f(x, y), 0, 1, lambda x: 0, lambda x: 1)[0]
    return A, b


if __name__ == '__main__':

    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 1, 1, 1]
    LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

    for i in range(2):
        m = LR.get_minimal_span_meshline(random.choice(LR.M), axis=i % 2)
        LR.insert_line(m)
    visualize_mesh(LR)
    plt.show()

    print('Assembling')
    # A, b = assemble_mass_and_load(LR, f)
    print('Solving')
    # C = np.linalg.solve(A, b)
    # for c, b in zip(C, LR.S):
    #    b.coefficient = c

    print('Evaluating')
    N = 30
    x = np.linspace(0, 1, N, endpoint=False)
    z = np.zeros((N, N))
    Z = np.zeros((N, N))
    start = timer()
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], x[j])
            Z[i, j] = f(x[i], x[j])
    print('Eval took seconds', timer() - start)

    X, Y = np.meshgrid(x, x)

    fig = plt.figure()
    axs1 = fig.add_subplot(121, projection='3d')
    axs2 = fig.add_subplot(122, projection='3d')
    axs1.plot_wireframe(X, Y, z, label='approx')
    axs2.plot_wireframe(X, Y, Z, label='exact')
    plt.legend()
    plt.show()

    # plt.spy(markersize=5)
    # plt.show()
