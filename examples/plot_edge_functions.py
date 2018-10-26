import matplotlib.pyplot as plt
import numpy as np

from LRSplines import init_tensor_product_LR_spline, BSpline


def plot_lr(b: BSpline):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axs = Axes3D(fig)

    N = 40
    x = np.linspace(b.knots_u[0], b.knots_u[-1], N)
    y = np.linspace(b.knots_v[0], b.knots_v[-1], N)
    z = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            z[i, j] = b(x[i], y[j])

    X, Y = np.meshgrid(x, y)
    axs.plot_surface(X, Y, z)
    plt.title(';'.join((np.array2string(b.knots_u), (np.array2string(b.knots_v)))))
    plt.show()


ku = [0, 1, 2]
kv = [0, 0, 0, 1, 2, 2, 2]
d = 1

lr = init_tensor_product_LR_spline(0, 2, ku, kv)
i = lr.edge_functions()

for j in i:
    plot_lr(lr.S[j])
