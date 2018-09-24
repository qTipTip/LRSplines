import matplotlib.pyplot as plt
import matplotlib.patches as plp
from mpl_toolkits.mplot3d import Axes3D

from LRSplines import init_tensor_product_LR_spline, Meshline, np


def plot_basis_function(LR, basis_function):
    N = 30
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1, projection='3d')

    x = np.linspace(LR.u_range[0], LR.u_range[1], N)
    y = np.linspace(LR.v_range[0], LR.v_range[1], N)
    z = np.zeros((N, N))

    X, Y = np.meshgrid(x, y)

    for i in range(N):
        for j in range(N):
            z[i, j] = basis_function(y[j], x[i])
    axs.set_zlim3d(0, 1)
    axs.plot_surface(X, Y, z)
    plt.show()

def plot_basis_support(LR, basis_function, axis=False, hatch='//'):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    for meshline in LR.meshlines:
        x = meshline.start, meshline.stop
        y = meshline.constant_value, meshline.constant_value

        if meshline.axis == 0:
            axs.plot(y, x, c='black')
        else:
            axs.plot(x, y, c='black')

        xy = basis_function.knots_u[0], basis_function.knots_v[0]
        w = basis_function.knots_u[-1] - basis_function.knots_u[0]
        h = basis_function.knots_v[-1] - basis_function.knots_v[0]
        basis_patch = plp.Rectangle(xy, w, h, hatch=hatch, fill=False, linewidth=0.2)

        axs.add_patch(basis_patch)
    if not axis:
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    LR = init_tensor_product_LR_spline(2, 2, [0, 0, 0, 1, 2, 3, 3, 3], [0, 0, 0, 1, 2, 3, 3, 3])
    m1 = Meshline(start=0, stop=2, constant_value=1.5, axis=0)
    m2 = Meshline(start=1, stop=3, constant_value=1.5, axis=1)

    LR.insert_line(m1)
    LR.insert_line(m2)
    for b in LR.S:
        plot_basis_support(LR, b)
        plot_basis_function(LR, b)
