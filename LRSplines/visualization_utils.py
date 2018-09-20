import matplotlib.pyplot as plt
import matplotlib.patches as plp

from LRSplines import init_tensor_product_LR_spline


def plot_basis_support(LR, basis_function):
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
        basis_patch = plp.Rectangle(xy, w, h, hatch='//', fill=False)

        axs.add_patch(basis_patch)
    axs.tick_params(axis=None)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    LR = init_tensor_product_LR_spline(2, 2, [0, 0, 0, 1, 2, 3, 3, 3], [0, 0, 0, 1, 2, 3, 3, 3])
    for b in LR.S:
        plot_basis_support(LR, b)
