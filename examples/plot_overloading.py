import matplotlib.patches as plp
import matplotlib.pyplot as plt

from src.lr_spline import init_tensor_product_LR_spline
from src.meshline import Meshline


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
    plt.show()


if __name__ == '__main__':
    LR = init_tensor_product_LR_spline(1, 1, [0, 0, 1, 2, 3, 4, 5, 6, 6], [0, 0, 1, 2, 3, 4, 5, 6, 6])

    visualize_mesh(LR)

    m1 = Meshline(2, 4, constant_value=2.5, axis=0)
    m2 = Meshline(2, 4, constant_value=2.5, axis=1)
    m3 = Meshline(2, 4, constant_value=3.5, axis=1)
    m4 = Meshline(2, 4, constant_value=3.5, axis=0)

    LR.insert_line(m1)

    # visualize_mesh(LR)

    LR.insert_line(m2)
    # visualize_mesh(LR)
    LR.insert_line(m3)
    visualize_mesh(LR)
    LR.insert_line(m4)
    visualize_mesh(LR)

    # visualize_mesh(LR)
    m5 = Meshline(1, 5, constant_value=2.5, axis=0)
    m6 = Meshline(1, 5, constant_value=3.5, axis=0)
    LR.insert_line(m5)
    LR.insert_line(m6)
