from LRSplines import Element, np
from LRSplines.lr_spline import init_tensor_product_LR_spline


def element_contribution(e: Element):
    mp_x, mp_y = e.midpoint

    return (mp_x - 3) ** 2 + (mp_y - 3) ** 2 < 3


def circle(t):
    return 2.5 * np.cos(t) + 3, 2.5 * np.sin(t) + 3


if __name__ == "__main__":
    LR = init_tensor_product_LR_spline(
        1, 1, [0, 0, 1, 2, 3, 4, 5, 6, 6], [0, 0, 1, 2, 3, 4, 5, 6, 6]
    )

    LR.visualize_mesh(False, False)
    # LR.refine(beta=0.5, error_function=element_contribution, refinement_strategy='full')
    # LR.visualize_mesh(False, False)

    while len(LR.S) < 200:
        print(len(LR.S))
        circl_vals = [circle(t) for t in np.linspace(0, 2 * np.pi, 50)]
        for x, y in circl_vals:
            e = LR.find_element_containing_point(x, y)
            LR.refine_by_element_minimal(e)
    LR.visualize_mesh(False, False)
