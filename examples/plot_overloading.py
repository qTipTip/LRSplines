import random

from LRSplines.lr_spline import init_tensor_product_LR_spline
from LRSplines.meshline import Meshline

if __name__ == "__main__":
    LR = init_tensor_product_LR_spline(
        1, 1, [0, 0, 1, 2, 3, 4, 5, 6, 6], [0, 0, 1, 2, 3, 4, 5, 6, 6]
    )

    m1 = Meshline(2, 4, constant_value=2.5, axis=0)
    m2 = Meshline(2, 4, constant_value=2.5, axis=1)
    m3 = Meshline(2, 4, constant_value=3.5, axis=1)
    m4 = Meshline(2, 4, constant_value=3.5, axis=0)

    LR.insert_line(m1)

    LR.insert_line(m2)
    LR.insert_line(m3)
    LR.insert_line(m4)

    m5 = Meshline(1, 5, constant_value=2.5, axis=0)
    m6 = Meshline(1, 5, constant_value=3.5, axis=0)
    LR.insert_line(m5)
    LR.insert_line(m6)

    for i in range(100):
        m = LR.get_minimal_span_meshline(random.choice(LR.M), axis=i % 2)
        LR.insert_line(m)

    LR.visualize_mesh(False, False)
