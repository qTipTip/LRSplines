import numpy as np

from src.aux_split_functions import split_single_basis_function
from src.b_spline import BSpline
from src.meshline import Meshline


def test_aux_split_methods():
    d1 = 2
    d2 = 2
    ku = [0, 1, 2, 4]
    kv = [1, 2, 4, 5]
    B = BSpline(d1, d2, ku, kv)

    m = Meshline(0, 5, constant_value=3, axis=0)
    eps = 1.0E-14
    b1, b2 = split_single_basis_function(m, B)

    assert b1.weight == 1
    assert (b2.weight - 1 / 3) < eps
    assert np.allclose(b1.knots_u, [0, 1, 2, 3])
    assert np.allclose(b2.knots_u, [1, 2, 3, 4])
