import numpy as np

from src.b_spline import BSpline, intersects
from src.element import Element


def test_b_spline_init():
    du = 2
    dv = 2
    ku = [0, 1, 2, 3]
    kv = [3, 4, 5, 6]

    w = 1

    B = BSpline(du, dv, ku, kv, w)

    np.testing.assert_array_almost_equal(B.knots_u, ku)
    np.testing.assert_array_almost_equal(B.knots_v, kv)
    assert (B.degree_v, B.degree_v) == (dv, du)
    assert B.weight == w


def test_b_spline_intersects():
    du = 2
    dv = 2
    ku = [0, 1, 2, 3]
    kv = [3, 4, 5, 6]

    w = 1

    B = BSpline(du, dv, ku, kv, w)

    e1 = Element(2, -1, 4, 4)
    e2 = Element(10, 10, 20, 20)
    e3 = Element(1, 4, 2, 5)

    assert intersects(B, e1)
    assert not intersects(B, e2)
    assert intersects(B, e3)


def test_b_spline_add_to_support_if_intersects():
    du = 2
    dv = 2
    ku = [0, 1, 2, 3]
    kv = [3, 4, 5, 6]

    w = 1

    B = BSpline(du, dv, ku, kv, w)

    e1 = Element(2, -1, 4, 4)
    e2 = Element(10, 10, 20, 20)
    e3 = Element(1, 4, 2, 5)

    assert B.add_to_support_if_intersects(e1)
    assert not B.add_to_support_if_intersects(e2)
    assert B.add_to_support_if_intersects(e3)

    assert e1 in B.elements_of_support
    assert e2 not in B.elements_of_support
    assert e3 in B.elements_of_support
