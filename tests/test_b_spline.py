import numpy as np
import pytest

from LRSplines import init_tensor_product_LR_spline
from LRSplines.b_spline import BSpline, _evaluate_univariate_b_spline, _find_knot_interval
from LRSplines.element import Element


@pytest.fixture(scope='function')
def B():
    du = 2
    dv = 2
    ku = [0, 1, 2, 3]
    kv = [3, 4, 5, 6]

    w = 1

    return BSpline(du, dv, ku, kv, w)


def test_b_spline_init(B):
    ku = [0, 1, 2, 3]
    kv = [3, 4, 5, 6]
    du = 2
    dv = 2
    np.testing.assert_array_almost_equal(B.knots_u, ku)
    np.testing.assert_array_almost_equal(B.knots_v, kv)
    assert (B.degree_v, B.degree_v) == (dv, du)


def test_b_spline_intersects(B):
    e1 = Element(2, -1, 4, 4)
    e2 = Element(10, 10, 20, 20)
    e3 = Element(1, 4, 2, 5)

    assert B.intersects(e1)
    assert not B.intersects(e2)
    assert B.intersects(e3)


def test_b_spline_add_to_support_if_intersects(B):
    e1 = Element(2, -1, 4, 4)
    e2 = Element(10, 10, 20, 20)
    e3 = Element(1, 4, 2, 5)
    e4 = Element(3, 3, 4, 6)

    assert B.add_to_support_if_intersects(e1)
    assert not B.add_to_support_if_intersects(e2)
    assert B.add_to_support_if_intersects(e3)
    assert not B.add_to_support_if_intersects(e4)

    assert e1 in B.elements_of_support
    assert e2 not in B.elements_of_support
    assert e3 in B.elements_of_support
    assert e4 not in B.elements_of_support


def test_b_spline_remove_from_support(B):
    e1 = Element(2, -1, 4, 4)
    e2 = Element(10, 10, 20, 20)
    e3 = Element(1, 4, 2, 5)

    assert B.add_to_support_if_intersects(e1)
    assert not B.add_to_support_if_intersects(e2)
    assert B.add_to_support_if_intersects(e3)

    assert not B.remove_from_support(e2)
    assert B.remove_from_support(e1)
    assert e1 not in B.elements_of_support
    assert B.remove_from_support(e3)
    assert len(B.elements_of_support) == 0


def test_evaluate_univariate_b_spline():
    d = 2
    k = [0, 1, 2, 3]
    X = np.linspace(0, 3, 20, endpoint=False)

    def exact(x):
        if 0 <= x < 1:
            return x * x / 2
        elif 1 <= x < 2:
            return x / 2 * (2 - x) + (3 - x) / 2 * (x - 1)
        else:
            return (3 - x) * (3 - x) / 2

    for x in X:
        np.testing.assert_almost_equal(_evaluate_univariate_b_spline(x, k, d), exact(x))


def test_evaluate_univariate_b_spline_two():
    d = 1
    k = [0, 0, 1]
    X = np.linspace(0, 1, 200)

    def exact(x):
        if 0 <= x <= 1:
            return 1 - x
        else:
            return 0

    for x in X:
        np.testing.assert_almost_equal(_evaluate_univariate_b_spline(x, k, d, endpoint=True), exact(x))


def test_evaluate_univariate_b_spline_outside_knots():
    d = 2
    k = [0, 1, 2, 3]
    x = -1.0e-14

    assert _evaluate_univariate_b_spline(x, k, d) == 0


def test_evaluate_b_spline():
    ku = [0, 1, 2, 3]
    kv = [0, 1, 2, 3]
    d1 = 2
    d2 = 2
    B = BSpline(d1, d2, ku, kv)

    np.testing.assert_almost_equal(B(0, 0), 0)
    np.testing.assert_almost_equal(B(1.5, 1.5), 0.5625)
    np.testing.assert_almost_equal(B(0, 1.5), 0)
    np.testing.assert_almost_equal(B(1, 2), 0.25)


def test_endpoint_interpolation():
    ku = [0, 0, 0, 1]
    kv = [3, 4, 4, 4]

    d1 = 2
    d2 = 2
    B = BSpline(d1, d2, ku, kv, end_u=True, end_v=True)

    assert B(0, 3) == 0
    assert B(0, 4) == 1
    assert B(1, 3) == 0
    assert B(1, 4) == 0


def test_endpoint_interpolation_univariate():
    ku = [0, 1, 1, 1]
    d = 2
    x = 1
    assert _evaluate_univariate_b_spline(x, ku, d, endpoint=True) == 1
    assert _evaluate_univariate_b_spline(x, ku, d) == 0


def test_endpoint_find_index():
    k = np.array([0, 0, 0, 1, 2, 3, 4])
    x = 4

    assert _find_knot_interval(x, k, endpoint=False) == -1
    assert _find_knot_interval(x, k, endpoint=True) == 5


def test_cached_univariate():
    B = BSpline(2, 2, [0.75, 0.875, 1, 1], [0, 1, 1, 1], end_v=True)


def test_edge_single():
    lr = init_tensor_product_LR_spline(1, 1, [0, 1, 2], [0, 1, 2])
    b = lr.S[0]

    assert b.north and b.south and b.east and b.west


def test_edge_double():
    lr = init_tensor_product_LR_spline(1, 1, [0, 1, 2, 3], [0, 1, 2])
    b1 = lr.S[0]
    b2 = lr.S[1]

    assert b1.north and b1.south and b1.west and not b1.east
    assert b2.north and b2.south and b2.east and not b2.west


def test_univariate_derivative():
    k = [0, 1, 2]
    d = 1

    x = np.linspace(0, 2, 20)

    def d_exact(x):
        if 0 <= x < 1:
            return 1
        elif 1 <= x <= 2:
            return -1
        else:
            return 0

    computed_derivative = [_evaluate_univariate_b_spline(X, k, d, endpoint=True, r=1) for X in x]
    expected_derivative = [d_exact(X) for X in x]

    np.testing.assert_allclose(computed_derivative, expected_derivative)


def test_univariate_derivative_quadratic():
    k = [0, 0, 0, 1]
    d = 2
    x = np.linspace(0, 3, 20)

    def d_exact(x):
        if 0 <= x < 1:
            return -2 * (1 - x)
        return 0

    computed_derivative = [_evaluate_univariate_b_spline(X, k, d, endpoint=True, r=1) for X in x]
    expected_derivative = [d_exact(X) for X in x]

    np.testing.assert_allclose(computed_derivative, expected_derivative)


def test_bivariate_derivative_mixed_deriv():
    k = [0, 0, 0, 1]
    d = 2
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)

    b = BSpline(d, d, k, k, end_u=True, end_v=True)

    def d_exact(x, y):
        res = 1
        if 0 <= x <= 1:
            res *= -2 * (1 - x)
        if 0 <= y <= 1:
            res *= -2 * (1 - y)
        return res

    computed_derivative = [b(X, Y, r1=1, r2=1) for X in x for Y in y]
    expected_derivative = [d_exact(X, Y) for X in x for Y in y]

    np.testing.assert_allclose(computed_derivative, expected_derivative)


def test_bivariate_derivative_gradient():
    k = [0, 0, 0, 1]
    d = 2
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)

    b = BSpline(d, d, k, k, end_u=True, end_v=True)

    def d_exact(x, y):
        return np.array((-2 * (1 - x) * (1 - y) ** 2, -2 * (1 - x) ** 2 * (1 - y)))

    computed_gradient = [b.grad(X, Y) for X in x for Y in y]
    expected_gradient = [d_exact(X, Y) for X in x for Y in y]

    np.testing.assert_allclose(computed_gradient, expected_gradient)
