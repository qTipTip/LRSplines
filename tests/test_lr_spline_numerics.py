import numpy as np
import pytest

from LRSplines.lr_spline import init_tensor_product_LR_spline


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity_interior(N):
    LR = init_tensor_product_LR_spline(
        2, 2, [0, 0, 0, 1, 2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6]
    )

    x = np.linspace(0, 6, N, endpoint=False)
    y = np.linspace(0, 6, N, endpoint=False)
    for j in range(12):
        m = LR.get_minimal_span_meshline(np.random.choice(LR.M), axis=j % 2)
        LR.insert_line(m)
    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], y[j])
    expected = np.ones((N, N))
    np.testing.assert_array_almost_equal(z, expected)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity_two_interior(N):
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

    x = np.linspace(0, 1, N, endpoint=False)
    np.random.seed(42)
    for k in range(12):
        m = LR.get_minimal_span_meshline(np.random.choice(LR.M), axis=k % 2)
        LR.insert_line(m)
    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], x[j])
    expected = np.ones((N, N))
    np.testing.assert_array_almost_equal(z, expected)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity_full(N):
    LR = init_tensor_product_LR_spline(
        2, 2, [0, 0, 0, 1, 2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6]
    )

    x = np.linspace(0, 6, N, endpoint=True)
    y = np.linspace(0, 6, N, endpoint=True)
    for j in range(12):
        m = LR.get_minimal_span_meshline(np.random.choice(LR.M), axis=j % 2)
        LR.insert_line(m)
    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], y[j])
    expected = np.ones((N, N))
    np.testing.assert_array_almost_equal(z, expected)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity_two_full(N):
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

    x = np.linspace(0, 1, N, endpoint=True)
    np.random.seed(42)
    for k in range(12):
        m = LR.get_minimal_span_meshline(np.random.choice(LR.M), axis=k % 2)
        LR.insert_line(m)
    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], x[j])
    expected = np.ones((N, N))
    np.testing.assert_array_almost_equal(z, expected)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity_tensor_product(N):
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

    x = np.linspace(0, 1, N, endpoint=True)
    np.random.seed(42)

    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z[i, j] = LR(x[i], x[j])
    expected = np.ones((N, N))
    np.testing.assert_array_almost_equal(z, expected)


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity_at_end(N):
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    LR = init_tensor_product_LR_spline(d1, d2, ku, ku)
    expected = np.ones(N)

    x = np.linspace(0, 1, N, endpoint=True)
    np.random.seed(42)

    z = np.zeros(N)
    for i in range(N):
        z[i] = LR(x[i], 1)
    np.testing.assert_array_almost_equal(z, expected)

    z = np.zeros(N)
    for i in range(N):
        z[i] = LR(1, x[i])
    np.testing.assert_array_almost_equal(z, expected)


def test_lr_spline_partition_at_end_uv_point():
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    kv = [0, 0, 0, 1, 2, 3, 3, 3]
    LR = init_tensor_product_LR_spline(d1, d2, ku, kv)

    e = LR.find_element_containing_point(1, 3)
    for b in e.supported_b_splines:
        assert b.end_v
        assert 3 in b.knots_v
        if b.end_u:
            assert 1 in b.knots_u


def test_lr_spline_at_end_vs_knots_tensor_product():
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    kv = [0, 0, 0, 1, 2, 3, 3, 3]
    LR = init_tensor_product_LR_spline(d1, d2, ku, kv)

    for b in LR.S:
        if b.end_u:
            assert ku[-1] in b.knots_u
        if b.end_v:
            assert kv[-1] in b.knots_v


def test_lr_spline_at_end_vs_knots_refined():
    d1, d2 = 2, 2
    ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
    kv = [0, 0, 0, 1, 2, 3, 3, 3]
    LR = init_tensor_product_LR_spline(d1, d2, ku, kv)

    np.random.seed(42)
    for k in range(12):
        m = LR.get_minimal_span_meshline(np.random.choice(LR.M), axis=k % 2)
        LR.insert_line(m)

    for b in LR.S:
        if b.end_u:
            assert ku[-1] in b.knots_u
        if b.end_v:
            assert kv[-1] in b.knots_v
