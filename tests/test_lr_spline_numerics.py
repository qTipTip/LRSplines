import numpy as np
import pytest

from LRSplines.lr_spline import init_tensor_product_LR_spline


@pytest.mark.parametrize("N", [2, 4, 6, 8])
def test_lr_spline_partition_of_unity(N):
    LR = init_tensor_product_LR_spline(2, 2, [0, 0, 0, 1, 2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6])

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
def test_lr_spline_partition_of_unity_two(N):
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
