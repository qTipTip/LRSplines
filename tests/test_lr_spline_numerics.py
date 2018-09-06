import numpy as np
import pytest

from lr_spline import init_tensor_product_LR_spline


@pytest.mark.parametrize("N", [20, 40, 60])
def test_lr_spline_partition_of_unity(N):
    LR = init_tensor_product_LR_spline(2, 2, [0, 0, 0, 1, 2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6])

    x = np.linspace(0, 6, N, endpoint=False)
    y = np.linspace(0, 6, N, endpoint=False)

    z = np.zeros((N, N))

    for j in range(5):
        m = LR.get_minimal_span_meshline(LR.M[10], axis=j % 1)
        LR.insert_line(m)
        for i in range(N):
            for j in range(N):
                z[i, j] = LR(x[i], y[j])
        expected = np.ones((N, N))
        np.testing.assert_array_almost_equal(z, expected)