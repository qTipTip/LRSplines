import numpy as np

from LRSplines import init_tensor_product_LR_spline

d = 2
k = [0] * (d + 1) + list(range(1, 20)) + [20] * (d + 1)

lr = init_tensor_product_LR_spline(d, d, k, k)

p = np.array([
    [x, y]
    for x in np.linspace(0, 20, 100)
    for y in np.linspace(0, 20, 100)
])

for x, y in p:
    lr(x, y)
