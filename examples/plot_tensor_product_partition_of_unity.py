import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from LRSplines import init_tensor_product_LR_spline

N = 10

d1, d2 = 2, 2
ku = [0, 0, 0, 0.5, 0.75, 1, 1, 1]
LR = init_tensor_product_LR_spline(d1, d2, ku, ku)

x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)
np.random.seed(42)

z = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        z[i, j] = LR(x[i], y[j])

X, Y = np.meshgrid(x, y)

fig = plt.figure()
axs = Axes3D(fig)

axs.plot_wireframe(X, Y, z)
axs.set_xlabel('u')
axs.set_ylabel('v')

plt.show()
