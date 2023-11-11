import LRSplines

du = 2
dv = 2
knots_u = [0, 0, 0, 1, 2, 3, 3, 3]
knots_v = [0, 0, 0, 1, 2, 3, 3, 3]

LR = LRSplines.init_tensor_product_LR_spline(du, dv, knots_u, knots_v)
LR.visualize_mesh()

m1 = LRSplines.Meshline(start=0, stop=2, constant_value=1.5, axis=0)
m2 = LRSplines.Meshline(start=1, stop=3, constant_value=1.5, axis=1)

LR.insert_line(m1)
LR.insert_line(m2)
LR.visualize_mesh()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# set the coefficients explicitly
for b in LR.S:
    b.coefficient = np.random.uniform(-3, 3)

N = 50
x = np.linspace(knots_u[0], knots_u[-1], N, endpoint=False)
y = np.linspace(knots_v[0], knots_v[-1], N)
z = np.zeros((N, N))
X, Y = np.meshgrid(x, y)

for i in range(N):
    for j in range(N):
        z[i, j] = LR(x[i], y[j])

fig = plt.figure()
axs = Axes3D(fig)
axs.plot_wireframe(X, Y, z)
plt.show()
