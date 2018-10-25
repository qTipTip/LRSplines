import numpy as np

from LRSplines import BSpline

d = 2
knots = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1]
])

B = BSpline(d, d, knots[0], knots[1])

N = 100
M = 100

XX = np.linspace(0, 3, N)
YY = np.linspace(0, 2, M)
XY = np.array([
    [x, y]
    for x in XX
    for y in YY
])
X, Y = np.meshgrid(XX, YY)

z = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        z[i, j] = B(XX[i], YY[j])
