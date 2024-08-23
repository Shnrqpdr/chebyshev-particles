import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import chebyshev_2d as Chebyshev2D
import scipy.special as sp
import scipy.integrate as integrate

r, theta, L = Chebyshev2D.getChebyshev2d()

k = np.linspace(1.0, 5.0, 1000)
theta_0 = np.linspace(0, 2*np.pi, 1000)

dif_cross_section = np.zeros((20, 20))
total_cross_section = np.zeros((20, 20))

for i in range(0, 20):
    kzin = k[i]
    for j in range(0, 20):
        theta0 = theta_0[j]
        A, B = Chebyshev2D.get_system_linear(kzin, theta0)
        X = np.linalg.solve(A, B)

        dif_cross_section[i, j] = Chebyshev2D.differential_cross_section(kzin, X)
        total_cross_section[i, j] = Chebyshev2D.total_cross_section(kzin, X)

## Plots

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
im = ax.imshow(total_cross_section, cmap='rainbow',interpolation='none', extent=[0,2*np.pi,1,5], aspect="auto")
fig.colorbar(im, ax=ax, label='$\sigma_{total}$')
ax.set_xlabel('$\\theta_0$', fontsize=16)
ax.set_ylabel('$k$', fontsize=16)
plt.savefig('plotSecaoChoqueTotal_beta5_gamma-10.png', dpi=400)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
ax.plot_surface(theta_0, k, total_cross_section, edgecolor='gray', lw=0.4, rstride=130, cstride=130, cmap='rainbow')

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(theta_0, k, total_cross_section, zdir='z', offset=-10, cmap='rainbow')
ax.contourf(theta_0, k, total_cross_section, zdir='x', offset=-30, cmap='rainbow')
ax.contourf(theta_0, k, total_cross_section, zdir='y', offset=7, cmap='rainbow')

ax.set(xlim=(-30, 30), ylim=(0, 7), zlim=(-10, 11))

ax.set_xlabel('$\\theta_0$', fontsize=16)
ax.set_ylabel('$k$', fontsize=16)
ax.set_zlabel('$\sigma_{tot}$', fontsize=16)

# plt.show()
plt.savefig('plotSecaoChoqueTotalAll.png', dpi=400)