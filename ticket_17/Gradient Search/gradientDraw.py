import numpy as np
import matplotlib.pyplot as plt
import random

def contourPlot(f, ax):
    x1 = np.arange(-4, 4.1, 0.1)
    y1 = np.arange(-4, 4.1, 0.1)
    xx, yy = np.meshgrid(x1, y1)
    F = np.zeros_like(xx)

    for i in range(len(x1)):
        for j in range(len(y1)):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)

    nlevels = 20
    contour = ax.contour(xx, yy, F, nlevels)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return contour

def gradientDraw(ax, coords, nsteps):
    fSize = 11
    x0 = coords[0]
    ax.text(x0[0] + 0.1, x0[1] + 0.1, '0', fontsize=fSize)
    for i in range(1, nsteps):
        x1 = coords[i]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], 'b-', lw=1.2, marker='s', ms=3)
        x0 = x1

    ax.text(x1[0] - 0.1, x1[1] - 0.5, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0], x1[1], c='red', zorder=12)

def draw(f, coords):
    nsteps = len(coords)
    fig, ax = plt.subplots()
    fig.suptitle('Gradient method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    gradientDraw(ax, coords, nsteps)
    contourPlot(f, ax)
    plt.show()

