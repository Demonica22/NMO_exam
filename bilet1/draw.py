import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import *
import random
import matplotlib.patches as patches
from trustregionsearch import *
from matplotlib.patches import FancyBboxPatch


def plotReg(x0, y0, Delta, ax):
    r = Delta
    color = [0, 0.4470, 0.7410]
    # Отрисовка круга
    circ = patches.Circle((x0, y0), radius=r, facecolor=color, ec='None', alpha=0.1)
    ax.add_patch(circ)


def contourPlot(ax, f):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-4, 4.1, 0.1)
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)
    n = len(y1)
    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)
    # массивы для графиков функции и ее производных по x и y
    F = np.zeros((n, m))
    # вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)

    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def trustregDraw(ax, coords, nsteps, radius):
    fSize = 11
    x0 = coords[0]
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=0.2)
        plotReg(x0[0], x0[1], radius[i], ax)

    ax.text(x1[0], x1[1] - 0.2, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)
    plotReg(x1[0], x1[1], radius[len(radius) - 1], ax)


def draw(coords, nsteps, flag, radius, f):
    fig, ax = plt.subplots()
    fig.suptitle('Trust region method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    trustregDraw(ax, coords, nsteps, radius)
    contourPlot(ax, f)
    name = "plot" + flag + ".png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)









