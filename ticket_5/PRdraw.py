import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main import *
import random
from polakribieresearch import *

def contourPlot(ax, f):

    #Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-10, 10.1, 0.1)
    m = len(x1)
    y1 = np.arange(-10, 10.1, 0.1)
    n = len(y1)

    #делаем сетку
    [xx, yy] = np.meshgrid(x1,y1)

    #массивы для графиков функции и ее производных по x и y
    F = np.zeros((n,m))

    #вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i,j],yy[i,j]]
            F[i,j] = f(X)

    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths = 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def prDraw(ax, coords, nsteps):
    fSize = 11
    x0=coords[0]
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0),fontsize = fSize)
    for i in range(nsteps-1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]],[x0[1], x1[1]], lw = 1.2, marker = 's', ms = 3)

    ax.text(x1[0] - 0.35 , x1[1] - 0.2, str(nsteps),fontsize = fSize)
    ax.scatter(x1[0], x1[1],marker = 'o', c = 'red', zorder=12)


def draw(coords, nsteps, f, flag):
    fig, ax = plt.subplots()
    fig.suptitle('Polar Ribiere method each step visualisation & Countour plot')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    prDraw(ax, coords, nsteps)
    contourPlot(ax, f)
    name = "plot" + flag + ".png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)


