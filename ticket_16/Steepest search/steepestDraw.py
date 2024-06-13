import numpy as np
import matplotlib.pyplot as plt
from main import *
import random
from steepestsearch import *

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

def steepDraw(ax, coords, nsteps):
    fSize = 11
    x0 = coords[0]
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=0.2)

    ax.text(x1[0] + 0.1, x1[1] - 0.2, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)

def draw(coords, nsteps, flag, f):
    fig, ax = plt.subplots()
    fig.suptitle('Steepest descent method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    steepDraw(ax, coords, nsteps)
    contourPlot(ax, f)
    plt.show()  # Вместо сохранения выводим изображение прямо на экран

# Пример использования (убедитесь, что координаты и функции правильно определены в вашем коде):
# draw(coords, nsteps, flag, f)
