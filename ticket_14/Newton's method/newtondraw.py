import numpy as np
import matplotlib.pyplot as plt
from newton import *
import random

def drawdf(df, a, b, h, ax1):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b + h, h)
    y = [df(i) for i in x_]
    ax1.plot(x_, y, lw=1, c=color)
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)
    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")
    ax1.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax1.set_xlim([-2.1, 7.1])
    ax1.set_ylim([-30, 40])

def drawf(f, a, b, h, ax2):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b + h, h)
    y = [f(i) for i in x_]
    ax2.plot(x_, y, lw=1, c=color)
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f(x)")
    ax2.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax2.set_xlim([-2.1, 7.1])

def drawpointsDF(df, ddf, coord, ax1, a, b, h, i):
    ax1.plot(coord, df(coord), marker='*')
    ax1.text(coord, df(coord) + 3, str(i + 1), backgroundcolor='white')
    x_ = np.arange(a, b + h, h)
    new_color = (random.random(), random.random(), random.random())
    k = ddf(coord)
    br = df(coord)
    yt = k * (x_ - coord) + br
    ax1.plot(x_, yt, c=new_color, lw=1)
    ax1.plot([coord, coord], [0, df(coord)], c=(0, 0, 0), lw=0.5)

def drawpointsF(f, coord, ax2, i):
    ax2.plot(coord, f(coord), marker='*')
    ax2.text(coord, f(coord) + 1, str(i + 1), backgroundcolor='white')

def newtondrawfig(interval, coords):
    fig = plt.figure(figsize=[20, 10])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle('Newton visualisation')
    a = interval[0]
    b = interval[1]
    h = (b - a) / 100
    drawdf(df1, a, b, h, ax1)  # Assume df1 is defined
    drawf(f1, a, b, h, ax2)  # Assume f1 is defined
    for i, coord in enumerate(coords):
        drawpointsDF(df1, ddf1, coord, ax1, a, b, h, i)  # Assume ddf1 is defined
        drawpointsF(f1, coord, ax2, i)
    plt.show()
