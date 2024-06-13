import numpy as np
import matplotlib.pyplot as plt
import random

def drawdf(df, a, b, h, ax1):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b, h)
    y = [df(i) for i in x_]
    ax1.plot(x_, y, lw=1, c=color)
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)
    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")
    ax1.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    color = (random.random(), random.random(), random.random())
    ax1.plot([a, b], [df(a), df(b)], marker='s', ms=3, c=color, lw=1)
    ax1.set_xlim([-2.1, 5.1])

def drawf(f, a, b, h, ax2):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b, h)
    y = [f(i) for i in x_]
    ax2.plot(x_, y, lw=1, c=color)
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f(x)")
    ax2.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax2.set_xlim([-2.1, 5.1])

def drawpointsDF(df, coord, ax1, i):
    ax1.plot(coord[0], df(coord[0]), marker='*')
    ax1.text(coord[0], df(coord[0]) + 3, str(i + 1))
    new_color = (random.random(), random.random(), random.random())
    ax1.scatter([coord[1], coord[2]], [df(coord[1]), df(coord[2])], marker='o', c=new_color)
    ax1.plot([coord[1], coord[2]], [df(coord[1]), df(coord[2])], marker='s', ms=2, c=new_color, lw=1)
    ax1.plot([coord[0], coord[0]], [df(coord[0]), 0], marker='s', ms=2, c=new_color, lw=1)

def drawpointsF(f, coord, ax2, i):
    new_color = (random.random(), random.random(), random.random())
    ax2.plot(coord[0], f(coord[0]), marker='*')
    ax2.text(coord[0], f(coord[0]) + 1.4, str(i + 1))
    ax2.scatter([coord[1], coord[2]], [f(coord[1]), f(coord[2])], marker='o', c=new_color)

def secantsearchsecants(f, df, coords, interval):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle('Secant search visualisation')
    a = interval[0]
    b = interval[1]
    h = (b - a) / 100
    drawdf(df, a, b, h, ax1)
    drawf(f, a, b, h, ax2)
    plt.show()  # Show the main function and derivative plots
    for i, coord in enumerate(coords):
        drawpointsDF(df, coord, ax1, i)
        drawpointsF(f, coord, ax2, i)
        plt.show()  # Show each step in the secant method

