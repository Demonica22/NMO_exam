import numpy as np
from numpy import sin, cos, exp
from numpy import real as re
from numpy import imag as im
from sympy import Abs, Derivative, sqrt, sign, pi, atan2


# Первая функция
# def f(x):
#     return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110
#
# def df(x):
#     return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)


# Вторая функция
def f(x):
    return -2 * np.sin(np.sqrt(np.abs(x/2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))
def df(x):
    term1 = -np.sin(np.sqrt(np.abs(x - 10)))
    term2 = -(x / np.sqrt(np.abs(x - 10))) * np.cos(np.sqrt(np.abs(x - 10))) * np.sign(x - 10)
    term3 = -np.sin(np.sqrt(np.abs(x/2 + 10))) / np.sqrt(np.abs(x/2 + 10)) * np.sign(x/2 + 10)
    return term1 + term2 + term3

def threepointsearch(interval, tol):
    # searches for minimum using bisection method
    # arguments: bisectionsearch(f,df,interval,tol)
    # f - an objective function
    # df -  an objective function derivative
    # interval = [a, b] - search interval
    #tol - tolerance for both range and function value
    # output: [xmin, fmin, neval, coords]
    # xmin - value of x in fmin
    # fmin - minimul value of f
    # neval - number of function evaluations
    # coords - array of x values found during optimization

    neval = 0
    coords = []

    while abs((interval[1] - interval[0]) > tol) and (abs(df(interval[0])) > tol):
        x = (interval[0] + interval[1]) / 2
        if df(x) > 0:
            interval[1] = x
        else:
            interval[0] = x

        coords.append(x)
        neval += 1

    xmin = coords[-1]
    fmin = df(coords[-1])

    answer_ = [xmin, fmin, neval, coords]
    return answer_