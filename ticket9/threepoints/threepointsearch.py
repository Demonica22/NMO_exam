import numpy as np
from scipy.misc import derivative

# Вторая функция
def f(x):
    return -2 * np.sin(np.sqrt(np.abs(x/2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))
def df(x):
    term1 = -np.sin(np.sqrt(np.abs(x - 10)))
    term2 = -(x / np.sqrt(np.abs(x - 10))) * np.cos(np.sqrt(np.abs(x - 10))) * np.sign(x - 10)
    term3 = -np.sin(np.sqrt(np.abs(x/2 + 10))) / np.sqrt(np.abs(x/2 + 10)) * np.sign(x/2 + 10)
    return term1 + term2 + term3

# Первая функция
# def f(x):
#     return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110
#
# def df(x):
#     return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)

def threepointsearch(interval,tol):
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

    print("DF: ", derivative(f, 5, dx=1e-6))

    a = interval[0]
    b = interval[1]
    xm = (a + b) / 2
    neval = 1
    coords = [xm]
    Lk = b - a

    while Lk > tol:
        x1 = a + Lk / 4
        x2 = b - Lk / 4

        f1 = f(x1)
        fm = f(xm)
        f2 = f(x2)

        neval += 3

        if f1 < fm:
            b = xm
            xm = x1
        elif fm <= f2:
            a = x1
            b = x2
        else:
            a = xm
            xm = x2
        coords.append(xm)
        Lk = b - a

    xmin = xm
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_