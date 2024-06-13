import numpy as np
import sys
from numpy.linalg import norm

def goldensectionsearch(f1dim, interval, tol):
    a, b = interval
    rho = (3 - np.sqrt(5)) / 2
    x1 = a + rho * (b - a)
    x2 = b - rho * (b - a)
    f1 = f1dim(x1)
    f2 = f1dim(x2)
    while (b - a) / 2 > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + rho * (b - a)
            f2 = f1
            f1 = f1dim(x1)
        else:
            a = x1
            x1 = x2
            x2 = b - rho * (b - a)
            f1 = f2
            f2 = f1dim(x2)
    return (a + b) / 2


def fSphere(X):
    x = X[0]
    y = X[1]
    return x ** 2 + y ** 2

def dfSphere(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = x * 2
    v[1] = y * 2
    return v


def fSumOfDifferentPowersFunction(X):
    x = X[0]
    y = X[1]
    v = np.abs(x)**2 + np.abs(y)**3
    return v


def dfSumOfDifferentPowersFunction(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * x
    v[1] = 3 * y**2
    return v

def sdsearch(f, df, x0, tol):

    kmax = 1000
    coords = []
    x = x0
    neval = 0  # Счетчик числа вычислений производной
    for k in range(kmax):
        coords.append(x)
        gradient = df(x)
        phi = lambda alpha: f(x - alpha * gradient) # Поиск оптимального шага (f1dim)
        interval = [-2, 10]
        alpha_star = goldensectionsearch(phi, interval, tol)
        x_new = x - alpha_star * gradient
        neval += 1

        # Проверка критерия остановки:
		# если норма разности двух последовательных точек меньше заданной точности
        # и не достигнуто максимальное число итераций, то прекращаем оптимизацию и возвращаем результат
        if np.linalg.norm(x_new - x) < tol and k < kmax:
            xmin = x_new
            fmin = f(x_new)
            return [xmin, fmin, neval, coords]
        x = x_new
    xmin = x
    fmin = f(x)
    return [xmin, fmin, neval, coords]