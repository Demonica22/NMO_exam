import numpy as np
import sys
from numpy.linalg import norm

np.seterr(all='warn')

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


def fBranin(X, a=1,
            b=5.1 / (4 * np.pi * 2),
            c=5 / np.pi,
            r=6,
            s=10,
            t=1 / (8 * np.pi)):
    x1 = X[0]
    x2 = X[1]
    value = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return value


def dfBranin(X, h=1e-3):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    x1 = np.array([x + h, y])
    x2 = np.array([x - h, y])
    y1 = np.array([x, y + h])
    y2 = np.array([x, y - h])
    v[0] = (fBranin(x1) - fBranin(x2)) / (2 * h)
    v[1] = (fBranin(y1) - fBranin(y2)) / (2 * h)
    return v


def H(x0, tol, df):
    n = len(x0)
    ddf = np.zeros((n, n))  # Создаем пустую матрицу Гессе
    deltaX = 0.1 * tol

    # Двойной цикл для заполнения элементов матрицы Гессе
    for i in range(n):
        for j in range(n):
            # Вычисляем градиенты в точках, сдвинутых на deltaX в положительную и отрицательную стороны
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += deltaX
            x_minus[i] -= deltaX
            grad_plus = df(x_plus)
            grad_minus = df(x_minus)
            # Вычисляем элемент матрицы Гессе по центральной разности
            ddf[i][j] = (grad_plus[j] - grad_minus[j]) / (2 * deltaX)

    return ddf


def nsearch(f, df, x0, tol):
    kmax = 1000
    neval = 0
    coords = [x0]
    x = x0

    # Итерация метода Ньютона
    k = 0
    while k < kmax:
        neval += 1
        grad = df(x0)  # Вычисляем градиент в текущей точке
        coords.append(x0)
        # КОП
        if np.linalg.norm(grad) < tol:
            break

        H0 = H(x0, tol, df)  # Вычисляем матрицу Гессе в текущей точке
        dx = np.linalg.solve(-H0, grad)  # Вычисляем направление движения
        x0 = x0 + dx

        k += 1

    xmin = x0
    fmin = f(xmin)

    return [xmin, fmin, neval, coords]
