import numpy as np
from numpy.linalg import norm


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

def fSixHumpCamel(X):
    x1 = X[0]
    x2 = X[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = (x1 * x2)
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

def dfSixHumpCamel(X, h=1e-3):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    x1 = np.array([x + h, y])
    x2 = np.array([x - h, y])
    y1 = np.array([x, y + h])
    y2 = np.array([x, y - h])
    v[0] = (fSixHumpCamel(x1) - fSixHumpCamel(x2)) / (2 * h)
    v[1] = (fSixHumpCamel(y1) - fSixHumpCamel(y2)) / (2 * h)
    return v

def grsearch(f, df, x0, tol):
    alpha = 0.01  # Параметр шага
    kmax = 1000
    neval = 0
    coords = [x0]  # Список координат x во время оптимизации

    # градиентнsq спуск
    for k in range(kmax):
        grad = df(x0)  # Вычисление градиента в текущей точке
        neval += 1
        x_next = x0 - alpha * grad
        coords.append(x_next)

        # КОП
        if norm(x_next - x0) < tol:
            xmin = x_next  # Минимум достигнут
            fmin = f(x_next)  # Значение функции в минимуме
            return [xmin, fmin, neval, coords]

        x0 = x_next  # Переход к следующей точке для новой итерации

    # Если максимальное число итераций достигнуто =>
    xmin = x0  # Последняя точка - минимум
    fmin = f(x0)  # Значение f в последней точке
    return [xmin, fmin, neval, coords]