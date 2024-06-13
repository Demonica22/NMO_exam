import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import inv

def He(x, tol, df):
    h = 2 * tol
    hessian = np.zeros((2, 2))

    df_plus_00 = df([x[0] + tol, x[1]])[0]
    df_minus_00 = df([x[0] - tol, x[1]])[0]
    hessian[0, 0] = (df_plus_00 - df_minus_00) / h

    df_plus_11 = df([x[0], x[1] + tol])[1]
    df_minus_11 = df([x[0], x[1] - tol])[1]
    hessian[1, 1] = (df_plus_11 - df_minus_11) / h

    df_plus_01 = df([x[0], x[1] + tol])[0]
    df_minus_01 = df([x[0], x[1] - tol])[0]
    hessian[0, 1] = (df_plus_01 - df_minus_01) / h

    df_plus_10 = df([x[0] + tol, x[1]])[1]
    df_minus_10 = df([x[0] - tol, x[1]])[1]
    hessian[1, 0] = (df_plus_10 - df_minus_10) / h

    return hessian

def fH(X):
    x = X[0]
    y = X[1]
    return x ** 2 + y ** 2

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = x * 2
    v[1] = y * 2
    return v

def fR(X):
    x = X[0]
    y = X[1]
    return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y)

def dfR(X):
    h = 1e-3
    x = X[0]
    y = X[1]
    v = np.copy(X)
    f = fR
    x1 = np.array([[x + h], [y]])
    x2 = np.array([[x - h], [y]])
    y1 = np.array([[x], [y + h]])
    y2 = np.array([[x], [y - h]])
    v[0] = (f(x1) - f(x2)) / (2 * h)
    v[1] = (f(y1) - f(y2)) / (2 * h)
    return v

def newton(f, df, x0, tol):
    neval = 1
    coords = [x0]
    kmax = 1000
    deltaX = 1000

    # ДЕМПФИРУЮЩИЙ ПАРАМЕТР (0;1]
    alfa = 0.5

    while (norm(deltaX) >= tol) and (neval < kmax):
        # МЕТОД НЬЮТОНА
        #x = x0 - np.linalg.lstsq(H(x0, tol, df), df(x0))[0]

        # ДЕМПФИРОВАННЫЙ МЕТОД НЬЮТОНА
        H = inv(He(x0, tol, df))
        x = x0 - alfa * np.dot(H, df(x0))

        neval += 1
        coords.append(x)
        deltaX = x - x0
        x0 = x

    xmin = x
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_