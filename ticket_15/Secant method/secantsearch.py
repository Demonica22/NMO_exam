import numpy as np


def f1(x):
    return -2 * np.sin(np.sqrt(np.abs(x/2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))

def df1(x):
    term1 = -np.sin(np.sqrt(np.abs(x - 10)))
    term2 = -(x / np.sqrt(np.abs(x - 10))) * np.cos(np.sqrt(np.abs(x - 10))) * np.sign(x - 10)
    term3 = -np.sin(np.sqrt(np.abs(x/2 + 10))) / np.sqrt(np.abs(x/2 + 10)) * np.sign(x/2 + 10)
    return term1 + term2 + term3

def f2(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def df2(x):
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)


def ssearch(f, df, interval, tol):
    a, b = interval
    neval = 1
    coords = []

    # Задать [a1, b1] : f′(a1)f′(b1) < 0
    if df(a) * df(b) < 0:
        ak = a
        bk = b
    else:
        print("Error: f'(a) * f'(b) >= 0")
        return None

    # Цикл
    while np.abs(bk - ak) > tol:
        # Вычислить новую точку xk+1
        xk1 = bk - df(bk) * (bk - ak) / (df(bk) - df(ak))
        coords.append([xk1, ak, bk])
        neval += 1

        if df(xk1) * df(bk) < 0:
            ak = bk
            bk = xk1
        elif df(xk1) * df(ak) < 0:
            bk = ak
            ak = xk1
        else:
            break

    xmin = xk1
    fmin = f(xmin)
    return [xmin, fmin, neval, coords]
