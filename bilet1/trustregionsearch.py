import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv


def fH(X):
    # SPHERE func
    x = X[0]
    y = X[1]
    # sum = 0
    # for xi in xx:
    #     sum += xi[0] ** 2
    # return sum
    return x ** 2 + y ** 2


def dfH(X):
    # sum = 0
    # for xi in xx:
    #     sum += 2 * xi[0]
    # return sum
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = x * 2
    v[1] = y * 2
    return v



def fR(X):
    # ACKLEY
    a = 20
    b = 0.2
    c = 2 * np.pi
    x = X[0]
    y = X[1]
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20



def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2.82842712474619*x*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2) + np.pi*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.sin(2*np.pi*x)
    v[1] = 2.82842712474619*y*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2) + np.pi*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.sin(2*np.pi*y)
    return v


def goldensectionsearch(f, interval, tol):
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    L = b - a
    x1 = b - L / Phi
    x2 = a + L / Phi
    y1 = f(x1)
    y2 = f(x2)
    neval = 2
    xmin = x1
    fmin = y1

    # main loop
    while np.abs(L) > tol:
        if y1 > y2:
            a = x1
            xmin = x2
            fmin = y2
            x1 = x2
            y1 = y2
            L = b - a
            x2 = a + L / Phi
            y2 = f(x2)
            neval += 1
        else:
            b = x2
            xmin = x1
            fmin = y1
            x2 = x1
            y2 = y1
            L = b - a
            x1 = b - L / Phi
            y1 = f(x1)
            neval += 1

    answer_ = [xmin, fmin, neval]
    return answer_


def pparam(pU, pB, tau):
    if (tau <= 1):
        p = np.dot(tau, pU)
    else:
        p = pU + (tau - 1) * (pB - pU)
    return p


def doglegsearch(mod, g0, B0, Delta, tol):
    # dogleg local search
    xcv = np.dot(-g0.transpose(), g0) / np.dot(np.dot(g0.transpose(), B0), g0)
    pU = xcv * g0
    xcvb = inv(- B0)
    pB = np.dot(inv(- B0), g0)

    func = lambda x: mod(np.dot(x, pB))
    al = goldensectionsearch(func, [-Delta / norm(pB), Delta / norm(pB)], tol)[0]
    pB = al * pB
    func_pau = lambda x: mod(pparam(pU, pB, x))
    tau = goldensectionsearch(func_pau, [0, 2], tol)[0]
    pmin = pparam(pU, pB, tau)
    if norm(pmin) > Delta:
        pmin_dop = (Delta / norm(pmin))
        pmin = np.dot(pmin_dop, pmin)
    return pmin


def trustreg(f, df, x0, tol):
    eta = 0.1
    delta = 1
    delta_max = 0.1

    B = np.eye(len(x0))
    H = np.eye(len(x0))
    coords = [x0]
    radii = [delta]

    kmax = 1000
    d = 1000
    tol = 0.00001
    while (norm(d) >= tol) and (len(coords) < kmax):
        m_k = lambda p: (
                f(x0) + np.dot(np.array(p).transpose(), df(x0)) + 1 / 2 * np.dot(np.dot(np.array(p).transpose(), B),
                                                                                 np.array(p)))
        # Находим следующий шаг
        p = doglegsearch(m_k, df(x0), B, delta, tol)

        # Находим отношение реального уменьшения целевой функции к отношению предсказанного моделью
        ro = ((f(x0)[0] - f(x0 + p)[0]) / (m_k(0)[0][0] - m_k(p)[0][0]))

        # Переходим в новую точку
        if (ro > eta):
            x = x0 + p

            d = x - x0
            y = df(x) - df(x0)
            # -> Так правильнее по Максиму Гальченко. Тут исправлено вычисление матрицы H (лаба 11) Артур Искандарович говорил что так правильнее
            B = B - np.dot(np.dot(np.dot(B, d), d.transpose()), B) / np.dot(np.dot(d.transpose(), B),
                                                                                      d) + np.dot(y,
                                                                                                       y.transpose()) / np.dot(
                y.transpose(), d)
            # H = H + np.dot(d, d.transpose()) / np.dot(d.transpose(), y) - np.dot(
            #     np.dot(np.dot(H, y), y.transpose()), H) / np.dot(np.dot(y.transpose(), H), y)
            # B = inv(H)

            x0 = x
        else:
            x = x0

        # Обновляем радиус
        if (ro < 1 / 4):
            delta = 1 / 4 * delta
        elif (ro > 3 / 4) and (norm(p) == delta):
            delta = min(2 * delta, delta_max)
        radii.append(delta)

        coords.append(x)

    answer_ = [coords[-1], f(coords[-1]), len(coords), coords, radii]
    return answer_
