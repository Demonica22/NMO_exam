import numpy as np
import sys
from numpy.linalg import norm
from numpy import sin, cos, exp
from numpy import real as re

from numpy import imag as im
from sympy import Abs, Derivative, sqrt, sign, pi, atan2

from scipy.misc import derivative


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


def f(x, y):
    return -0.0001 * (
                np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x ** 2 + y ** 2) / np.pi)))) + 1) ** 0.1


def fR(X):
    x = X[0]
    y = X[1]
    v = f(x, y)
    return v


def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    h = 1e-3
    v[0] = (f(x + h, y) - f(x - h, y)) / (2 * h)
    v[1] = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return v


def zoom(phi, dphi, alo, ahi, c1, c2):
    j = 1
    jmax = 1000
    while j < jmax:
        a = cinterp(phi, dphi, alo, ahi)
        if phi(a) > phi(0) + c1 * a * dphi(0) or phi(a) >= phi(alo):
            ahi = a
        else:
            if abs(dphi(a)) <= -c2 * dphi(0):
                return a  # a is found
            if dphi(a) * (ahi - alo) >= 0:
                ahi = alo
            alo = a
        j += 1
    return a


def cinterp(phi, dphi, a0, a1):
    if np.isnan(dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1))) or (a0 - a1) == 0:
        a = a0
        return a

    d1 = dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1)) / (a0 - a1)
    if np.isnan(np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))):
        a = a0
        return a
    d2 = np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))
    a = a1 - (a1 - a0) * (dphi(a1) + d2 - d1) / (dphi(a1) - dphi(a0) + 2 * d2)

    return a


def wolfesearch(f, df, x0, p0, amax, c1, c2):
    a = amax
    aprev = 0
    phi = lambda x: f(x0 + x * p0)
    dphi = lambda x: np.dot(p0.transpose(), df(x0 + x * p0))

    phi0 = phi(0)
    dphi0 = dphi(0)
    i = 1
    imax = 1000
    while i < imax:
        if (phi(a) > phi0 + c1 * a * phi0) or ((phi(a) >= phi(aprev)) and (i > 1)):
            a = zoom(phi, dphi, aprev, a, c1, c2)
            return a

        if abs(dphi(a)) <= -c2 * dphi0:
            return a  # a is found already

        if dphi(a) >= 0:
            a = zoom(phi, dphi, a, aprev, c1, c2)
            return a

        a = cinterp(phi, dphi, a, amax)
        i += 1

    return a


def dfpsearch(f, df, x0, tol):
    coords = [x0]
    kmax = 1000
    c1 = tol
    c2 = 0.1
    amax = 3
    dx = 1
    H = np.ones((2, 2))
    H[0][1] = 0
    H[1][0] = 0

    while (norm(dx) >= tol) and (len(coords) < kmax):
        g = df(coords[-1])

        p = np.dot(-H, g)

        ak = wolfesearch(f, df, coords[-1], p, amax, c1, c2)

        dx = ak * p

        coords.append(coords[-1] + dx)

        dy = df(coords[-1]) - g

        dx_transposed = dx.transpose()
        dy_transposed = dy.transpose()

        second = np.dot(dx, dx_transposed) / np.dot(dx_transposed, dy)
        third = np.dot(H, np.dot(np.dot(dy, dy_transposed), H)) / np.dot(np.dot(dy_transposed, H), dy)

        H += (second - third)


    answer_ = [coords[-1], f(coords[-1]), len(coords), coords]
    return answer_
