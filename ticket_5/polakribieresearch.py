import numpy as np
import sys
from numpy.linalg import norm

def fSphere(X):
    # SPHERE func
    x = X[0]
    y = X[1]
    return x ** 2 + y ** 2

def dfSphere(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * x
    v[1] = 2 * y
    return v

def fHolderTable(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))

    y = -np.abs(fact1 * fact2)
    return y

def dfHolderTable(xx):
    x1 = xx[0]
    x2 = xx[1]

    f = np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    sign_f = np.sign(f)

    g = 1 - np.sqrt(x1**2 + x2**2) / np.pi
    abs_g = np.abs(g)

    exp_term = np.exp(abs_g)

    # Partial derivative with respect to x1
    df_dx1 = (np.cos(x1) * np.cos(x2) * exp_term +
              np.sin(x1) * np.cos(x2) * exp_term * (g / abs_g) * (-x1 / (np.pi * np.sqrt(x1**2 + x2**2))))

    # Partial derivative with respect to x2
    df_dx2 = (-np.sin(x1) * np.sin(x2) * exp_term +
              np.sin(x1) * np.cos(x2) * exp_term * (g / abs_g) * (-x2 / (np.pi * np.sqrt(x1**2 + x2**2))))

    v = np.copy(xx)

    # Partial derivatives of y
    v[0] = -sign_f * df_dx1
    v[1] = -sign_f * df_dx2

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

def prsearch(f, df, x0, tol):
    # PRSEARCH searches for minimum using Polak-Ribiere method
    # 	answer_ = sdsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    c1 = tol
    c2 = 0.1

    coords = [x0]

    kmax = 1000
    amax = 3

    g0 = -df(x0)
    p0 = np.copy(g0)

    while (norm(g0) >= tol) and (len(coords) < kmax):
        # Ищем оптимальный размер шага
        ak = wolfesearch(f, df, x0, p0, amax, c1, c2)

        # метод спуска с использованием длины шага
        x0 = x0 + ak*p0

        gk = -df(x0)

        g_diff_transposed = (gk - g0).transpose()

        # Обновляем значение направления по методу Полака-Рибьера
        denom = np.dot(np.transpose(g0), g0)
        if denom != 0:
            b = np.dot(g_diff_transposed, gk)/ denom
        else:
            b = 0


        p0 = gk + b * p0

        g0 = -df(x0)
        coords.append(x0)

    answer_ = [x0, f(x0), len(coords), coords]
    return answer_