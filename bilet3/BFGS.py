import numpy
import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
from numpy import sin, fabs, sqrt


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
    # EGG HOLDER
    a = sqrt(fabs(y + x / 2 + 47))
    b = sqrt(fabs(x - (y + 47)))
    c = -(y + 47) * sin(a) - x * sin(b)
    return c
# def f(x1,x2):
#     term1 = -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1 / 2 + 47)))
#     term2 = -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
#
#     return term1 + term2

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
def cinterp(phi, dphi, a0, a1):
    if np.isnan(dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1))) or (a0 - a1) == 0:
        a = a0
        return a

    d1 = dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1)) / (a0 - a1)
    if np.isnan(np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))):
        a = a0
        return a
    d2 = np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))
    print((dphi(a1) - dphi(a0) + 2 * d2))
    a = a1 - (a1 - a0) * (dphi(a1) + d2 - d1) / (dphi(a1) - dphi(a0) + 2 * d2)

    return a
def zoom(phi, phi_grad, alpha_lo, alpha_hi, c1, c2, max_iter=100):
    i = 0
    while True:
        alpha_j = (alpha_lo + alpha_hi) / 2.0
        phi_alpha_j = phi(alpha_j)

        if (phi_alpha_j > phi(0) + c1 * alpha_j * phi_grad(0)) or (phi_alpha_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            phi_grad_alpha_j = phi_grad(alpha_j)
            if np.abs(phi_grad_alpha_j) <= -c2 * phi_grad(0):
                return alpha_j
            if phi_grad_alpha_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
        i += 1
        if i >= max_iter:
            return None

def wolf_line_search(f, grad, x, p, max_iter=100, c1=10 ** -4, c2=0.9, alpha_1=1.0):
    alpha_max = 1000
    def phi(alpha):
        return f(x + alpha * p)

    def phi_grad(alpha):
        return np.dot(grad(x + alpha * p).T, p)

    alpha_i_1 = 0
    alpha_i = alpha_1

    for i in range(1, max_iter + 1):
        phi_alpha_i = phi(alpha_i)
        if (phi_alpha_i > phi(0) + c1 * alpha_i * phi_grad(0)) or (i > 1 and phi_alpha_i >= phi(alpha_i_1)):
            return zoom(phi, phi_grad, alpha_i_1, alpha_i, c1, c2)

        phi_grad_alpha_i = phi_grad(alpha_i)
        if np.abs(phi_grad_alpha_i) <= -c2 * phi_grad(0):
            return alpha_i
        if phi_grad_alpha_i >= 0:
            return zoom(phi, phi_grad, alpha_i, alpha_i_1, c1, c2)
        alpha_i_1 = alpha_i
        alpha_i = min(2 * alpha_i, alpha_max)

    if i == max_iter:
        return None

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

def armijo(f, df, x, p, alpha=1, c1=1e-4, c2=0.9):
    while f(x + alpha * p) > f(x) + c1 * alpha * np.dot(df(x).transpose(), p) or np.dot(df(x + alpha * p).transpose(), p) < c2 * np.dot(df(x).transpose(), p):
        alpha *= 0.15
    return alpha

def BFGS(f, df, x0, tol=1e-8, kmax=1000):
    n = len(x0)
    x = x0
    coords = [x]

    Hk = np.eye(n)  # Initial approximation of the inverse Hessian matrix
    k = 0
    c1 = 1e-5
    c2 = 1e-2
    amax = 1
    dk = 1000

    while (norm(dk) >= tol) and k < kmax:
        gk = df(x)

        if np.linalg.norm(gk) < tol:
            break

        pk = -np.dot(Hk, gk)

        alpha = wolfesearch(f, df, coords[-1], pk, amax, c1, c2)
        # alpha = armijo(f, df, x, pk)
        dk = alpha * pk
        x_new = x + dk
        sk = x_new - x
        x = x_new

        gk_new = df(x)
        yk = gk_new - gk

        rho_k = 1.0 / np.dot(yk.transpose(), sk)

        I = np.eye(n)
        term1 = I - rho_k * np.outer(sk, yk)
        term2 = I - rho_k * np.outer(yk, sk)
        Hk = np.dot(term1, np.dot(Hk, term2)) + rho_k * np.outer(sk, sk)

        coords.append(x)
        k += 1

    return [coords[-1], f(coords[-1]), len(coords), coords]

