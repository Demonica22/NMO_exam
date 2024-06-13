import numpy
import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
from numpy import sin, pi


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


# def f(x, y):
#     # LEVY N.13
#     return sin(3 * pi * x) ** 2 + (x - 1) ** 2 * (1 + sin(3 * pi * y) * sin(3 * pi * y)) + (y - 1) * (y - 1) * (
#                 1 + sin(2 * pi * y) * sin(2 * pi * y))
def f(x,y):
    # LEVY
    w = [0, 0]

    w[0] = 1 + (x - 1) / 4
    w[1] = 1 + (y - 1) / 4

    term1 = (np.sin(np.pi * w[0])) ** 2
    term3 = (w[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[1])) ** 2)

    new = (w[0] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[0] + 1)) ** 2)

    return term1 + new + term3

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

def LBFGS(f, df, x0, tol=1e-4, max_iter=100, m=10):
    n = len(x0)
    x = x0
    coords = [x]
    c1 = 1e-4
    c2 = 0.1
    amax = 1


    s_list = []
    y_list = []
    rho_list = []

    I = np.eye(n)
    k = 0

    while k < max_iter:
        g = df(x)

        if np.linalg.norm(g) < tol:
            break

        q = g
        alpha = []

        for i in range(len(s_list) - 1, -1, -1):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            alpha_i = rho * np.dot(s.transpose(), q)
            alpha.append(alpha_i)
            q = q - alpha_i * y

        if len(s_list) > 0:
            s = s_list[-1]
            y = y_list[-1]
            gamma = np.dot(s.transpose(), y) / np.dot(y.transpose(), y)
            Hk0 = gamma * I
        else:
            Hk0 = I

        r = np.dot(Hk0, q)

        for i in range(len(s_list)):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            beta = rho * np.dot(y.transpose(), r)
            r = r + s * (alpha[len(s_list) - 1 - i] - beta)

        p = -r

        alpha_k = wolfesearch(f, df, coords[-1], p, amax, c1, c2)

        x_new = x + alpha_k * p

        s = x_new - x
        y = df(x_new) - g

        if np.dot(s.transpose(), y) > 1e-10:  # Ensure curvature condition
            if len(s_list) == m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / np.dot(y.transpose(), s))

        x = x_new
        coords.append(x)
        k += 1

    return [coords[-1], f(coords[-1]), len(coords), coords]