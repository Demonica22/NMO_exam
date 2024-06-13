import numpy as np
import sys
from numpy.linalg import norm



# Первая функция
def fFirstFunc(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def dfFirstFunc(x):
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)


# Вторая функция
def fSecondFunc(x):
    return -2 * np.sin(np.sqrt(np.abs(x/2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))
def dfSecondFunc(x):
    term1 = -np.sin(np.sqrt(np.abs(x - 10)))
    term2 = -(x / np.sqrt(np.abs(x - 10))) * np.cos(np.sqrt(np.abs(x - 10))) * np.sign(x - 10)
    term3 = -np.sin(np.sqrt(np.abs(x/2 + 10))) / np.sqrt(np.abs(x/2 + 10)) * np.sign(x/2 + 10)
    return term1 + term2 + term3

#
# # F_HIMMELBLAU is a Himmelblau function
# # 	v = F_HIMMELBLAU(X)
# #	INPUT ARGUMENTS:
# #	X - is 2x1 vector of input variables
# #	OUTPUT ARGUMENTS:
# #	v is a function value
# def fH(X):
#     x = X[0]
#     y = X[1]
#     v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#     return v
#
#
# # DF_HIMMELBLAU is a Himmelblau function derivative
# # 	v = DF_HIMMELBLAU(X)
# #	INPUT ARGUMENTS:
# #	X - is 2x1 vector of input variables
# #	OUTPUT ARGUMENTS:
# #	v is a derivative function value
#
# def dfH(X):
#     x = X[0]
#     y = X[1]
#     v = np.copy(X)
#     v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
#     v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
#
#     return v
#
#
# # F_ROSENBROCK is a Rosenbrock function
# # 	v = F_ROSENBROCK(X)
# #	INPUT ARGUMENTS:
# #	X - is 2x1 vector of input variables
# #	OUTPUT ARGUMENTS:
# #	v is a function value
#
# def fR(X):
#     x = X[0]
#     y = X[1]
#     v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#     return v
#
#
# # DF_ROSENBROCK is a Rosenbrock function derivative
# # 	v = DF_ROSENBROCK(X)
# #	INPUT ARGUMENTS:
# #	X - is 2x1 vector of input variables
# #	OUTPUT ARGUMENTS:
# #	v is a derivative function value
#
# def dfR(X):
#     x = X[0]
#     y = X[1]
#     v = np.copy(X)
#     v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
#     v[1] = 200 * (y - x ** 2)
#     return v

#---------------------------------------------------------------------------------------------

def armijo(f, df, x, p, s, c1):
    fi = lambda alfa: f(x + alfa * p)
    dfi = lambda alfa: np.dot(p.transpose(), df(x + alfa * p))

    # Задаем начальные условия
    a = s
    b = 0.15
    fi0 = fi(0)
    dfi0 = dfi(0)
    kmax = 1000

    for k in range(kmax):
        if fi(a) <= fi0 + c1 * a * dfi0:
            break
        else:
            a = b * a

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
        ak = armijo(f, df, x0, p0, amax, c1)

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