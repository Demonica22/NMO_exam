import numpy as np
from numpy.linalg import norm


# Определяем функции Sphere и Zakharov и их градиенты
def sphere(X):
    return np.sum(X ** 2)


def grad_sphere(X):
    return 2 * X


def zakharov(X):
    term1 = np.sum(X ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(X) + 1) * X) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(X) + 1) * X) ** 4
    return term1 + term2 + term3


def grad_zakharov(X):
    term1 = 2 * X
    term2 = np.arange(1, len(X) + 1) * X
    term2 = 2 * np.sum(0.5 * term2) * 0.5 * np.arange(1, len(X) + 1)
    term3 = np.arange(1, len(X) + 1) * X
    term3 = 4 * (np.sum(0.5 * term3) ** 3) * 0.5 * np.arange(1, len(X) + 1)
    return term1 + term2 + term3


# Метод Барзилая-Борвейна 2
def bbsearch(f, df, x0, tol):
    delta = 0.1
    x = np.array(x0)
    neval = 0
    kmax = 1000
    coords = [x]
    alpha = 0.01
    g = df(x)
    deltaX = np.inf
    k = 0

    while (norm(deltaX) >= tol) and (k < kmax):
        g_norm = norm(g)
        if g_norm == 0:
            break
        alpha_stab = delta / g_norm
        alpha = min(alpha, alpha_stab)
        x_next = x - alpha * g
        deltaX = x_next - x
        x = x_next
        neval += 1
        coords.append(x)

        g_next = df(x)
        delta_x = x - coords[-2]
        delta_g = g_next - g
        numerator = np.sum(delta_x * delta_x)
        denominator = np.sum(delta_x * delta_g)
        alpha_bb = numerator / denominator if denominator != 0 else alpha_stab
        g_next_norm = norm(g_next)
        if g_next_norm == 0:
            break
        alpha_stab = delta / g_next_norm
        alpha = min(alpha_bb, alpha_stab)

        g = g_next
        k += 1

    xmin = x
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_


# Начальные параметры и точка
tol = 1e-6
x0 = np.array([5.0, 5.0])

# Запуск метода для функции Sphere
result_sphere = bbsearch(sphere, grad_sphere, x0, tol)
print("Sphere Function Optimization:")
print("Minimum point:", result_sphere[0])
print("Minimum value:", result_sphere[1])
print("Number of evaluations:", result_sphere[2])

# Запуск метода для функции Zakharov
result_zakharov = bbsearch(zakharov, grad_zakharov, x0, tol)
print("\nZakharov Function Optimization:")
print("Minimum point:", result_zakharov[0])
print("Minimum value:", result_zakharov[1])
print("Number of evaluations:", result_zakharov[2])
