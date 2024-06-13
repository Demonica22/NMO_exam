import numpy as np

def f1(x):
    return -2 * np.sin(np.sqrt(np.abs(x/2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))

def df1(x):
    term1 = -np.sin(np.sqrt(np.abs(x - 10)))
    term2 = -(x / np.sqrt(np.abs(x - 10))) * np.cos(np.sqrt(np.abs(x - 10))) * np.sign(x - 10)
    term3 = -np.sin(np.sqrt(np.abs(x/2 + 10))) / np.sqrt(np.abs(x/2 + 10)) * np.sign(x/2 + 10)
    return term1 + term2 + term3

def ddf1(x):
    term1 = -2 * np.cos(np.sqrt(np.abs(x/2 + 10))) / (2 * np.abs(x/2 + 10))
    term2 = -np.cos(np.sqrt(np.abs(x - 10))) / np.abs(x - 10)
    term3 = -2 * x * np.sin(np.sqrt(np.abs(x - 10))) / (np.abs(x - 10) ** 1.5) * np.sign(x - 10)
    return term1 + term2 + term3

def f2(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def df2(x):
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)

def ddf2(x):
    return 2 + 2.5 * (np.pi ** 2) * np.cos(0.5 * np.pi * x)


def nsearch(f, df, ddf, tol, x0):
    # Инициализация переменных
    neval = 0
    coords = []

    # Начальная точка
    x = x0

    # Проведение метода Ньютона
    converged = False
    while not converged:
        # Добавление текущей координаты в массив
        coords.append(x)
        neval += 3

        # Вычисление обновления по методу Ньютона
        x_new = x - df(x) / ddf(x)

        # Проверка сходимости
        if abs(x_new - x) <= tol:
            xmin = x_new
            fmin = f(xmin)
            answer_ = [xmin, fmin, neval, coords]
            return answer_

        # Обновление x для следующей итерации
        x = x_new

