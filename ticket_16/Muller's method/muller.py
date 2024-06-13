import numpy as np
import matplotlib.pyplot as plt


def f(x):
    # Определите здесь свою функцию, для которой вы ищете корень.
    return -2 * np.sin(np.sqrt(np.abs(x / 2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))


def df(x):
    # Производная функции f с использованием метода конечных разностей.
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2 * h)


def muller(f, df, x0, delta, max_iter=1000):
    x = x0  # начальное приближение, может быть комплексным числом
    for k in range(max_iter):
        g_x = df(x)
        g_x_minus_2 = df(x - 2)
        g_x_minus_1 = df(x - 1)

        # Квадратичная интерполяция
        w = df(x) ** 2 - 4 * f(x) * ((f(x) - f(x - 2)) / (x - (x - 2)) - (f(x - 1) - f(x)) / ((x - 1) - x))
        sqrt_term = np.sqrt(w)

        # Выбор знаменателя с большим модулем
        d_plus = df(x) + sqrt_term
        d_minus = df(x) - sqrt_term
        if abs(d_minus) < abs(d_plus):
            denom = d_plus
        else:
            denom = d_minus

        # Обновление значения x
        x_new = x - 2 * g_x / denom
        if np.abs(x_new - x) <= delta:
            return x_new
        x = x_new

    raise ValueError("Максимальное количество итераций достигнуто")


# Начальное приближение и интервал
x0 = 1.0 + 1.0j  # начальное приближение с комплексной частью
delta = 1e-3  # Точность

root = muller(f, df, x0, delta)
print(f"Найден корень: {root}")

# Визуализация не может быть выполнена для комплексных чисел прямо таким же образом, поскольку это требует рассмотрения в плоскости комплексных чисел.
