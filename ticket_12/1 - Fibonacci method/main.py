import numpy as np


def fibonacci_search(f, a, b, tol=1e-5):
    def fibonacci_numbers(n):
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib

    n = 1
    while fibonacci_numbers(n)[-1] < (b - a) / tol:
        n += 1

    fib = fibonacci_numbers(n + 1)

    k = 0
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    while k < n - 2:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[n - k - 2] / fib[n - k - 1]) * (b - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[n - k - 3] / fib[n - k - 1]) * (b - a)
            f1 = f(x1)
        k += 1

    if f1 < f2:
        return (a + x2) / 2, k
    else:
        return (x1 + b) / 2, k


# Функции для минимизации
def f1(x):
    return -2 * np.sin(np.sqrt(np.abs(x / 2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))


def f2(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110


# Интервал поиска
interval = [-2, 10]

# Минимизация функции f1
min_f1, iter_f1 = fibonacci_search(f1, interval[0], interval[1])
print(f"Минимум функции f1(x) в интервале {interval}: x = {min_f1}, f1(x) = {f1(min_f1)}, итераций: {iter_f1}")

# Минимизация функции f2
min_f2, iter_f2 = fibonacci_search(f2, interval[0], interval[1])
print(f"Минимум функции f2(x) в интервале {interval}: x = {min_f2}, f2(x) = {f2(min_f2)}, итераций: {iter_f2}")
