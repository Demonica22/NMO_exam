import numpy as np

def fibonacci_search(f, a, b, tol=1e-5):
    def fibonacci_numbers(n):
        fib = [1, 1]  # Инициализация первых двух чисел Фибоначчи
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])  # Заполнение списка чисел Фибоначчи до n-го числа
        return fib

    n = 1
    while fibonacci_numbers(n)[-1] < (b - a) / tol:  # Определение необходимого количества чисел Фибоначчи
        n += 1

    fib = fibonacci_numbers(n + 1)  # Генерация чисел Фибоначчи, необходимых для алгоритма

    k = 0
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)  # Вычисление первой внутренней точки
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)  # Вычисление второй внутренней точки
    f1 = f(x1)  # Значение функции в точке x1
    f2 = f(x2)  # Значение функции в точке x2

    while k < n - 2:  # Основной цикл алгоритма поиска Фибоначчи
        if f1 > f2:  # Если значение в x1 больше значения в x2
            a = x1  # Сдвигаем начало интервала
            x1 = x2  # Перемещаем x1 на место x2
            f1 = f2  # Значение функции в x1 равно f2
            x2 = a + (fib[n - k - 2] / fib[n - k - 1]) * (b - a)  # Новая внутренняя точка x2
            f2 = f(x2)  # Вычисляем значение функции в новой точке x2
        else:  # Если значение в x1 меньше или равно значению в x2
            b = x2  # Сдвигаем конец интервала
            x2 = x1  # Перемещаем x2 на место x1
            f2 = f1  # Значение функции в x2 равно f1
            x1 = a + (fib[n - k - 3] / fib[n - k - 1]) * (b - a)  # Новая внутренняя точка x1
            f1 = f(x1)  # Вычисляем значение функции в новой точке x1
        k += 1  # Увеличиваем счетчик итераций

    if f1 < f2:  # Проверяем, где находится минимум
        return (a + x2) / 2, k  # Возвращаем найденный минимум и количество итераций
    else:
        return (x1 + b) / 2, k  # Возвращаем найденный минимум и количество итераций

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
