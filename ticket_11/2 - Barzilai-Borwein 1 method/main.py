import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Определение функции Sphere и ее градиента
def sphere(X):
    x = X[0]
    y = X[1]
    return x**2 + y**2  # Функция сферы

def dsphere(X):
    x = X[0]
    y = X[1]
    return np.array([2*x, 2*y])  # Градиент функции сферы

# Определение функции McCormick и ее градиента
def mccormick(X):
    x, y = X
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1  # Функция МакКормика

def dmccormick(X):
    x, y = X
    df_dx = np.cos(x + y) + 2 * (x - y) - 1.5  # Частная производная по x
    df_dy = np.cos(x + y) - 2 * (x - y) + 2.5  # Частная производная по y
    return np.array([df_dx, df_dy])  # Градиент функции МакКормика

def bbsearch(f, df, x0, tol):
    delta = 0.1  # Стабилизационный параметр
    x = np.array(x0)  # Начальная точка
    neval = 0  # Количество вычислений функции
    kmax = 1000  # Максимальное количество итераций
    coords = [x]  # Список координат для визуализации
    alpha = 0.01  # Начальное значение шага
    g = df(x)  # Градиент в начальной точке
    deltaX = np.inf  # Изменение координат
    k = 0  # Счетчик итераций

    while (norm(deltaX) >= tol) and (k < kmax):  # Основной цикл
        g_norm = norm(g)  # Норма градиента
        if g_norm == 0:  # Если градиент равен нулю, прерываем цикл
            break
        alpha_stab = delta / g_norm  # Стабилизированное значение шага
        alpha = min(alpha, alpha_stab)  # Выбираем минимальное значение шага
        x_next = x - alpha * g  # Следующая точка
        deltaX = x_next - x  # Изменение координат
        x = x_next  # Обновляем текущую точку
        neval += 1  # Увеличиваем количество вычислений функции
        coords.append(x)  # Добавляем точку в список координат

        g_next = df(x)  # Градиент в новой точке
        delta_x = x - coords[-2]  # Изменение координат
        delta_g = g_next - g  # Изменение градиента
        numerator = np.sum(delta_x * delta_x)  # Числитель для шага Барзилая-Борвейна
        denominator = np.sum(delta_x * delta_g)  # Знаменатель для шага Барзилая-Борвейна
        if denominator != 0:
            alpha_bb = numerator / denominator  # Шаг Барзилая-Борвейна
        else:
            alpha_bb = alpha_stab  # Если знаменатель равен нулю, используем стабилизированное значение шага
        g_next_norm = norm(g_next)  # Норма градиента в новой точке
        if g_next_norm == 0:  # Если градиент равен нулю, прерываем цикл
            break
        alpha_stab = delta / g_next_norm  # Обновляем стабилизированное значение шага
        alpha = min(alpha_bb, alpha_stab)  # Выбираем минимальное значение шага

        g = g_next  # Обновляем градиент
        k += 1  # Увеличиваем счетчик итераций

    xmin = x  # Точка минимума
    fmin = f(xmin)  # Значение функции в точке минимума
    answer_ = [xmin, fmin, neval, coords]  # Возвращаем результат
    return answer_

# Начальные параметры и точка
x0_sphere = np.array([2.0, 2.0])  # Начальная точка для функции Sphere
x0_mccormick = np.array([0.0, 0.0])  # Начальная точка для функции McCormick
tol = 1e-6  # Точность

# Запуск метода для функции Sphere
result_sphere = bbsearch(sphere, dsphere, x0_sphere, tol)
print("Sphere Function:")
print("Minimum point:", result_sphere[0])
print("Minimum value:", result_sphere[1])
print("Number of evaluations:", result_sphere[2])

# Запуск метода для функции McCormick
result_mccormick = bbsearch(mccormick, dmccormick, x0_mccormick, tol)
print("\nMcCormick Function:")
print("Minimum point:", result_mccormick[0])
print("Minimum value:", result_mccormick[1])
print("Number of evaluations:", result_mccormick[2])

# Визуализация
x = np.linspace(-3, 3, 400)  # Создаем массив значений x
y = np.linspace(-3, 3, 400)  # Создаем массив значений y
X, Y = np.meshgrid(x, y)  # Создаем сетку значений

# Визуализация функции Sphere
Z_sphere = X**2 + Y**2  # Значения функции Sphere на сетке
plt.figure()
plt.contour(X, Y, Z_sphere, levels=50)  # Рисуем контурный график
coords_sphere = np.array(result_sphere[3])  # Получаем координаты из результата оптимизации
plt.plot(coords_sphere[:, 0], coords_sphere[:, 1], 'ro-')  # Рисуем путь оптимизации
plt.title('Optimization of Sphere Function')  # Заголовок графика
plt.xlabel('x')  # Подпись оси x
plt.ylabel('y')  # Подпись оси y
plt.colorbar()  # Добавляем цветовую шкалу
plt.show()  # Отображаем график

# Визуализация функции McCormick
Z_mccormick = np.sin(X + Y) + (X - Y) ** 2 - 1.5 * X + 2.5 * Y + 1  # Значения функции McCormick на сетке
plt.figure()
plt.contour(X, Y, Z_mccormick, levels=50)  # Рисуем контурный график
coords_mccormick = np.array(result_mccormick[3])  # Получаем координаты из результата оптимизации
plt.plot(coords_mccormick[:, 0], coords_mccormick[:, 1], 'ro-')  # Рисуем путь оптимизации
plt.title('Optimization of McCormick Function')  # Заголовок графика
plt.xlabel('x')  # Подпись оси x
plt.ylabel('y')  # Подпись оси y
plt.colorbar()  # Добавляем цветовую шкалу
plt.show()  # Отображаем график
