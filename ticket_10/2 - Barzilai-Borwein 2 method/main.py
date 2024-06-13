import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Определяем функции Sphere и Zakharов и их градиенты
def sphere(X):
    return np.sum(X ** 2)  # Функция Sphere

def grad_sphere(X):
    return 2 * X  # Градиент функции Sphere

def zakharov(X):
    term1 = np.sum(X ** 2)
    term2 = np.sum(0.5 * np.arange(1, len(X) + 1) * X) ** 2
    term3 = np.sum(0.5 * np.arange(1, len(X) + 1) * X) ** 4
    return term1 + term2 + term3  # Функция Zakharov

def grad_zakharov(X):
    term1 = 2 * X
    term2 = np.arange(1, len(X) + 1) * X
    term2 = 2 * np.sum(0.5 * term2) * 0.5 * np.arange(1, len(X) + 1)
    term3 = np.arange(1, len(X) + 1) * X
    term3 = 4 * (np.sum(0.5 * term3) ** 3) * 0.5 * np.arange(1, len(X) + 1)
    return term1 + term2 + term3  # Градиент функции Zakharov

# Метод Барзилая-Борвейна 2
def bbsearch(f, df, x0, tol):
    delta = 0.1  # Шаг для стабилизации альфа
    x = np.array(x0)  # Начальная точка
    neval = 0  # Количество вычислений функции
    kmax = 1000  # Максимальное количество итераций
    coords = [x]  # Координаты для визуализации
    alpha = 0.01  # Начальное значение альфа
    g = df(x)  # Градиент в начальной точке
    deltaX = np.inf  # Начальное значение изменения координат
    k = 0  # Счетчик итераций

    while (norm(deltaX) >= tol) and (k < kmax):  # Основной цикл
        g_norm = norm(g)  # Норма градиента
        if g_norm == 0:
            break
        alpha_stab = delta / g_norm  # Стабилизированное значение альфа
        alpha = min(alpha, alpha_stab)  # Выбор минимального альфа
        x_next = x - alpha * g  # Следующая точка
        deltaX = x_next - x  # Изменение координат
        x = x_next  # Обновление текущей точки
        neval += 1  # Увеличение количества вычислений функции
        coords.append(x)  # Сохранение координат

        g_next = df(x)  # Вычисление градиента в новой точке
        delta_x = x - coords[-2]  # Изменение координат
        delta_g = g_next - g  # Изменение градиента
        numerator = np.sum(delta_x * delta_g)  # Числитель для альфа Барзилая-Борвейна
        denominator = np.sum(delta_g * delta_g)  # Знаменатель для альфа Барзилая-Борвейна
        alpha_bb = numerator / denominator if denominator != 0 else alpha_stab  # Альфа Барзилая-Борвейна
        g_next_norm = norm(g_next)  # Норма нового градиента
        if g_next_norm == 0:
            break
        alpha_stab = delta / g_next_norm  # Стабилизированное значение альфа
        alpha = min(alpha_bb, alpha_stab)  # Выбор минимального альфа

        g = g_next  # Обновление градиента
        k += 1  # Увеличение счетчика итераций

    xmin = x  # Минимальная точка
    fmin = f(xmin)  # Значение функции в минимальной точке
    answer_ = [xmin, fmin, neval, coords]  # Результат
    return answer_  # Возвращение результата

# Начальные параметры и точка
tol = 1e-6  # Точность
x0 = np.array([5.0, 5.0])  # Начальная точка

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

# Визуализация
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Визуализация функции Sphere
Z_sphere = X**2 + Y**2
plt.figure()
plt.contour(X, Y, Z_sphere, levels=50)
coords_sphere = np.array(result_sphere[3])
plt.plot(coords_sphere[:, 0], coords_sphere[:, 1], 'ro-')
plt.title('Optimization of Sphere Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()

# Визуализация функции Zakharov
Z_zakharov = X**2 + (0.5 * X + 0.5 * Y)**2 + (0.5 * X + 0.5 * Y)**4
plt.figure()
plt.contour(X, Y, Z_zakharov, levels=50)
coords_zakharov = np.array(result_zakharov[3])
plt.plot(coords_zakharov[:, 0], coords_zakharov[:, 1], 'ro-')
plt.title('Optimization of Zakharov Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
