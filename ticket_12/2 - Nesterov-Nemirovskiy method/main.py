import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Игнорировать ошибки деления на ноль и недопустимые операции
np.seterr(divide='ignore', invalid='ignore')

# Определение функции сферы
def sphere_function(X):
    x = X[0]
    y = X[1]
    return x**2 + y**2

# Определение градиента функции сферы
def grad_sphere_function(X):
    x = X[0]
    y = X[1]
    return np.array([2*x, 2*y])

# Определение гессиана функции сферы
def hessian_sphere_function(X):
    return 2 * np.eye(len(X))

# Определение функции трехгорбого верблюда
def three_hump_camel_function(X):
    x = X[0]
    y = X[1]
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

# Определение градиента функции трехгорбого верблюда
def grad_three_hump_camel_function(X):
    x = X[0]
    y = X[1]
    grad = np.zeros_like(X)
    grad[0] = 4 * x - 4.2 * x**3 + x**5 + y
    grad[1] = x + 2 * y
    return grad

# Определение гессиана функции трехгорбого верблюда
def hessian_three_hump_camel_function(X):
    x = X[0]
    hessian = np.zeros((2, 2))
    hessian[0, 0] = 4 - 12.6 * x**2 + 5 * x**4
    hessian[0, 1] = 1
    hessian[1, 0] = 1
    hessian[1, 1] = 2
    return hessian

# Метод Нестерова-Немировского для оптимизации
def nesterov_nemirovski(f, grad_f, hessian_f, x0, tol=1e-5, kmax=1000):
    x = np.copy(x0)  # Начальная точка
    coords = [x]  # Список для хранения координат
    neval = 0  # Счетчик числа вычислений градиента
    k = 0  # Итерационный счетчик

    # Главный цикл оптимизации
    while norm(grad_f(x)) >= tol and k < kmax:
        gk = grad_f(x)  # Вычисление градиента
        Hk = hessian_f(x)  # Вычисление гессиана
        Hk_inv = np.linalg.inv(Hk)  # Обратная матрица гессиана
        delta_k = np.sqrt(gk.T @ Hk_inv @ gk)  # Дельта для адаптивного шага

        # Определение адаптивного шага alpha_k
        if delta_k <= 1/4:
            alpha_k = 1
        else:
            alpha_k = 1 / (1 + delta_k)

        x_next = x - alpha_k * Hk_inv @ gk  # Обновление точки

        # Проверка условия остановки
        if norm(grad_f(x_next)) < tol:
            break

        x = x_next  # Переход к следующей точке
        k += 1  # Увеличение счетчика итераций
        neval += 1  # Увеличение счетчика вычислений градиента
        coords.append(x)  # Добавление координаты в список

    xmin = x  # Точка минимума
    fmin = f(xmin)  # Значение функции в точке минимума

    return [xmin, fmin, neval, coords]  # Возвращение результатов

# Тестирование на функции сферы
x0_sphere = np.array([1.0, 1.0])
result_sphere = nesterov_nemirovski(sphere_function, grad_sphere_function, hessian_sphere_function, x0_sphere)
print("Sphere Function Optimization")
print("Minimum point:", result_sphere[0])
print("Function value at minimum point:", result_sphere[1])
print("Number of gradient evaluations:", result_sphere[2])

# Тестирование на функции трехгорбого верблюда
x0_camel = np.array([0.0, 3.0]) # если тут поставить 1 1, то работать не будет, тк не самосопряженная
result_camel = nesterov_nemirovski(three_hump_camel_function, grad_three_hump_camel_function, hessian_three_hump_camel_function, x0_camel)
print("\nThree-Hump Camel Function Optimization")
print("Minimum point:", result_camel[0])
print("Function value at minimum point:", result_camel[1])
print("Number of gradient evaluations:", result_camel[2])

# Визуализация функции сферы
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z_sphere = X**2 + Y**2

plt.figure()
plt.contour(X, Y, Z_sphere, levels=50)
coords_sphere = np.array(result_sphere[3])  # Получаем координаты из результата оптимизации
plt.plot(coords_sphere[:, 0], coords_sphere[:, 1], 'ro-')  # Рисуем путь оптимизации
plt.title('Optimization of Sphere Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()

# Визуализация функции трехгорбого верблюда
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z_camel = 2 * X**2 - 1.05 * X**4 + (X**6) / 6 + X * Y + Y**2

plt.figure()
plt.contour(X, Y, Z_camel, levels=50)
coords_camel = np.array(result_camel[3])  # Получаем координаты из результата оптимизации
plt.plot(coords_camel[:, 0], coords_camel[:, 1], 'ro-')  # Рисуем путь оптимизации
plt.title('Optimization of Three-Hump Camel Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
