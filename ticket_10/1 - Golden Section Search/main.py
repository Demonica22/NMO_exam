import numpy as np
import matplotlib.pyplot as plt
from goldensearch import gsearch

# Определение функций
def f1(x):
    return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def f2(x):
    return x**2 - 10 * np.cos(0.5 * np.pi * x) - 110

# Интервал поиска и точность
interval = [-2, 10]
tol = 0.01e-3

# Тестирование функции f1
result_f1 = gsearch(f1, interval, tol)
print(f"Function f1: x_min = {result_f1[0]}, f_min = {result_f1[1]}, evaluations = {result_f1[2]}")

# Тестирование функции f2
result_f2 = gsearch(f2, interval, tol)
print(f"Function f2: x_min = {result_f2[0]}, f_min = {result_f2[1]}, evaluations = {result_f2[2]}")

# Визуализация результатов
x = np.linspace(interval[0], interval[1], 400)
y1 = f1(x)
y2 = f2(x)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x, y1, label='f1(x)')
plt.scatter(result_f1[0], result_f1[1], color='red', label='Minimum')
plt.title('Function f1')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y2, label='f2(x)')
plt.scatter(result_f2[0], result_f2[1], color='red', label='Minimum')
plt.title('Function f2')
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend()

plt.tight_layout()
plt.show()
