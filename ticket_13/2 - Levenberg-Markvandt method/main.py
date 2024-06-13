import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def levenberg_marquardt(f, jac, x0):
    result = least_squares(f, x0, jac=jac, method='lm')
    return result.x, result.cost

# Modified Sphere function to return residuals
def sphere_residuals(x):
    return np.array([x[0], x[1]])

def sphere_jac_residuals(x):
    return np.eye(len(x))

# Modified Branin function to return residuals with parameters alpha and v
def branin_residuals(x, alpha=100, v=3):
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    f1 = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
    f2 = alpha * (x[0] + x[1]) ** v  # Добавляем дополнительный резидуал с параметрами alpha и v
    return np.array([f1, f2])

def branin_jac_residuals(x, alpha=100, v=3):
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    grad = np.zeros((2, x.size))
    grad[0, 0] = 2 * a * (x[1] - b * x[0] ** 2 + c * x[0] - r) * (-2 * b * x[0] + c) - s * (1 - t) * np.sin(x[0])
    grad[0, 1] = 2 * a * (x[1] - b * x[0] ** 2 + c * x[0] - r)
    grad[1, 0] = alpha * v * (x[0] + x[1]) ** (v - 1)  # Добавляем градиент для дополнительного резидуала с параметрами alpha и v
    grad[1, 1] = alpha * v * (x[0] + x[1]) ** (v - 1)
    return grad

# Initial guess
x0_sphere = np.array([1.0, 1.0])
x0_branin = np.array([1.0, 1.0])

# Optimize Sphere function
opt_x_sphere, cost_sphere = levenberg_marquardt(sphere_residuals, sphere_jac_residuals, x0_sphere)
print(f"Optimized parameters for Sphere function: {opt_x_sphere}")
print(f"Cost for Sphere function: {cost_sphere}")

# Optimize Branin function with parameters alpha and v
opt_x_branin, cost_branin = levenberg_marquardt(
    lambda x: branin_residuals(x, alpha=100, v=3),
    lambda x: branin_jac_residuals(x, alpha=100, v=3),
    x0_branin
)
print(f"Optimized parameters for Branin function: {opt_x_branin}")
print(f"Cost for Branin function: {cost_branin}")

# Plotting the functions
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Sphere function plot
Z_sphere = X**2 + Y**2
plt.figure()
plt.contour(X, Y, Z_sphere, levels=50)
plt.plot(opt_x_sphere[0], opt_x_sphere[1], 'ro')
plt.title('Optimization of Sphere Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()

# Branin function plot
a = 1.0
b = 5.1 / (4 * np.pi ** 2)
c = 5 / np.pi
r = 6.0
s = 10.0
t = 1 / (8 * np.pi)
Z_branin = a * (Y - b * X ** 2 + c * X - r) ** 2 + s * (1 - t) * np.cos(X) + s
plt.figure()
plt.contour(X, Y, Z_branin, levels=50)
plt.plot(opt_x_branin[0], opt_x_branin[1], 'ro')
plt.title('Optimization of Branin Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
