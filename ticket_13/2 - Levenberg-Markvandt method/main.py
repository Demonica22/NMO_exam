import numpy as np
import matplotlib.pyplot as plt


def levenberg_marquardt(f, grad_f, hess_f, x0, alpha0=1.0, v=2.0, epsilon=1e-6, delta=1e-6):
    xk = x0
    alpha_k = alpha0
    I = np.eye(len(x0))

    while True:
        # Step 1: Update xk
        hess_inv = np.linalg.inv(alpha_k * I + hess_f(xk))
        xk_new = xk - np.dot(hess_inv, grad_f(xk))

        # Step 2: Check if f(xk_new) < f(xk)
        if f(xk_new) < f(xk):
            alpha_k = alpha_k / v
        else:
            alpha_k = alpha_k * v

        # Check stopping condition
        if np.linalg.norm(grad_f(xk_new)) < epsilon or np.linalg.norm(xk_new - xk) < delta:
            return xk_new, f(xk_new)

        xk = xk_new


# Define the Sphere function and its derivatives
def sphere_function(x):
    return np.sum(x ** 2)


def sphere_grad(x):
    return 2 * x


def sphere_hess(x):
    return 2 * np.eye(len(x))


# Define the Branin function and its derivatives
def branin_function(x):
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


def branin_grad(x):
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    grad = np.zeros(2)
    grad[0] = 2 * a * (x[1] - b * x[0] ** 2 + c * x[0] - r) * (-2 * b * x[0] + c) - s * (1 - t) * np.sin(x[0])
    grad[1] = 2 * a * (x[1] - b * x[0] ** 2 + c * x[0] - r)
    return grad


def branin_hess(x):
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    hess = np.zeros((2, 2))
    hess[0, 0] = 2 * a * ((-2 * b * x[0] + c) ** 2 + (x[1] - b * x[0] ** 2 + c * x[0] - r) * (-2 * b)) - s * (
                1 - t) * np.cos(x[0])
    hess[0, 1] = hess[1, 0] = 2 * a * (-2 * b * x[0] + c)
    hess[1, 1] = 2 * a
    return hess


# Initial guess
x0_sphere = np.array([1.0, 1.0])
x0_branin = np.array([1.0, 1.0])

# Optimize Sphere function
opt_x_sphere, cost_sphere = levenberg_marquardt(sphere_function, sphere_grad, sphere_hess, x0_sphere)
print(f"Optimized parameters for Sphere function: {opt_x_sphere}")
print(f"Cost for Sphere function: {cost_sphere}")

# Optimize Branin function
opt_x_branin, cost_branin = levenberg_marquardt(branin_function, branin_grad, branin_hess, x0_branin)
print(f"Optimized parameters for Branin function: {opt_x_branin}")
print(f"Cost for Branin function: {cost_branin}")

# Plotting the functions
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Sphere function plot
Z_sphere = X ** 2 + Y ** 2
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
