import numpy as np
from numpy.linalg import norm

np.seterr(divide='ignore', invalid='ignore')

def sphere_function(X):
    x = X[0]
    y = X[1]
    return x**2 + y**2

def grad_sphere_function(X):
    x = X[0]
    y = X[1]
    return np.array([2*x, 2*y])

def three_hump_camel_function(X):
    x = X[0]
    y = X[1]
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2

def grad_three_hump_camel_function(X):
    x = X[0]
    y = X[1]
    grad = np.zeros_like(X)
    grad[0] = 4 * x - 4.2 * x**3 + x**5 + y
    grad[1] = x + 2 * y
    return grad

def nesterov_nemirovski(f, grad_f, x0, tol=1e-5):
    al = 0.05  # Initial step size
    eta = al / 10  # Step for updating x
    gamma = 0.75  # Inertia coefficient
    kmax = 1000  # Maximum number of iterations

    x = np.copy(x0)  # Current point
    y = np.copy(x0)  # Intermediate point for gradient computation
    coords = [x]  # List to store coordinates at each iteration

    neval = 0
    k = 0

    while (norm(grad_f(x)) >= tol) and (k < kmax):
        grad_y = grad_f(y)  # Compute gradient at current intermediate point y
        x_next = y - eta * grad_y  # Update x
        y_next = x_next + gamma * (x_next - x)  # Update y with inertia

        if norm(grad_f(x_next)) < tol:  # Check stopping condition
            break

        x = np.copy(x_next)
        y = np.copy(y_next)
        k += 1
        neval += 1
        coords.append(x)

    xmin = x
    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coords]
    return answer_

# Testing on Sphere function
x0_sphere = np.array([1.0, 1.0])
result_sphere = nesterov_nemirovski(sphere_function, grad_sphere_function, x0_sphere)
print("Sphere Function Optimization")
print("Minimum point:", result_sphere[0])
print("Function value at minimum point:", result_sphere[1])
print("Number of gradient evaluations:", result_sphere[2])

# Testing on Three-Hump Camel function
x0_camel = np.array([1.0, 1.0])
result_camel = nesterov_nemirovski(three_hump_camel_function, grad_three_hump_camel_function, x0_camel)
print("\nThree-Hump Camel Function Optimization")
print("Minimum point:", result_camel[0])
print("Function value at minimum point:", result_camel[1])
print("Number of gradient evaluations:", result_camel[2])
