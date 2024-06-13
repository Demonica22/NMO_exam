import numpy as np
from numpy.linalg import norm


# Define the Sphere functions
def sphere(X):
    x = X[0]
    y = X[1]
    return x**2 + y**2

def dsphere(X):
    x = X[0]
    y = X[1]
    return np.array([2*x, 2*y])


# Define the McCormick function
def mccormick(X):
    x, y = X
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


# Define the gradient of the McCormick function
def dmccormick(X):
    x, y = X
    df_dx = np.cos(x + y) + 2 * (x - y) - 1.5
    df_dy = np.cos(x + y) - 2 * (x - y) + 2.5
    return np.array([df_dx, df_dy])


def bbsearch(f, df, x0, tol):
    delta = 0.1
    x = np.array(x0)
    neval = 0
    kmax = 1000
    coords = [x]
    alpha = 0.01
    g = df(x)
    deltaX = np.inf
    k = 0

    while (norm(deltaX) >= tol) and (k < kmax):
        g_norm = norm(g)
        if g_norm == 0:
            break
        alpha_stab = delta / g_norm
        alpha = min(alpha, alpha_stab)
        x_next = x - alpha * g
        deltaX = x_next - x
        x = x_next
        neval += 1
        coords.append(x)

        g_next = df(x)
        delta_x = x - coords[-2]
        delta_g = g_next - g
        numerator = np.sum(delta_x * delta_x)
        denominator = np.sum(delta_x * delta_g)
        if denominator != 0:
            alpha_bb = numerator / denominator
        else:
            alpha_bb = alpha_stab
        g_next_norm = norm(g_next)
        if g_next_norm == 0:
            break
        alpha_stab = delta / g_next_norm
        alpha = min(alpha_bb, alpha_stab)

        g = g_next
        k += 1

    xmin = x
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_


# Initial guess and tolerance
x0_sphere = np.array([2.0, 2.0])
x0_mccormick = np.array([0.0, 0.0])
tol = 1e-6

# Apply the Barzilai-Borwein method to Sphere function
result_sphere = bbsearch(sphere, dsphere, x0_sphere, tol)
print("Sphere Function:")
print("Minimum point:", result_sphere[0])
print("Minimum value:", result_sphere[1])
print("Number of evaluations:", result_sphere[2])

# Apply the Barzilai-Borwein method to McCormick function
result_mccormick = bbsearch(mccormick, dmccormick, x0_mccormick, tol)
print("\nMcCormick Function:")
print("Minimum point:", result_mccormick[0])
print("Minimum value:", result_mccormick[1])
print("Number of evaluations:", result_mccormick[2])
