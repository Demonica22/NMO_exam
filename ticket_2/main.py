from LBFGS import *
import numpy as np
from draw import *


def main():
    print("Sphere function:")
    x0 = np.array([[2.0], [1.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = LBFGS(fH, dfH, x0, tol)
    print(xmin, f, neval)
    draw(coords, len(coords),fH, "h")

    print("Levy function:")
    x0 = np.array([[-2], [0]])
    tol = 1e-3
    [xmin, f, neval, coords] = LBFGS(fR, dfR, x0, tol)
    print(xmin, f, neval)
    draw(coords, len(coords), fR, "r")


if __name__ == '__main__':
    main()
