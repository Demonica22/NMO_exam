from BFGS import *
import numpy as np
from draw import *


def main():
    print("Sphere function:")
    x0 = np.array([[2.0], [1.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = BFGS(fH, dfH, x0, tol)
    print(xmin, f, neval)
    draw(coords, len(coords),fH, "h")

    print("Eggholder function:")
    x0 = np.array([[500], [400]])
    tol = 1e-5
    [xmin, f, neval, coords] = BFGS(fR, dfR, x0, tol)
    print(xmin, f, neval)
    draw(coords, len(coords), fR, "r", 400,600,400,500)


if __name__ == '__main__':
    main()
