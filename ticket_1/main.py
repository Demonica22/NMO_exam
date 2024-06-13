from trustregionsearch import *
import numpy as np
from draw import *


def main():
    print("Sphere function:")
    x0 = np.array([[2.0], [1.0]])
    tol = 1e-3
    [xmin, f, neval, coords, rad] = trustreg(fH, dfH, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)
    draw(coords, len(coords), "h", rad, fH)

    print("Ackley function:")
    x0 = np.array([[-2], [0]])
    tol = 1e-3
    [xmin, f, neval, coords, rad] = trustreg(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    draw(coords, len(coords), "r", rad, fR)


if __name__ == '__main__':
    main()
