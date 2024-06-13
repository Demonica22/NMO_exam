from newton import *
import numpy as np
from newtonDraw import *


def main():
    print("Himmelblau function:")
    x0 = np.array([[-2.0], [-2.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = nsearch(fSphere, dfSphere, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)
    draw(coords, len(coords), "h", fSphere)

    print("Rosenbrock function:")
    x0 = np.array([[-1.0], [-1.0]])
    tol = 1e-9
    [xmin, f, neval, coords] = nsearch(fBranin, dfBranin, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    draw(coords, len(coords), "r", fBranin)


if __name__ == '__main__':
    main()
