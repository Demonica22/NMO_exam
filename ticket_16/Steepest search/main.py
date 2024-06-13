from steepestsearch import *
import numpy as np
from steepestDraw import *



def main():
    print("Sphere function:")
    x0 = np.array([[-5.0], [5.0]])
    tol = 1e-3
    [xmin, f, neval, coords] = sdsearch(fSphere, dfSphere, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)
    draw(coords, len(coords), "h", fSphere)


    print("Sum Of Different Powers function:")
    x0 = np.array([[-1.0], [1.0]])
    tol = 1e-7
    [xmin, f, neval, coords] = sdsearch(fSumOfDifferentPowersFunction, dfSumOfDifferentPowersFunction, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    draw(coords, len(coords), "r", fSumOfDifferentPowersFunction)


if __name__ == '__main__':
    main()
