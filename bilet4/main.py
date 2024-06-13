from DFPsearch import *
import numpy as np
from DFPdraw import *



def main():
    print("Sphere function:")
    x0 = np.array([[2], [1]])
    tol = 1e-3
    [xmin, f, neval, coords] = dfpsearch(fH, dfH, x0, tol)  #функция Химмельблау
    print(xmin, f, neval)
    draw(coords,  len(coords), fH, "h")

    print("CROSS-IN-TRAY function:")
    x0 = np.array([[-3], [-2]])
    tol = 1e-5
    [xmin, f, neval, coords] = dfpsearch(fR, dfR, x0, tol)  # функция Розенброка
    print(xmin, f, neval)
    draw(coords,  len(coords), fR, "r")


if __name__ == '__main__':
    main()
