from polakribieresearch import *
import numpy as np
from PRdraw import *

def main():
    print("Sphere function:")
    x0 = np.array([[4.0], [0.0]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fSphere, dfSphere, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fSphere, "sphere")

    print("Holder Table function:")
    x0 = np.array([[8], [9]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fHolderTable, dfHolderTable, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fHolderTable, "holdertable")



if __name__ == '__main__':
    main()
