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

    print("Bukin N. 6 function:")
    x0 = np.array([[4], [0]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fBukin, dfBukin, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fBukin, "bukin")



if __name__ == '__main__':
    main()
