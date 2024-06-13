from polakribieresearch import *
import numpy as np
from PRdraw import *

def main():
    print("Sphere function:")
    x0 = np.array([[4], [0.0]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fSphere, dfSphere, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fSphere, "sphere")

    print("Matyas function:")
    x0 = np.array([[4], [0]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fMatyas, dfMatyas, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fMatyas, "matyas")



if __name__ == '__main__':
    main()
