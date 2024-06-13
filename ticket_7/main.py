from polakribieresearch import *
import numpy as np
from PRdraw import *

def main():
    print("First function:")
    x0 = np.array([[1.0], [0.0]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fFirstFunc, dfFirstFunc, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fFirstFunc, "h")

    print("Seconf function:")
    x0 = np.array([[-2], [0]])
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fSecondFunc, dfSecondFunc, x0, tol)
    print(xmin, f, neval)
    draw(coords,  len(coords), fSecondFunc, "r")



if __name__ == '__main__':
    main()
