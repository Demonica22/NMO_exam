# Файл main.py
from gradientsearch import *
import numpy as np
from gradientDraw import *

def main():
    x0 = np.array([-2, 10])
    tol = 1e-3
    [xmin, f, neval, coords] = grsearch(fSphere, dfSphere, x0, tol)
    print("Sphere function:")
    print(xmin, f, neval)
    draw(fSphere, coords)  # Используем функцию fSphere для отрисовки

    x1 = np.array([0.5, 0.5])
    tol = 1e-9
    [xmin, f, neval, coords] = grsearch(fSixHumpCamel, dfSixHumpCamel, x1, tol)
    print("SixHumpCamel function:")
    print(xmin, f, neval)
    draw(fSixHumpCamel, coords)  # Используем функцию fSixHumpCamel для отрисовки

if __name__ == '__main__':
    main()
