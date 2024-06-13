from damped import *
import numpy as np
# from draw import *



def main():
    print("Sphere:")
    x0 = np.array([[10.0], [10.0]])
    tol = 1e-9
    [xmin, f, neval, coords] = newton(fH, dfH, x0, tol)  #функция Химмельблау
    print(xmin, f, neval)
    # draw(coords,  len(coords), fH, "h")

    print("Styblinski-Tang:")
    x0 = np.array([[-2.0], [-2.0]])
    tol = 1e-9
    [xmin, f, neval, coords] = newton(fR, dfR, x0, tol)  # функция Розенброка
    print(xmin, f, neval)
    # draw(coords,  len(coords), fR, "r")


if __name__ == '__main__':
    main()