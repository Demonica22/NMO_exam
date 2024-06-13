from secantdraw import *
from secantsearch import *

def main():
    interval = [-2, 10]
    tol = 1e-9
    [xmin, f, neval, coords] = bsearch(f1, df1, interval, tol)
    print("f1:")
    print("x =", xmin)
    print("f(x) =", f)
    print("k =", neval)
    print("\n")
    bisectionsearch2slides(f1, interval, coords)

    [xmin, f, neval, coords] = bsearch(f2, df2, interval, tol)
    print("f2:")
    print("x =", xmin)
    print("f(x) =", f)
    print("k =", neval)
    bisectionsearch2slides(f2, interval, coords)

if __name__ == '__main__':
    main()
