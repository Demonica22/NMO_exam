from secantdraw import *
from secantsearch import *

def main():
    print("First function:")
    interval = [-2, 5]
    tol = 1e-6
    [xmin, f, neval, coords] = ssearch(f1, df1, interval, tol)
    print([xmin, f, neval])
    secantsearchsecants(f1, df1, coords, interval)

    print("Second function:")
    interval = [-2, 5]
    tol = 1e-6
    [xmin, f, neval, coords] = ssearch(f2, df2, interval, tol)
    print([xmin, f, neval])
    secantsearchsecants(f2, df2, coords, interval)


if __name__ == '__main__':
    main()
