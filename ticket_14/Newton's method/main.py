from newtondraw import *
from newton import *

def main():
    print("First function:")
    interval = [-2, 10] #for drawing
    tol = 1e-9
    [xmin, f, neval, coords] = nsearch(f1, df1, ddf1, tol, 1.3)
    print([xmin, f, neval])
    newtondrawfig(interval, coords)

    print("Second function:")
    interval = [-2, 10]  # for drawing
    tol = 1e-9
    [xmin, f, neval, coords] = nsearch(f2, df2, ddf2, tol, 0.001)
    print([xmin, f, neval])
    newtondrawfig(interval, coords)


if __name__ == '__main__':
    main()