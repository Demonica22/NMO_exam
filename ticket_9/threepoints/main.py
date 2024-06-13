from bisectiondraw import bisectionsearch2slides
from threepointsearch import threepointsearch

def main():
    print("Find:")
    interval = [-2, 10]
    tol = 1e-10
    [xmin, f, neval, coords] = threepointsearch(interval,tol)
    print([xmin, f, neval])
    bisectionsearch2slides(interval, coords)

if __name__ == '__main__':
    main()