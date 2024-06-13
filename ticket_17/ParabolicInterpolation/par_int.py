import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def f1(x):
    term1 = -2 * np.sin(np.sqrt(np.abs(x / 2 + 10)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - 10)))
    return term1 + term2

def df1(x):
    h = 1e-3
    return (f1(x + h) - f1(x - h)) / (2 * h)

def f2(x):
    term1 = x**2
    term2 = -10 * np.cos(0.5 * np.pi * x)
    term3 = -110
    return term1 + term2 + term3

def df2(x):
    h = 1e-9
    return (f2(x + h) - f2(x - h)) / (2 * h)

def bsearch(f, df, interval, tol):
    a = interval[0]
    b = interval[1]
    neval = 0
    x_2 = a
    x_1 = (a + b) / 4
    x_0 = b
    coords = []

    while (np.abs(x_0 - x_2)) > tol:

        df_0 = df(x_0)
        df_1 = df(x_1)
        df_2 = df(x_2)

        x1 = ((df_1 * df_0) / ((df_2 - df_1) * (df_2 - df_0))) * x_2 + (
                    (df_2 * df_0) / ((df_1 - df_2) * (df_1 - df_0))) * x_1 + (
                         (df_2 * df_1) / ((df_0 - df_2) * (df_0 - df_1))) * x_0
        x_0, x_1, x_2 = x1, x_0, x_1
        neval += 1
        coords.append(x_0)
        if (np.abs(df_0)) <= tol:
            break
    xmin = x_0
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_

def bisectionsearch2slides(f, interval, coords):
    # make plot
    t1 = np.arange(interval[0], interval[1], 0.001)
    fig, ax = plt.subplots()
    plt.plot(t1, f(t1))
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig('plot.png')
    print("<img width=\"550px\" src=\"/resources/plot.png\">")

    nfigs = min([10, len(coords)])  # output 10 figures or less if number of points is less
    '''
    for i in range(nfigs):
        plt.plot(coords[i], f(coords[i]), 'ro')
        name = "plot" + str(i) + ".png"
        fig.savefig(name)
        ad = "<img width=\"550px\" src=\"/resources/" + name + "\">"
        print(ad)
    '''
    plt.plot(coords[-1], f(coords[-1]), 'bo')  # print last point blue
    fig.savefig('plotFin.png')
    print("<img width=\"550px\" src=\"/resources/plotFin.png\">")

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

    [xmin, f, neval, coords] = bsearch(f2, df2,interval, tol)
    print("f2:")
    print("x =", xmin)
    print("f(x) =", f)
    print("k =", neval)
    bisectionsearch2slides(f2, interval, coords)

if __name__ == '__main__':
    main()