import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def f1(x):
    term1 = -2 * np.sin(np.sqrt(np.abs(x / 2 + 10)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - 10)))
    return term1 + term2

def df1(x):
    h = 1e-9
    return (f1(x + h) - f1(x - h)) / (2 * h)

def f2(x):
    term1 = x**2
    term2 = -10 * np.cos(0.5 * np.pi * x)
    term3 = -110
    return term1 + term2 + term3

def df2(x):
    h = 1e-9
    return (f2(x + h) - f2(x - h)) / (2 * h)

def bsearch(interval, tol, df, f):
    x3 = interval[0]
    x2 = interval[1]
    x1 = 6
    k = 0
    neval = 1
    coords = [x1]

    while k < 1000:
        g_x = df(x1)
        g_x1 = df(x2)
        g_x2 = df(x3)
        neval += 3

        h1 = x1 - x2
        h2 = x2 - x3
        d0 = (g_x - g_x1) / h1
        d1 = (g_x1 - g_x2) / h2
        a = (d0 - d1) / (h1 + h2)
        b = a * h2 + d1
        c = g_x

        discriminant = np.sqrt(b ** 2 - 4 * a * c)
        den_plus = b + discriminant
        den_minus = b - discriminant

        if abs(den_minus) > abs(den_plus):
            denominator = den_minus
        else:
            denominator = den_plus

        x_new = x1 - 2 * c / denominator
        coords.append(x_new)

        if abs(x_new - x1) <= tol:
            fmin = f(x_new)
            neval += 1
            return [x_new, fmin, neval, coords]

        x3, x2, x1 = x2, x1, x_new
        k += 1
    fmin = f(x_new)
    answer_ = [x_new, fmin, neval, coords]
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
    [xmin, f, neval, coords] = bsearch(interval, tol, df1, f1)
    print("f1:")
    print("x =", xmin)
    print("f(x) =", f)
    print("k =", neval)
    print("\n")
    bisectionsearch2slides(f1, interval, coords)

    [xmin, f, neval, coords] = bsearch(interval, tol, df2, f2)
    print("f2:")
    print("x =", xmin)
    print("f(x) =", f)
    print("k =", neval)
    bisectionsearch2slides(f2, interval, coords)

if __name__ == '__main__':
    main()