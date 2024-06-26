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

def bsearch(f, df, interval, tol):
    a = interval[0]
    b = interval[1]
    neval = 2
    x = a
    coords = [x]

    df_a = df(a)
    df_b = df(b)

    while (np.abs(b - a)) > tol:
        x = b - df_b * (b - a) / (df_b - df_a)
        df_x = df(x)
        if (df_x > 0):
            b = x
            xmin = b
            df_b = df_x
        else:
            a = x
            xmin = a
            df_a = df_x

        neval += 1
        coords.append(x)
        if (np.abs(df_x)) <= tol:
            break

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

    for i in range(nfigs):
        plt.plot(coords[i], f(coords[i]), 'ro')
        name = "plot" + str(i) + ".png"
        fig.savefig(name)
        ad = "<img width=\"550px\" src=\"/resources/" + name + "\">"
        print(ad)
    plt.plot(coords[-1], f(coords[-1]), 'bo')  # print last point blue
    fig.savefig('plotFin.png')
    print("<img width=\"550px\" src=\"/resources/plotFin.png\">")