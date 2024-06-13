***Билет 8***

[Конспект по методу дихотомии](https://open.etu.ru/assets/courseware/v1/5771ff9e51deca175f6cf9f2be7aa0b9/asset-v1:kafedra-cad+opt-methods+spring_2024+type@asset+block/конспект2_1.pdf)

***Метод  дихотомии. Запрограммировать, показать работу на функциях  f(x) = -2*sin(sqrt(abs(x/2 + 10))) - x.*sin(sqrt(abs(x - 10))),
f(x) = x.^2 -  10*cos(0.5*pi*x) – 110,
Поисковый интервал в обоих случаях [-2,10].***
[КОД](https://open.etu.ru/courses/course-v1:kafedra-cad+opt-methods+spring_2024/courseware/0648cf091a7240d8a93f52d3d9a9eeb7/bcee3bfc437c4a94990af341801866b9/3?activate_block_id=block-v1%3Akafedra-cad%2Bopt-methods%2Bspring_2024%2Btype%40vertical%2Bblock%4096abdcdbfb6b45dc90082f79a0a5671b)

```python

def bsearch(interval,tol):
# searches for minimum using bisection method
# arguments: bisectionsearch(f,df,interval,tol)
# f - an objective function
# df -  an objective function derivative
# interval = [a, b] - search interval
#tol - tolerance for both range and function value
# output: [xmin, fmin, neval, coords]
# xmin - value of x in fmin
# fmin - minimul value of f
# neval - number of function evaluations
# coords - array of x values found during optimization
    
    neval = 0
    coords = []

    while abs((interval[1] - interval[0]) > tol) and (abs(df(interval[0])) > tol):
        x = (interval[0] + interval[1]) / 2
        if df(x) > 0:
            interval[1] = x
        else:
            interval[0] = x

        coords.append(x)
        neval += 1

    xmin = coords[-1]
    fmin = df(coords[-1])

    answer_ = [xmin, fmin, neval, coords]
    return answer_
```


***Запрограммировать  метод Хестенеса-Штифеля. Показать его работу на функциях Sphere и Matyas, размерность d = 2***
[КОД](https://open.etu.ru/courses/course-v1:kafedra-cad+opt-methods+spring_2024/courseware/36e24e85aa75401a9ac7002730b64bb0/d9acc0c322074580a3a52d45be116b2c/2?activate_block_id=block-v1%3Akafedra-cad%2Bopt-methods%2Bspring_2024%2Btype%40vertical%2Bblock%40cbf3f93d6d6145a7bb10d97b35917cd0)

```python

# Метод Хестенеса-Штифеля
def prsearch(f, df, x0, tol):
    # PRSEARCH searches for minimum using Polak-Ribiere method
    # 	answer_ = sdsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    c1 = tol
    c2 = 0.1

    coords = [x0]

    kmax = 1000
    amax = 3

    g0 = -df(x0)
    p0 = np.copy(g0)

    while (norm(g0) >= tol) and (len(coords) < kmax):
        # Ищем оптимальный размер шага
        ak = wolfesearch(f, df, x0, p0, amax, c1, c2)

        # метод спуска с использованием длины шага
        x0 = x0 + ak*p0

        gk = -df(x0)

        g_diff_transposed = gk.transpose()

        # Обновляем значение направления по методу Хестенеса-Штифеля
        denom = np.dot(np.transpose(p0), (gk - g0))
        if denom != 0:
            b = np.dot(g_diff_transposed, (gk - g0))/ denom
        else:
            b = 0
    # b = - (np.dot(np.transpose(g_k), (g_k - g_prev))) / (np.dot(np.transpose(p), (g_k - g_prev)))

        p0 = gk + b * p0

        g0 = -df(x0)
        coords.append(x0)

    answer_ = [x0, f(x0), len(coords), coords]
    return answer_
```
