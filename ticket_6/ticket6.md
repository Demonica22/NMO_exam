***Билет 6***\
**Квадратичная форма. Квадратичная функция. Критерии минимума (максимума) квадратичной формы.**

**Квадратичная форма** – функция на векторном пространстве, задаваемая однородным многочленом второй степени от координат вектора\
![ticket_5_1.png](../ticket_5/ticket_5_1.png)

В зависимости от знака принимаемых квадратичной формой значений, различают по-
ложительно определенную, положительно полуопределенную, отрицательно определенную, отрицательно полуопределенную и знакопеременную квадратичные формы.

![ticket_5_2.png](../ticket_5/ticket_5_2.png)

Для матрицы A **собственными числами и собственными векторами** называются такие числа λi и вектора vi, что

![ticket_5_3.png](../ticket_5/ticket_5_3.png)

Смысл собственного вектора – умножение его на матрицу A дает коллинеарный вектор, умноженный на некоторое скалярное значение.

Квадратичная функция – функция вида

![ticket_5_6.png](../ticket_5/ticket_5_6.png)
где c – скаляр, b – вектор, A – матрица.

**Критерий минимума (максимума) дважды дифференцируемой функции векторного аргумента.**

![ticket_5_5.png](../ticket_5/ticket_5_5.png)

[Конспект](https://open.etu.ru/assets/courseware/v1/98ff340fdb30155841a7af9315c3b889/asset-v1:kafedra-cad+opt-methods+spring_2024+type@asset+block/конспект1_5.pdf)



***Запрограммировать  метод  Дая-Юана. Показать его работу на функциях Sphere и Bukin N. 6, размерность d = 2***
[КОД](https://www.open.etu.ru/courses/course-v1:kafedra-cad+opt-methods+spring_2024/courseware/36e24e85aa75401a9ac7002730b64bb0/216c21b8e9bc4aaf82f40ca52a72d9f3/1?activate_block_id=block-v1%3Akafedra-cad%2Bopt-methods%2Bspring_2024%2Btype%40vertical%2Bblock%403efa8bfb7030457faf40de0804b09543)

```python

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

        g_diff_transposed = (gk).transpose()

        # Обновляем значение направления по методу Дая-Юана
        denom = np.dot(np.transpose(p0), gk - g0)
        if denom != 0:
            b = - np.dot(g_diff_transposed, gk)/ denom
        else:
            b = 0

        p0 = gk + b * p0

        g0 = -df(x0)
        coords.append(x0)

    answer_ = [x0, f(x0), len(coords), coords]
    return answer_
```
