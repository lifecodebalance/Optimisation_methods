import time
from math import sin, cos, pi, sqrt, exp

import numpy as np
import matplotlib.pyplot as plt

# Настройки задачи (их можно править под нужную функцию и отрезок)

# Целевая функция задаётся строкой. Доступны sin, cos, pi, sqrt, exp, np.
FUNCTION_STRING = "x**2 - 10*cos(2*pi*x) + 10"   # функция Растригина (1D)
LEFT_BOUND = -5.00                               # левая граница отрезка
RIGHT_BOUND = 5.00                               # правая граница отрезка
EPSILON = 0.01                                   # требуемая точность по x

# Параметры алгоритма
RELIABILITY = 2.0        # параметр надёжности метода Стронгина (r)
MAX_TRIALS = 10_000      # максимальное число итераций/испытаний


# Обёртка над функцией: eval по строке FUNCTION_STRING
def target_function(x: float) -> float:
    """
    Вычисляем значение функции f(x), заданной строкой FUNCTION_STRING.
    Разрешённые имена: x, np, sin, cos, pi, sqrt, exp.
    """
    env = {
        "x": x,
        "np": np,
        "sin": sin,
        "cos": cos,
        "pi": pi,
        "sqrt": sqrt,
        "exp": exp,
    }
    # Ограничиваем окружение, чтобы в eval не было ничего лишнего
    return eval(FUNCTION_STRING, {"__builtins__": {}}, env)

# Метод Стронгина (глобальный поиск минимума)
def strongin_search(func, left, right, eps, r=2.0, max_evals=10_000):
    """
    Одномерный метод Стронгина для глобального поиска минимума
    липшицевой функции на отрезке [left, right].

    Возвращает:
        x_min       – аргумент минимума
        f_min       – значение функции в минимуме
        iterations  – число итераций основного цикла
        elapsed     – затраченное время (сек)
        sample_x    – список всех испытанных точек
        sample_f    – значения функции в этих точках
    """

    start_time = time.time()

    # Начинаем с концов отрезка
    sample_x = [left, right]
    sample_f = [func(left), func(right)]
    iterations = 0

    while iterations < max_evals:
        iterations += 1

        # 1. Упорядочиваем точки по x
        order = np.argsort(sample_x)
        sample_x = [sample_x[i] for i in order]
        sample_f = [sample_f[i] for i in order]

        n = len(sample_x)

        # 2. Оцениваем константу Липшица (M) по текущим точкам
        M = 0.0
        for i in range(1, n):
            dx = sample_x[i] - sample_x[i - 1]
            if dx <= 0:
                continue
            slope = abs(sample_f[i] - sample_f[i - 1]) / dx
            if slope > M:
                M = slope

        # Если все значения одинаковые, не даём M обнулиться
        if M < 1e-9:
            M = 1.0

        # Параметр m = r * M
        m = r * M

        # 3. Считаем характеристику R(i) для каждого интервала
        characteristics = []  # (R_i, индекс правой точки интервала)
        for i in range(1, n):
            dx = sample_x[i] - sample_x[i - 1]
            df = sample_f[i] - sample_f[i - 1]
            R_i = (
                m * dx
                + (df ** 2) / (m * dx)
                - 2.0 * (sample_f[i] + sample_f[i - 1])
            )
            characteristics.append((R_i, i))

        # 4. Выбираем интервал с максимальной характеристикой
        characteristics.sort(key=lambda pair: pair[0], reverse=True)
        best_R, best_index = characteristics[0]

        # Интервал [x_{k-1}, x_k]
        left_x = sample_x[best_index - 1]
        right_x = sample_x[best_index]
        left_f = sample_f[best_index - 1]
        right_f = sample_f[best_index]
        dx = right_x - left_x
        df = right_f - left_f

        # 5. Новая точка по формуле Стронгина
        x_new = 0.5 * (left_x + right_x) - df / (2.0 * m)

        # Чтобы численно не перейти за интервал
        if x_new < left_x:
            x_new = left_x
        elif x_new > right_x:
            x_new = right_x

        sample_x.append(x_new)
        sample_f.append(func(x_new))

        # 6. Критерий остановки по длине максимального интервала
        if dx < eps:
            break

    elapsed = time.time() - start_time
    best_idx = int(np.argmin(sample_f))
    x_min = sample_x[best_idx]
    f_min = sample_f[best_idx]

    return x_min, f_min, iterations, elapsed, sample_x, sample_f

# Построение вспомогательной ломаной (нижняя оценка функции)
def build_support_line(points, values, r=2.0):
    order = np.argsort(points)
    points = [points[i] for i in order]
    values = [values[i] for i in order]

    # Оценка M (константы Липшица)
    M = 0.0
    for i in range(1, len(points)):
        dx = points[i] - points[i - 1]
        if dx <= 0:
            continue
        slope = abs(values[i] - values[i - 1]) / dx
        if slope > M:
            M = slope
    if M < 1e-9:
        M = 1.0
    m = r * M

    aux_x, aux_y = [], []

    for i in range(1, len(points)):
        x1, x2 = points[i - 1], points[i]
        f1, f2 = values[i - 1], values[i]

        # Теоретически оптимальная точка вспомогательной параболы
        x_star = 0.5 * (x1 + x2) - (f2 - f1) / (2.0 * m)

        # Проекция на отрезок [x1, x2]
        if x_star < x1:
            x_star = x1
        elif x_star > x2:
            x_star = x2

        # Нижняя оценка в точке x_star
        y_star = min(
            f1 - m * abs(x_star - x1),
            f2 - m * abs(x_star - x2),
        )

        aux_x.extend([x1, x_star, x2])
        aux_y.extend([f1, y_star, f2])

    return aux_x, aux_y

# Точка входа
def main():
    print("Глобальная минимизация одномерной функции (метод Стронгина)")
    print(f"f(x) = {FUNCTION_STRING}")
    print(f"Отрезок: [{LEFT_BOUND}, {RIGHT_BOUND}]")
    print(f"Точность по x: eps = {EPSILON}\n")

    x_min, f_min, iterations, elapsed, xs, fs = strongin_search(
        target_function,
        LEFT_BOUND,
        RIGHT_BOUND,
        EPSILON,
        r=RELIABILITY,
        max_evals=MAX_TRIALS,
    )

    print("Результаты поиска:")
    print(f"  x* ≈ {x_min:.8f}")
    print(f"  f(x*) ≈ {f_min:.8f}")
    print(f"  Итераций: {iterations}")
    print(f"  Испытаний (точек): {len(xs)}")
    print(f"  Время работы: {elapsed:.4f} с")

    # Плотная сетка для красивого графика функции
    dense_x = np.linspace(LEFT_BOUND, RIGHT_BOUND, 1000)
    dense_y = [target_function(x) for x in dense_x]

    # Вспомогательная ломаная
    aux_x, aux_y = build_support_line(xs, fs, r=RELIABILITY)

    # Визуализация matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dense_x, dense_y, label="f(x)")
    ax.scatter(xs, fs, s=25, alpha=0.8, label=f"Точки испытаний ({len(xs)})")
    ax.plot(aux_x, aux_y, "--", linewidth=1.5,
            label="Вспомогательная ломаная (нижняя оценка)")
    ax.scatter([x_min], [f_min],
               marker="*", s=180,
               label=f"Минимум: x≈{x_min:.4f}, f≈{f_min:.4f}")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Поиск глобального минимума методом Стронгина")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()