'''
Задача:
Максимизировать Z = 2x1 + x2 + 3x3 + 2x4
при условиях:
    x1 + 2x2 + x3 <= 11
    x1 + x3 + x4 = 8
    x2 + x4 >= 3
    xj >= 0
'''

import numpy as np
from dataclasses import dataclass
import re
from typing import List, Optional, Dict

@dataclass
class LP:
    sense: str
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    signs: List[str]
    var_names: List[str]


def parse_lp(text: str) -> LP:
    """Парсит текст задачи в структуру LP"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    sense = lines[0].lower()
    c = np.array(list(map(float, re.findall(r"[-+]?\d*\.?\d+", lines[1]))))
    A, b, signs = [], [], []
    for ln in lines[2:]:
        parts = ln.split()
        if "<=" in parts:
            sign = "<="
            idx = parts.index("<=")
        elif ">=" in parts:
            sign = ">="
            idx = parts.index(">=")
        else:
            sign = "="
            idx = parts.index("=")
        A.append(list(map(float, parts[:idx])))
        b.append(float(parts[idx + 1]))
        signs.append(sign)
    var_names = [f"x{i+1}" for i in range(len(c))]
    return LP(sense, c, np.array(A), np.array(b), signs, var_names)


# ------------------------- Решение через SciPy -------------------------

from scipy.optimize import linprog

def solve_lp(lp: LP):
    """Решает задачу LP с помощью линейного программирования"""
    # Преобразуем задачу к виду минимизации
    c = -lp.c if lp.sense == "max" else lp.c

    # Разделим неравенства по типу
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i, sign in enumerate(lp.signs):
        if sign == "<=":
            A_ub.append(lp.A[i])
            b_ub.append(lp.b[i])
        elif sign == ">=":
            A_ub.append(-lp.A[i])
            b_ub.append(-lp.b[i])
        else:
            A_eq.append(lp.A[i])
            b_eq.append(lp.b[i])

    res = linprog(
        c=c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=[(0, None)] * len(lp.c),
        method="highs"
    )

    if res.success:
        print("Оптимальное решение найдено:")
        for i, x in enumerate(res.x):
            print(f"  x{i+1} = {x:.4f}")
        print(f"  Z* = {lp.c @ res.x:.4f}")
    else:
        print("Решение не найдено:", res.message)


# -------

if __name__ == "__main__":
    with open("variant9.txt", encoding="utf-8") as f:
        text = f.read()

    lp = parse_lp(text)
    solve_lp(lp)
