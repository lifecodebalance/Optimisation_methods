import sys, re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

class LP:
    def __init__(self, sense: str, c: np.ndarray, A: np.ndarray, b: np.ndarray, signs: List[str]):
        self.sense = sense.lower()  # 'max' | 'min'
        self.c = c.astype(float)
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.signs = signs

def parse_lp_file(path: str) -> LP:
    txt = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Пустой входной файл.")
    sense = lines[0].lower()
    if sense not in ("max", "min"):
        raise ValueError("Первая строка должна быть MAX или MIN")
    if not lines[1].lower().startswith("c:"):
        raise ValueError("Вторая строка должна начинаться с 'c:'")

    c = np.array(list(map(float, re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lines[1]))))
    A, b, signs = [], [], []
    for ln in lines[2:]:
        parts = ln.split()
        if "<=" in parts: s = "<="; k = parts.index("<=")
        elif ">=" in parts: s = ">="; k = parts.index(">=")
        elif "="  in parts: s = "=";  k = parts.index("=")
        else:
            raise ValueError(f"Не найден знак (<=, >=, =) в строке: {ln}")
        A.append(list(map(float, parts[:k])))
        b.append(float(parts[k+1]))
        signs.append(s)
    return LP(sense, c, np.array(A, dtype=float), np.array(b, dtype=float), signs)

def build_phase_matrix(lp: LP):
    A = lp.A.copy()
    b = lp.b.copy()
    signs = lp.signs.copy()
    m, n = A.shape

    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1
            if signs[i] == "<=": signs[i] = ">="
            elif signs[i] == ">=": signs[i] = "<="

    A_phase = A.copy()
    slack_idx, surplus_idx, artificial_idx = [], [], []
    row_info = []  # по строкам: ("slack",col) или ("artificial",col)
    col = n

    for i, s in enumerate(signs):
        if s == "<=":
            v = np.zeros((m, 1)); v[i, 0] = 1.0
            A_phase = np.hstack([A_phase, v])
            slack_idx.append(col)
            row_info.append(("slack", col))
            col += 1
        elif s == ">=":
            v1 = np.zeros((m, 1)); v1[i, 0] = -1.0
            v2 = np.zeros((m, 1)); v2[i, 0] = 1.0
            A_phase = np.hstack([A_phase, v1, v2])
            surplus_idx.append(col)
            artificial_idx.append(col+1)
            row_info.append(("artificial", col+1))
            col += 2
        elif s == "=":
            v = np.zeros((m, 1)); v[i, 0] = 1.0
            A_phase = np.hstack([A_phase, v])
            artificial_idx.append(col)
            row_info.append(("artificial", col))
            col += 1
        else:
            raise ValueError("Неизвестный знак ограничения")

    return A_phase, b, signs, slack_idx, surplus_idx, artificial_idx, row_info, n

def build_tableau(A: np.ndarray, b: np.ndarray, c: np.ndarray, basis: List[int]) -> Dict:
    m, N = A.shape
    B_idx = np.array(basis, dtype=int)
    B = A[:, B_idx]
    Binv = np.linalg.inv(B)
    xB = Binv @ b
    pi = (c[B_idx] @ Binv)
    N_idx = np.array([j for j in range(N) if j not in B_idx], dtype=int)
    rN = c[N_idx] - pi @ A[:, N_idx]
    return {"A": A, "b": b, "c": c, "B_idx": B_idx, "N_idx": N_idx, "Binv": Binv, "xB": xB, "pi": pi, "rN": rN}

def choose_entering(rN: np.ndarray, N_idx: np.ndarray) -> Optional[int]:
    eps = 1e-12
    cand = [k for k in range(len(N_idx)) if rN[k] < -eps]
    if not cand:
        return None
    # Правило Бленда: минимальный глобальный индекс
    return min(cand, key=lambda k: int(N_idx[k]))

def simplex_iterations(A: np.ndarray, b: np.ndarray, c: np.ndarray, basis: List[int], max_iter=10000):
    m, N = A.shape
    B_idx = basis[:]
    for _ in range(max_iter):
        B = A[:, B_idx]
        try:
            Binv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return {"status": "singular", "basis": B_idx}
        xB = Binv @ b
        pi = (c[B_idx] @ Binv)
        N_idx = [j for j in range(N) if j not in B_idx]
        rN = c[N_idx] - pi @ A[:, N_idx]
        # Оптимально (для минимизации)
        if np.all(rN >= -1e-10):
            return {"status": "optimal", "basis": B_idx, "xB": xB}
        # Входящая
        kN = choose_entering(rN, np.array(N_idx))
        if kN is None:
            return {"status": "optimal", "basis": B_idx, "xB": xB}
        j_enter = N_idx[kN]
        d = Binv @ A[:, j_enter]
        if np.all(d <= 1e-12):
            return {"status": "unbounded", "basis": B_idx}
        ratios = [xB[i] / d[i] if d[i] > 1e-12 else np.inf for i in range(m)]
        rho = min(ratios)
        I = [i for i, val in enumerate(ratios) if abs(val - rho) <= 1e-12]
        # Выходящая (Бленд): по минимальному индексу базисной переменной
        i_leave = min(I, key=lambda i: B_idx[i])
        B_idx[i_leave] = j_enter
    return {"status": "iterlimit", "basis": B_idx}

# ------------------------- Две фазы -------------------------

def two_phase_solve(lp: LP):
    # Строим расширенную систему равенств
    A_phase, b, signs, slack_idx, surplus_idx, artificial_idx, row_info, n = build_phase_matrix(lp)
    m, N = A_phase.shape

    # Фаза I: минимизируем сумму искусственных
    c_phase = np.zeros(N)
    for j in artificial_idx:
        c_phase[j] = 1.0

    # Стартовый базис: slack для '<=', artificial для '>=' и '='
    basis = [idx for typ, idx in row_info]

    # Запускаем Фазу I
    res1 = simplex_iterations(A_phase, b, c_phase, basis)
    if res1["status"] != "optimal":
        raise RuntimeError("Фаза I не сошлась или неограниченная.")
    xB1, basis1 = res1["xB"], res1["basis"]
    x_full1 = np.zeros(N); x_full1[basis1] = xB1
    if float(c_phase @ x_full1) > 1e-8:
        raise RuntimeError("Задача недопустима (сумма искусственных > 0).")

    # УДАЛЯЕМ искусственные столбцы и чиним базис
    artificial_set = set(artificial_idx)
    keep_cols = [j for j in range(N) if j not in artificial_set]
    A2 = A_phase[:, keep_cols]
    # remap basis
    mapping = {old: new for new, old in enumerate(keep_cols)}
    new_basis = []
    rows_to_fix = []
    for i, bidx in enumerate(basis1):
        if bidx in mapping:
            new_basis.append(mapping[bidx])
        else:
            new_basis.append(None)
            rows_to_fix.append(i)

    # Пытаемся выпихнуть искусственные: найдём в строке ненулевой небазисный столбец
    for row in rows_to_fix:
        current = set(j for j in new_basis if j is not None)
        candidates = [j for j in range(A2.shape[1]) if abs(A2[row, j]) > 1e-10 and j not in current]
        if candidates:
            new_basis[row] = candidates[0]
        else:
            # редкая ситуация: строка линейно зависима — удалим её
            A2 = np.delete(A2, row, axis=0)
            b   = np.delete(b, row, axis=0)
            new_basis.pop(row)
            break

    # Фаза II: минимизируем -c (эквивалент max c)
    c2 = np.zeros(A2.shape[1])
    if lp.sense == "max":
        c2[:n] = -lp.c
    else:
        c2[:n] = lp.c

    res2 = simplex_iterations(A2, b, c2, new_basis)
    if res2["status"] != "optimal":
        raise RuntimeError("Фаза II не сошлась или неограниченная.")

    xB2, basis2 = res2["xB"], res2["basis"]
    x_full2 = np.zeros(A2.shape[1]); x_full2[basis2] = xB2
    x = x_full2[:n]
    Z = float(lp.c @ x)
    return x, Z

# ----------------------------- main -----------------------------

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "variant9.txt"
    lp = parse_lp_file(in_path)
    x_opt, z_opt = two_phase_solve(lp)

    print("Оптимальное решение (по исходным переменным):")
    for i, xi in enumerate(x_opt, 1):
        print(f"x{i} = {xi:.6g}")
    print(f"Z* = {z_opt:.6g}")

    # Проверка ограничений на исходной постановке
    A, b, signs = lp.A, lp.b, lp.signs
    vals = A @ x_opt
    print("\nПроверка ограничений:")
    for i in range(A.shape[0]):
        if signs[i] == "<=":
            ok = vals[i] <= b[i] + 1e-9; rel = "<="
        elif signs[i] == ">=":
            ok = vals[i] + 1e-9 >= b[i]; rel = ">="
        else:
            ok = abs(vals[i] - b[i]) <= 1e-9; rel = "="
        status = "OK" if ok else "НАРУШЕНО"
        print(f"  {i+1}) {vals[i]:.6g} {rel} {b[i]:.6g}  -> {status}")

if __name__ == "__main__":
    main()
