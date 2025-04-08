import numpy as np
from scipy.optimize import linprog


def defuzzify(triangular_matrix):
    """
    Застосовує метод центру ваги для трикутних нечітких чисел.
    """
    crisp_matrix = np.array([[sum(triple) / 3 for triple in row] for row in triangular_matrix])
    return crisp_matrix


def solve_game(matrix):
    """
    Розв'язок матричної гри за допомогою лінійного програмування.
    """
    m, n = matrix.shape
    c = [-1] * n  # Максимізуємо виграш
    A_ub = -matrix.T  # Перетворюємо на задачу мінімізації
    b_ub = [-1] * m
    bounds = [(0, None) for _ in range(n)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x / sum(res.x)  # Оптимальні стратегії
    else:
        return None


# Вхідна нечітка платіжна матриця
A = [
    [(3, 5, 7), (2, 4, 6)],
    [(1, 3, 5), (4, 6, 8)]
]

# Виконання дефазифікації
A_crisp = defuzzify(A)

# Аналіз матричної гри
optimal_strategy = solve_game(A_crisp)

# Виведення результату
print("Чітка платіжна матриця після дефазифікації:")
print(A_crisp)

if optimal_strategy is not None:
    print("Оптимальна стратегія гравця A:", optimal_strategy)
else:
    print("Не вдалося знайти оптимальну стратегію.")
