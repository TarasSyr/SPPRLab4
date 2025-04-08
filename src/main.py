import numpy as np
from scipy.optimize import linprog


def defuzzify(triangular_matrix):

    #метод центру ваги для трикутних нечітких чисел (V=(a+b+c)/3)
    crisp_matrix = np.array([[sum(triple) / 3 for triple in row] for row in triangular_matrix])
    return crisp_matrix


def solve_game(matrix):
    #додатковий розв'язок матричної гри за допомогою лінійного програмування для
    # аналізу платіжної матриці
    m, n = matrix.shape

    # Додаємо змінну v та формулюємо нову систему обмежень
    c = [0] * n + [-1]  # Мінімізуємо v
    A_ub = np.hstack((-matrix.T, np.ones((n, 1))))  # -p_i * A + v ≥ 0
    b_ub = np.zeros(n)
    A_eq = [np.append(np.ones(n), 0)]  # p1 + p2 = 1
    b_eq = [1]
    bounds = [(0, 1) for _ in range(n)] + [(None, None)]  # p_i ∈ [0,1], v - необмежене

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        probabilities = res.x[:-1] * 100
        return probabilities
    else:
        return None, res.message



A = [
    [(3, 5, 7), (2, 4, 6)],
    [(1, 3, 5), (4, 6, 8)]
]


A_crisp = defuzzify(A)

# Аналіз матричної гри
optimal_strategy = solve_game(A_crisp)

# Виведення результату
print("Чітка платіжна матриця після дефазифікації:")
print(A_crisp)

if isinstance(optimal_strategy, tuple):
    print("Не вдалося знайти оптимальну стратегію. Причина:", optimal_strategy[1])
else:
    print("Оптимальна стратегія гравця A (у відсотках):", optimal_strategy)
