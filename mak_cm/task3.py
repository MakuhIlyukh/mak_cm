'''Задача 3'''


import numpy as np
from scipy.linalg.special_matrices import hilbert
import pandas as pd

from task1 import cond_s


def t(A, i, j):
    '''Матрица T для разложения вращением'''
    x = A[i, i]
    y = A[i, j]
    r = np.sqrt(x*x + y*y)
    cos_phi = x/r
    sin_phi = -y/r
    res = np.eye(A.shape[0], dtype=np.float64)
    res[i, i] = cos_phi
    res[i, j] = -sin_phi
    res[j, j] = cos_phi
    res[j, i] = sin_phi
    return res


def rotating_decomposition(A):
    '''Метод разложения вращением'''
    n = A.shape[0]
    R = A.copy()
    Q = np.eye(n)
    for i in range(n - 1):
        for j in range(i + 1, n):
            T = t(A, i, j)
            R = T @ R
            Q = Q @ T.T
    return Q, R


def solve_rotating(A, b):
    '''Метод вращения для решения СЛАУ'''
    Q, R = rotating_decomposition(A)
    y = np.linalg.solve(Q, b)
    return np.linalg.solve(R, y)


# --------------------------------------------------------------------
# ДАЛЕЕ КОД СКОПИПАЩЕН ИЗ ЗАДАЧИ 2 И АДАПТИРОВАН ПОД QR 
# --------------------------------------------------------------------


def comp_values(A, x, alpha):
    '''Вычисляет спектральные числа обусловленности
     и норму разности погрешности(max(abs))'''
    b = A.dot(x)
    A2 = A + alpha * np.eye(A.shape[0]) # A + alpha*E
    x_app = solve_rotating(A2, b) # приблизительное решение
    A_cond_s = cond_s(A)
    A2_cond_s = cond_s(A2)
    error_norm = np.linalg.norm(x - x_app, ord=np.inf) # norm = max(abs(x-x0))
    return A_cond_s, A2_cond_s, error_norm


def print_table(table, columns, precision=3):
    '''Вывод таблицы'''
    df = pd.DataFrame(table, columns=columns)
    pd.set_option('precision', precision)
    print(df)


def iterate_alpha_for_test(A, x):
    '''Перебирает alpha, возвращает лучшее'''
    table = list() # Таблица результатов для разных alpha
    alphas = list(map(lambda i: 10**i, range(-12, 0)))
    for alpha in alphas:
        A_cond_s, A2_cond_s, error_norm = comp_values(A, x, alpha)
        table.append((alpha, A_cond_s, A2_cond_s, error_norm))
    print_table(table, ['alpha', 'A cond_s', '(A + alpha*E) cond_s', '||x-x0||'])
    print()
    # Лучшее alpha, для которого обусловленность меньше 10^4 и норма наименьшая
    return alphas[np.argmin([row[3] if row[2] < 1e+4 else np.inf
                             for row in table])]
     

def test_alpha(A, alpha):
    '''Выводит результаты для случайного вектора
    Норма - max(abs())
    '''
    print('Лучшее alpha:', alpha)
    x0 = np.random.randn(A.shape[1]) # Случайный вектор
    b = A.dot(x0)
    alphas = [0, 0.1*alpha, alpha, 10*alpha]
    errors = list()
    for alp in alphas:
        x = solve_rotating(A + alp*np.eye(A.shape[0]), b)
        errors.append(np.linalg.norm(x-x0, ord=np.inf)) # Норма - max(abs())
    print('Нормы погрешности: ')
    print_table([errors], columns=['0*alpha', '0.1*alpha', 'alpha', '10*alpha'])


def get_matrices_for_test():
    '''Матрицы для тестов'''
    return [
        hilbert(4),
        hilbert(5),
        hilbert(7)
    ]


if __name__ == '__main__':
    # Точность вывода
    np.set_printoptions(precision=3)
    # seed для датчика случайных чисел
    np.random.seed(15)
    # Для каждой матрицы:
    for A in get_matrices_for_test():
        print(A)
        print()
        best_alpha = iterate_alpha_for_test(A, np.ones(A.shape[1]))
        test_alpha(A, best_alpha)
        print('\n-----------------------------------------------------\n\n')