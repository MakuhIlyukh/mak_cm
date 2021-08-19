'''Задание 5'''


import numpy as np
from scipy.linalg import hilbert

from task2 import print_table


def power_solve(A, start_x, eps, n_iter=np.inf):
    '''Степенной метод'''
    x = start_x.copy() # Вектор
    e_val = -np.inf # собственное значение
    i = 0 # номер итерации
    flag = True # Условие выхода из цикла
    while i < n_iter and flag:
        # Обновляем переменные для старых значений
        x_prev = x
        e_val_prev = e_val
        # Обновляем переменные для новых значения
        x = A @ x
        e_val = x[0] / x_prev[0]
        # Условие выхода из цикла
        flag = np.abs(e_val - e_val_prev) > eps  
        i += 1
    return e_val, x, i # Собственное значение, вектор, число итераций  


def scalar_solve(A, start_x, start_y, eps, n_iter=np.inf):
    '''Скалярный метод'''
    x = start_x.copy() # Вектор
    y = start_y.copy() # Yet another vector
    e_val = -np.inf # собственное значение
    i = 0 # номер итерации
    flag = True # Условие выхода из цикла
    while i < n_iter and flag:
        # Обновляем переменные для старых значений
        x_prev = x
        e_val_prev = e_val
        # Обновляем переменные для новых значения
        x = A @ x
        y = A.T @ x
        e_val = (x @ y) / (x_prev @ y)
        # Условие выхода из цикла
        flag = np.abs(e_val - e_val_prev) > eps  
        i += 1
    return e_val, x, i # Собственное значение, вектор, число итераций 


# ---------------------------------------------------------------------------
# ДАЛЕЕ КОД СКОПИПАЩЕН ИЗ ЗАДАЧИ 6 И АДАПТИРОВАН ПОД ЗАДАЧУ 5
# (А ЭТОТ КОММЕНТАРИЙ СКОПИПАЩЕН ИЗ ЗАДАЧИ 3 И АДАПТИРОВАН ПОД ЗАДАЧУ 5 :) ) 
# ---------------------------------------------------------------------------


def accurate_maxabs_eigval(A):
    '''Возращает точное значение максимального по модулю
    собсвенного числа.'''
    return np.max(np.abs(np.linalg.eigvals(A)))


def test(A, epsilons):
    '''Тестирование методов нахождения максимальных собственных чисел'''
    print(A)
    print()
    table = list()
    ac_values = accurate_maxabs_eigval(A)
    for eps in epsilons:
        eig_p, eig_v_p, i_p = power_solve(
            A, np.random.randn(A.shape[0]), eps
        )
        eig_s, eig_v_s, i_s = scalar_solve(
            A, np.random.randn(A.shape[0]), np.random.randn(A.shape[0]), eps
        )
        err_p = np.max(np.abs(np.abs(eig_p) - ac_values))
        err_s = np.max(np.abs(np.abs(eig_s) - ac_values))
        table.append([
            f'{eps:.1e}',
            f'{err_p:.1e}, {i_p}',
            f'{err_s:.1e}, {i_s}'
        ])
    print_table(table, ['eps', 'power_method', 'scalar_method'])
    print('-----------------------------------------------------')
    print()


def matrices_for_test():
    '''Матрицы для тестов'''
    return [
        hilbert(10),
        np.array([[-0.81417, -0.01937, 0.41372],
                  [-0.01937, 0.54414, 0.00590],
                  [0.41372, 0.00590, -0.81445]]),
    ]


if __name__ == '__main__':
    np.random.seed(12)
    np.set_printoptions(precision=3)
    for A in matrices_for_test():
        test(A, [10**(-k) for k in range(10)])
