'''Задание 6'''


import numpy as np
from scipy.linalg import hilbert
from sympy import Matrix as sp_mat

from task2 import print_table


def r(A, i):
    '''Считает радиус'''
    return np.sum([np.abs(A[i, j]) for j in range(A.shape[0]) if j != i])


def gergoshin_theorem_check(A, eig_vals):
    '''Проверяет выполняется ли теорема'''
    n = A.shape[0]
    l_r = np.array([r(A, i) for i in range(n)])
    diag = np.diagonal(A)
    for e_val in eig_vals:
        abs_diffs = np.abs(diag - e_val)
        if not np.any(abs_diffs <= l_r):
            return False
    return True


def Jacobi(A, eps, policy='max', n_iter=np.inf):
    '''Метод Якоби для нахождения собственных чисел

    Parameters
    ----------
    policy : str
        Стратегия выбора опорного элемента. 
        policy='max' для выбора максимального недиагонального.
        policy='cyclic' для выбора циклическим образом.
    '''
    def t(A, i, j):
        '''Матрица плоского вращения'''
        if i >= j:
            raise ValueError('i >= j!')
        x = -2*A[i, j]
        y = A[i, i] - A[j, j]
        if y == 0:
            cos_phi = sin_phi = 1 / np.sqrt(2)
        else:
            root = np.sqrt(x**2 + y**2)
            cos_phi = np.sqrt((1 + np.abs(y) / root) / 2)
            sin_phi = np.sign(x*y)*np.abs(x) / (2*cos_phi*root)
        T = np.eye(A.shape[0])
        T[i, i] = cos_phi
        T[i, j] = -sin_phi
        T[j, i] = sin_phi
        T[j, j] = cos_phi
        return T

    def max_policy(A):
        '''Стратегия максимального abs элемента'''
        n = A.shape[0]
        m_i = 0
        m_j = 1
        for i in range(n):
            for j in range(i + 1, n):
                if np.abs(A[i, j]) > np.abs(A[m_i, m_j]):
                    m_i = i
                    m_j = j
        return m_i, m_j
    
    def cycling_policy(A):
        '''Циклический опорный элемент'''
        # cur_ij Инициализируется при выборе стратегии(код ниже)
        nonlocal cur_ij
        n = A.shape[0]
        if cur_ij[0] == n - 2:
            cur_ij = (0, 1)
        elif cur_ij[1] == n - 1:
            cur_ij = (cur_ij[0] + 1, cur_ij[0] + 2)
        else:
            cur_ij = (cur_ij[0], cur_ij[1] + 1)
        return cur_ij

    def check_r_sizes(A, eps):
        '''Проверяет не уменьшились ли r до допустимого размера'''
        return any(r(A, i) > eps for i in range(A.shape[0]))

    # Выбираем стратегию
    if policy == 'max':
        policy = max_policy
    elif policy == 'cyclic':
        cur_ij = (0, 1)
        policy = cycling_policy
    else:
        raise ValueError('Policy must be "max" or "cyclic"')

    # Алгоритм Якоби
    iteration = 0
    r_more_than_eps = True
    while iteration < n_iter and check_r_sizes(A, eps):
        pos = policy(A)
        T = t(A, *pos)
        A = T @ A @ T.T
        iteration += 1
    return np.sort(np.diagonal(A)), iteration


def get_accurate_eigvals(A):
    '''Возвращает точные собственные значения используя sympy'''
    M = sp_mat(A)
    res = list()
    for k, v in M.eigenvals().items():
        res.extend([k.n()]*v)
    return np.sort(res)


def test(A, epsilons):
    '''Тестирование методов нахождения собственных чисел'''
    print(A)
    print()
    table = list()
    ac_values = get_accurate_eigvals(A)
    for eps in epsilons:
        eig_m, i_m = Jacobi(A, eps, 'max')
        eig_c, i_c = Jacobi(A, eps, 'cyclic') 
        err_m = np.max(np.abs(eig_m - ac_values))
        err_c = np.max(np.abs(eig_c - ac_values))
        table.append([
            f'{eps:.1e}',
            f'{err_m:.1e}, {i_m}, {gergoshin_theorem_check(A, eig_m)}',
            f'{err_c:.1e}, {i_c}, {gergoshin_theorem_check(A, eig_c)}'
        ])
    print_table(table, ['eps', 'max_method', 'cyclic_method'])
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