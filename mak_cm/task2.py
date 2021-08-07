'''Задача 2'''


import numpy as np
from numpy.linalg.linalg import cond
from scipy.linalg.special_matrices import hilbert

from task1 import cond_s


def sqrt_decomposition(A):
    '''Метод квадратного корня для разложения матрицы'''
    L = np.zeros_like(A)
    n = L.shape[0]
    for k in range(n):
        slice1 = L[k, 0:k] # l[k, 0], l[k, 1], ..., l[k, k-1]
        L[k, k] = np.sqrt(A[k, k] - (slice1 @ slice1))
        for i in range(k + 1, n):
            slice2 = L[i, 0:k] # l[i, 0], l[i, 1], ..., l[i, k-1]
            L[i, k] = (A[i, k] - (slice1 @ slice2)) / L[k, k]
    return L


def LL_solve(A, b):
    '''Решает слау, используя L @ L.T разложение матрицы:
    A @ x = L @ L.T @ x = b
    y = L.T @ x
    L @ y = b'''
    L = sqrt_decomposition(A)
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def comp_values(A, x, alpha):
    '''Вычисляет спектральные числа обусловленности
     и норму разности погрешности(max(abs))'''
    b = A.dot(x)
    A2 = A + alpha * np.eye(A.shape[0])
    x_app = LL_solve(A2, b)
    A_cond_s = cond_s(A)
    A2_cond_s = cond_s(A2)
    return A_cond_s, A2_cond_s, np.linalg.norm(x-x_app, ord=np.inf)


def iterate_alpha_for_test(A, x):
    '''Перебирает alpha, возвращает лучшее'''
    for alpha in map(lambda i: 10**i, range(-12, 0)):
        A_cond_s, A2_cond_s, error_norm = comp_values(A, x, alpha)
        print(f'{A_cond_s:.3f}\t{A2_cond_s:.3f}\t{error_norm:.3f}')
     


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.random.seed(15)
    iterate_alpha_for_test(hilbert(4), np.ones(4))