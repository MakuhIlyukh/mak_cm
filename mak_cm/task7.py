'''Задание 7'''


import numpy as np


class BoundaryProblem:
    '''Класс для хранения краевой задачи'''
    def __init__(self, k, p, q, f, ap1, ap2, ap, a, bt1, bt2, bt, b):
        """
        Уравнение
        ---------
        ku'' + pu' + qu = f

        Краевые условия
        ---------------
        ap1*u + ap2*u' = ap
        bt*u + bt*u' = bt
        """
        self.k = k
        self.p = p
        self.q = q
        self.f = f
        self.ap1 = ap1
        self.ap2 = ap2
        self.ap = ap
        self.a = a
        self.bt1 = bt1
        self.bt2 = bt2
        self.bt = bt
        self.b = b


def get_system(bp: BoundaryProblem, n: int):
    '''ПРИВЕДЕНИЕ К СЛАУ С ТРЕХДИАГОНАЛЬНОЙ МАТРИЦЕЙ
    A_i*u_(i-1) + B_i*u_i + C_i*u_(i+1) = D_i
    ''' 
    h = (bp.b - bp.a) / n
    x = np.linspace(bp.a, bp.b, n + 1)
    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    C = np.zeros(n + 1)
    D = np.zeros(n + 1)
    A[n] = -bp.bt2
    B[0] = h*bp.ap1 - bp.ap2
    B[n] = h*bp.bt1 + bp.bt2
    C[0] = bp.ap2
    C[n] = 0
    D[0] = h * bp.a
    D[n] = h * bp.b
    for i in range(1, n):
        A[i] = 2*bp.k(x[i]) - h*bp.p(x[i])
        B[i] = -4*bp.k(x[i]) + 2*h*h*bp.q(x[i])
        C[i] = 2*bp.k(x[i]) + h*bp.p(x[i])
        D[i] = 2*h*h*bp.f(x[i])
    return A, B, C, D


def solve_bp_system(bp: BoundaryProblem, n: int):
    '''Решает систему методом прогонки'''
    A, B, C, D = get_system(bp, n)
    s = np.zeros(n + 1)
    t = np.zeros(n + 1)
    u = np.zeros(n + 1)
    s[0] = - C[0] / B[0]
    t[0] = D[0] / B[0]
    for i in range(1, n + 1):
        s[i] = -C[i] / (A[i]*s[i - 1] + B[i])
        t[i] = (D[i] - A[i]*t[i - 1]) / (A[i]*s[i - 1] + B[i])
    u[n] = t[n]
    for i in range(n - 1, -1, -1):
        u[i] = s[i] * u[i + 1] + t[i]
    return u


def grid_method(bp: BoundaryProblem, start_n, r=2, n_iter=np.inf):
    '''Сеточный метод'''
    n = start_n
    i = 0
    u = solve_bp_system(bp, n)
    while i < n_iter:
        n *= r
        u_prev = u
        u = solve_bp_system(bp, n)
        




if __name__ == '__main__':
    np.random.seed(12)