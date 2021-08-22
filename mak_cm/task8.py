'''Задание 8'''


import numpy as np
from scipy.integrate import simps
from scipy.special import eval_jacobi, jacobi
import matplotlib.pyplot as plt

from task7 import BoundaryProblem, solve_bp_system


class RitzFormProblem:
    '''Ly = −(p(x)y′)′ + r(x)y = f(x)
    
    alpha1*y(-1) - alpha2*y'(-1) = 0
    beta1*y(1) + beta2*y'(1) = 0
    '''
    def __init__(self, p, r, f, alpha1, alpha2, beta1, beta2):
        self.p = p
        self.r = r
        self.f = f
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2


class Basis:
    '''Базис функций'''
    def __init__(self, w, diff_w):
        if len(w) != len(diff_w):
            raise ValueError('HMM!')
        self.w = w
        self.diff_w = diff_w


def integrate(fun, A, B, m):
    '''Метод составных квадратур СИМПСОН'''
    h = (B - A)/(2*m)
    s = fun(A) + fun(B)
    for i in range(1, 2*m, 2):
        s += 4*fun(A + i*h)
    for i in range(2, 2*m-1, 2):
        s += 2*fun(A + i*h)
    s = s * (B - A) / (6*m)
    return s


def integration_wrapper(int_fun):
    '''Декоратор для встроенных функций интегрирования'''
    def wrapper(int_expr, a, b, n):
        x = np.linspace(a, b, n)
        y = np.vectorize(int_expr) (x)
        return int_fun(y, x)
    return wrapper


class RitzSolver():
    '''Алгоритм Ритца'''
    def __init__(self, rfp, basis, integrator='scipy', n=100):
        self.rfp = rfp
        self.basis = basis
        self.set_integrator(integrator)
        self.n = n

    def set_integrator(self, integrator='scipy'):
        if integrator == 'numpy':
            self.integrator = integration_wrapper(np.trapz)
        elif integrator == 'scipy':
            self.integrator = integration_wrapper(simps)
        elif integrator == 'self-made':
            self.integrator = integrate
        else:
            raise ValueError('Unknown type of integrator.')

    def set_n(self, n):
        '''Параметр для составной квадратурной формулы'''
        self.n = n
    
    def set_basis(self, basis):
        self.basis = basis

    def bilineal_form(self, i, j):
        '''Вычисляет билинейную между двумя векторами
        базиса. Использует численные методы.'''
        rfp, basis = self.rfp, self.basis

        def int_expr(x):
            return (rfp.p(x) * basis.diff_w[i](x) * basis.diff_w[j](x)
                    + rfp.r(x) * basis.w[i](x) * basis.w[j](x))
        
        int_res = self.integrator(int_expr, -1, 1, self.n)
        if rfp.alpha1 == 0 or rfp.alpha2 == 0:
            Q_l = 0
            Q_r = 0
        else:
            Q_l = (rfp.alpha1 / rfp.alpha2
                   * rfp.p(-1) * basis.w[i](-1) * basis.w[j](-1))
            Q_r = (rfp.beta1 / rfp.beta2 
                   * rfp.p(1) * basis.w[i](1) * basis.w[j](1))
        return int_res + Q_l + Q_r
    
    def scalar_mult(self, f, g):
        '''Скалярное произведение в L2(-1, 1)'''
        int_expr = lambda x: f(x)*g(x)
        int_res = self.integrator(int_expr, -1, 1, self.n)
        return int_res
    
    def solve(self):
        '''Решает систему'''
        b_len = len(self.basis.w) # Размер базиса
        A = np.empty((b_len, b_len))
        b = np.empty(b_len)
        for i in range(b_len):
            b[i] = self.scalar_mult(self.rfp.f, self.basis.w[i])
            for j in range(b_len):
                A[i, j] = self.bilineal_form(i, j)
        return np.linalg.solve(A, b)


def comp(x, basis, coefs):
    '''Вычисляет ответ для заданых базиса и коэффициентов'''
    s = 0
    for w, c in zip(basis.w, coefs):
        s += w(x)*c
    return s


def jacobi_polynom(n):
    '''Многочлен Якоби k = 1'''
    return lambda x: eval_jacobi(n, 1, 1, x)


def jacobi_dx(n):
    '''Производные'''
    coefs = jacobi(n, 1, 1)
    coefs_dx = np.polyder(coefs, 1)
    return lambda x: np.sum(coefs_dx * np.power(x, np.arange(n + 1)))


def get_jacobi_basis(n):
    '''Возращает базис'''
    return Basis([jacobi_polynom(i) for i in range(n)],
                 [jacobi_dx(i) for i in range(n)])


if __name__ == '__main__':
    np.random.seed(111)
    grid_n = 10**3
    basis_n = 30

    bp = BoundaryProblem(
        k=lambda x: - 1 / (x + 2),
        p=lambda x: 1 / (x + 2)**2,
        q=np.cos,
        f=lambda x: 1 + x,
        ap1=1,
        ap2=0,
        ap=0,
        bt1=1,
        bt2=0,
        bt=0,
        a=-1,
        b=1
    )
    u = solve_bp_system(bp, grid_n)

    u_len = len(u)
    x = np.linspace(-1, 1, u_len)

    rfp = RitzFormProblem(
        p=lambda x: 1 / (2 + x),
        r=np.cos,
        f=lambda x: 1 + x,
        alpha1=1,
        alpha2=0,
        beta1=1,
        beta2=0
    )
    basis = get_jacobi_basis(basis_n)
    solver = RitzSolver(rfp, basis)
    coefs = solver.solve()
    u2 = np.vectorize(lambda x: comp(x, basis, coefs)) (x)
    
    plt.plot(x, u)
    plt.plot(x, u2)
    plt.show()
