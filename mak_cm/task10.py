'''Задание 10'''


import numpy as np


class TEquation:
    '''Уравнение теплопроводности
    
    u_t = a(x, t)*u_xx + f(x, t)
    u(x, 0) = phi(x)
    alpha1(t)*u(0, t) - alpha2(t)*u(0, t)_x = alpha(t)
    beta1(t)*u(1, t) + beta2(t)*u(1, t)_x = beta(t) 
    '''
    def __init__(self,
                 a, f,
                 phi,
                 alpha1, alpha2, alpha,
                 beta1, beta2, beta):
        self.a = a
        self.f = f
        self.phi = phi
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta = beta


def explicit(te, N, M, T):
    '''Явный метод решения уравнения теплопроводности'''
    u = np.empty((N + 1, M + 1))
    x = np.linspace(0, 1, N + 1)
    t = np.linspace(0, T, M + 1)
    h = 1 / N
    tau = T / M
    
    # step 1
    for i in range(N + 1):
        u[i, 0] = te.phi(x[i])

    for k in range(1, M + 1):
        # step 2
        for i in range(1, N):
            L = (te.a(x[i], t[k - 1])
                 * (u[i + 1, k - 1] - 2*u[i, k - 1] + u[i - 1, k - 1])
                 / h**2) 
            u[i, k] = u[i, k - 1] + tau*(L + te.f(x[i], t[k - 1]))

        # step 3
        u[0, k] = (te.alpha(t[k])
                   + te.alpha2(t[k])*(4*u[1, k] - u[2, k]) / (2*h)
                   / (te.alpha1(t[k]) + te.alpha2(t[k]) * 3 / (2*h)))

        # step 4
        u[N, k] = (te.beta(t[k])
                   + te.beta2(t[k])*(4*u[N - 1, k] - u[N - 2, k]) / (2*h)
                   / (te.beta1(t[k]) + te.beta2(t[k]) * 3 / (2*h)))
    return u


def implicit(te, N, M, T):
