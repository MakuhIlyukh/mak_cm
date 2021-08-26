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


def implicit(te, N, M, T, sigma=0.5):
    '''Неявный метод'''
    if sigma == 0:
        raise ValueError('Invalid sigma value!')

    u = np.empty((N + 1, M + 1))
    x = np.linspace(0, 1, N + 1)
    t = np.linspace(0, T, M + 1)
    h = 1 / N
    tau = T / M

    def get_system(k):
        '''to СЛАУ'''
        A = np.empty(N + 1)
        B = np.empty(N + 1)
        C = np.empty(N + 1)
        D = np.empty(N + 1)
        A[0] = 0
        B[0] = te.alpha1(t[k]) + te.alpha2(t[k]) / h
        C[0] = -te.alpha2(t[k]) / h
        D[0] = te.alpha(t[k])
        B[N] = te.beta1(t[k]) + te.beta2(t[k]) / h
        A[N] = -te.beta2(t[k]) / h
        C[N] = 0
        D[N] = te.beta(t[k])
        for i in range(1, N):
            coef = sigma * te.a(x[i], t[k]) / h**2
            A[i] = coef
            B[i] = -2*coef - 1 / tau
            C[i] = coef
            L = (te.a(x[i], t[k - 1])
                 * (u[i + 1, k - 1] - 2*u[i, k - 1] + u[i - 1, k - 1])
                 / h**2)
            # t с чертой как вычислять?
            D[i] = -1*u[i, k - 1] / tau - (1 - sigma)*L - te.f(x[i], t[k])
        return A, B, C, D

    def solve_system(A, B, C, D):
        '''Решает систему методом прогонки'''
        s = np.zeros(N + 1)
        t2 = np.zeros(N + 1)
        y = np.zeros(N + 1)
        s[0] = - C[0] / B[0]
        t2[0] = D[0] / B[0]
        for i in range(1, N + 1):
            s[i] = -C[i] / (A[i]*s[i - 1] + B[i])
            t2[i] = (D[i] - A[i]*t2[i - 1]) / (A[i]*s[i - 1] + B[i])
        y[N] = t2[N]
        for i in range(N - 1, -1, -1):
            y[i] = s[i] * y[i + 1] + t2[i]
        return y

    # step 1
    for i in range(N + 1):
        u[i, 0] = te.phi(x[i])    
    # step 2
    for k in range(1, M + 1):
        A, B, C, D = get_system(k)
        u[:, k] = solve_system(A, B, C, D)
    return u


def get_good_M(te, N, T):
    '''Возращает устойчивое M по заданному N'''
    h = 1 / N
    tau = h**2 / (2 * te.a(0, 0)) 
    return int(T / tau) + 1


if __name__ == '__main__':
    np.random.seed(44)
    # Variant 6
    N = 50
    # M = N*N*4*3*8
    T = 8
    te = TEquation(
        a=lambda x, t: 3,
        f=lambda x, t: 8*t - 18,
        phi=lambda x: 3*x*x + 5,
        alpha1=lambda t: 1,
        alpha2=lambda t: 0,
        alpha=lambda t: 4*t**2 + 5,
        beta1=lambda t: 1,
        beta2=lambda t: 0,
        beta=lambda t: 4*t**2 + 8
    )
    M = get_good_M(te, N, T)

    u1 = explicit(te, N, M, T)
    #u2 = implicit(te, N, M, T, sigma=1)

    x = np.linspace(0, 1, N + 1)
    t = np.linspace(0, T, M + 1)
    u3 = np.empty((N + 1, M + 1))
    for i in range(N + 1):
        for j in range(M + 1):
            u3[i, j] = 3*x[i]**2 + 4*t[j]**2 + 5