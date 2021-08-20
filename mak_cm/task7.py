'''Задание 7'''


import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np


NORM = np.inf


class BoundaryProblem:
    '''Класс для хранения краевой задачи'''
    def __init__(self, k, p, q, f, ap1, ap2, ap, a, bt1, bt2, bt, b):
        """
        Уравнение
        ---------
        ku'' + pu' + qu = f

        Краевые условия
        ---------------
        ap1*u + ap2*u' = ap \\
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


def get_delta(u, u_prev, r, p):
    '''Вычисляет delta'''
    delta = np.empty(u.shape[0])
    for j in range(u_prev.shape[0]):
            delta[r*j] = (u[r*j] - u_prev[j]) / (r**p - 1)
    for j in range(1, u.shape[0], r):
        delta[j] = (delta[j - 1] + delta[j + 1]) / 2
    return delta


def grid_method(bp: BoundaryProblem, start_n, eps=1e-6, r=2, p=1, n_iter=np.inf):
    '''Сеточный метод'''
    n = start_n # резмер сетки
    i = 0 # число итераций
    u = solve_bp_system(bp, n)
    u_l = list() 
    u_l.append(u)
    delta_l = [[0.0]*(start_n + 1)]
    while i < n_iter:
        n *= r
        u_prev = u
        u = solve_bp_system(bp, n)
        delta = get_delta(u, u_prev, r, p)
        delta_l.append(delta)
        u_l.append(u)
        if np.linalg.norm(delta, ord=NORM) < eps:
            u += delta
            break
        i += 1
    return u_l, delta_l, n, i


def get_problems():
    '''Возращает задачи для тестов'''
    return [
        # Вариант 3 Пакулина
        BoundaryProblem(
            k=lambda x: -1 / (x - 3),
            p=lambda x: (1 + x / 2),
            q=lambda x: np.exp(x / 2),
            f=lambda x: 2 - x,
            ap1=1,
            ap2=0,
            ap=0,
            a=-1,
            bt1=1,
            bt2=0,
            bt=0,
            b=1
        ),
        # Вариант 5 Пакулина
        BoundaryProblem(
            k=lambda x: -1 / (x + 3),
            p=lambda x: -x,
            q=lambda x: np.log(2 + x),
            f=lambda x: 1 - x / 2,
            ap1=0,
            ap2=1,
            ap=0,
            a=-1,
            bt1=1/2,
            bt2=1,
            bt=0,
            b=1
        )
    ]


def get_lims():
    '''Границы графиков'''
    return [
        ((-1, 1), (-3, 1)),
        ((-1, 1), (0, 6))
    ]


def plot_results(a, b, u_l, delta_l, xlim, ylim, name):
    '''Анимированные результаты'''
    fig, ax = plt.subplots()
    m = len(u_l)

    def animate(i):
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        u = u_l[i]
        d = delta_l[i]
        x = np.linspace(a, b, u.shape[0])
        line = ax.scatter(x, u, s=0.5)
        ax.set_title(f'i={i}, delta={np.linalg.norm(d, ord=NORM):.2e}')
    
    animation = ani.FuncAnimation(fig, animate, frames=m, interval=200, repeat=False)
    animation.save(name, writer='html', fps=1)


if __name__ == '__main__':
    np.random.seed(12)
    bp_l = get_problems()
    lims = get_lims()
    for j in range(len(bp_l)):
        u_l, delta_l, n, i = grid_method(bp_l[j], start_n=4, n_iter=15)
        plot_results(
            bp_l[j].a,
            bp_l[j].b,
            u_l,
            delta_l,
            lims[j][0],
            lims[j][1],
            f'{j}_animation.html'
        )
