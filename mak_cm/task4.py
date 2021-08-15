'''Задание 4

Метод релаксаций сходится не при всех матрицах :(
'''


from abc import abstractmethod

import numpy as np
from scipy import linalg
from scipy.linalg import hilbert
from scipy.linalg.misc import norm

from task2 import print_table


# вид нормы: max(abs()) 
VECTOR_NORM_TYPE = np.inf


class Solver:
    '''Базовый класс для реализации шага в итерационных методах.
    Функция iter_solve будет вызывать метод step наследника
    класса Solver.
    '''
    @abstractmethod
    def step(self, A, b, x, i):
        pass


class StationarySolver(Solver): # Наследование от Solver
    '''Класс для реализации шага простого стационарного итерационного
    метода.
    '''
    def __init__(self, B, C):
        self.B = B
        self.C = C
    
    def step(self, A, b, x, i):
        return self.B @ x + self.C
    
    @staticmethod
    def generate_BC(A, b):
        '''Возращает матрицу B и вектор C для метода итераций
        https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B8%D1%82%D0%B5%D1%80%D0%B0%D1%86%D0%B8%D0%B8
        B и С взяты из статьи на википедии(ссылка указана выше)

        ВНИМАНИЕ: B и C подходят не для каждого СЛАУ!
        (пример: Гильбертова-4)
        '''
        n = A.shape[0]
        B = np.zeros((n, n))
        C = np.zeros(n)
        for i in range(n):
            C[i] = b[i] / A[i, i]
            for j in range(n):
                B[i, j] = (-A[i, j] / A[i, i]) if i != j else 0
        return B, C


class ZeydelSolver(Solver):
    '''Класс для реализации шага метода зейделя'''
    def step(self, A, b, x, i):
        n = x.shape[0]
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] += b[i] / A[i, i] # третье слагаемое
            for j in range(n):
                if j <= i - 1: # Первое слагаемое
                    x_new[i] -= x_new[j] * A[i, j] / A[i, i]
                elif j >= i + 1: # Второе слагаемое
                    x_new[i] -= x[j] * A[i, j] / A[i, i]
        return x_new


class RelaxationSolver(Solver):
    '''Класс для реализации метода релаксации
    
    По неизвестной причине сходится не при всех Матрицах.
    Пример: np.array([[10, 4, -1], [1, 50, 1], [1, 4, 35]])
            np.array([-0.46820879, -0.82282485, -0.0653801 ])
    '''
    def step(self, A, b, x, i):
        n = A.shape[0]
        x = x.copy() # Чтобы не менять исходный x
        mask = np.full(n, True) # неиспользованные невязки
        for j in range(n):
            # невязки
            deltas = A @ x - b # FIXME: растет очень быстро(что я делаю не так?)
            # индекс максимальной невязки
            k = np.nanargmax(np.abs(np.where(mask, deltas, np.nan)))
            # WARNING: возможно стоит заменить на np.abs(A[k, j]) > SMALL_CONST
            if A[k, j] != 0:
                x[j] = (b[k] - A[k] @ x + A[k, j]*x[j]) / A[k, j] 
            mask[k] = False
        return x


def iter_solve(A, b, start_x, solver, eps=1e-6, n_iter=np.inf):
    '''Функция для применения итерационных алгоритмов'''
    x = start_x.copy()
    i = 0 # номер итерации
    diff_norm = eps + 1 # В эту переменную будет записана норма разности
    while i < n_iter and diff_norm > eps:
        x_new = solver.step(A, b, x, i) # шаг
        diff_norm = np.linalg.norm((x - x_new), ord=VECTOR_NORM_TYPE)
        x = x_new
        i += 1
    return x, i


def matrices_1st_test():
    '''Возращает матрицы для первой части теста'''
    return [
        hilbert(2),
        np.array([[-402.94, 200.02],
                  [1200.12, -600.96]])
    ]


def matrices_2nd_test():
    '''Возращает матрицы для второй части теста'''
    n = 1000
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = sum([abs(A[i, j]) for j in range(n)])
    return [
        A
    ]


def test(A, relaxation=True):
    '''Первая часть теста(маленькие матрицы)'''
    epsilons = [10**k for k in range(-1, -9, -1)]
    print('A:')
    print(A)
    print()
    x0 = np.random.randn(A.shape[0])
    b = A @ x0
    table = list()
    for eps in epsilons:
        s_solver = StationarySolver(*StationarySolver.generate_BC(A, b))
        z_solver = ZeydelSolver()
        r_solver = RelaxationSolver()
        table.append([
            eps,
            iter_solve(A, b, s_solver.C, s_solver, eps, np.inf)[1],
            iter_solve(A, b, np.zeros_like(b), z_solver, eps, np.inf)[1],
            iter_solve(A, b, np.zeros_like(b), r_solver, eps, np.inf)[1] if relaxation else np.nan
        ])
    print_table(table, ['eps', 'stationary', 'zeydel', 'relaxation'])
    print('-------------------------------------------------')
    print()
    print()


def test1():
    for i, A in enumerate(matrices_1st_test()):
        test(A, i == 0)


def test2():
    for i, A in enumerate(matrices_2nd_test()):
        test(A, False)


if __name__ == '__main__':
    # Фиксируем random-seed
    np.random.seed(777)
    test1()
    test2()


    # A = np.array([[10, 4, -1], [1, 50, 1], [1, 4, 35]])

    # #
    # n = 1000
    # A = np.random.rand(n, n)
    # for i in range(n):
    #     A[i, i] = sum([abs(A[i, j]) for j in range(n)])
    # #

    # x0 = np.random.randn(A.shape[0])
    # b = A @ x0

    # B, C = StationarySolver.generate_BC(A, b)
    # solver = Iter()
    # x, i = iter_solve(A, b, np.zeros_like(x0), solver, 10e-9, 300)

    # print(np.linalg.norm(x - x0, ord=VECTOR_NORM_TYPE))
    # print(x0)
    # print(x)