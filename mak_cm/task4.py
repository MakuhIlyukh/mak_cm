'''Задание 4'''


from abc import abstractmethod

import numpy as np
from scipy.linalg import hilbert


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

        ВНИМАНИЕ: B и C подходят не для каждого СЛАУ! (пример: Гильбертова-4)
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
                

if __name__ == '__main__':
    # Фиксируем random-seed
    np.random.seed(777)

    A = np.array([[10, 4, -1], [1, 50, 1], [1, 4, 35]])
    x0 = np.random.randn(A.shape[0])
    b = A @ x0

    B, C = StationarySolver.generate_BC(A, b)
    solver = ZeydelSolver()
    x, i = iter_solve(A, b, np.zeros(A.shape[0]), solver, 10e-9, 300)

    print(np.linalg.norm(x - x0, ord=VECTOR_NORM_TYPE))
    print(x0)
    print(x)