'''Проверка метода релаксаций'''


import numpy as np
from scipy.linalg import hilbert
from tqdm import tqdm

from task4 import Solver, RelaxationSolver, iter_solve
from task1 import s_pert


def get_system():
    '''Возвращает Матрицу и решение'''
    A = np.array([[10, 4, -1], [1, 50, 1], [1, 4, 35]])
    x0 = np.array([-0.46820879, -0.82282485, -0.0653801 ])
    return A, x0


def get_start_x(A):
    '''Возращает случайное стартовое значение x
    для метода релаксаций'''
    return np.random.uniform(-1, 1, A.shape[0])


if __name__ == '__main__':
    np.random.seed(142)
    # Степень сходимости
    eps = 1e-6 
    # СЛАУ
    A, x0 = get_system()
    b = A @ x0

    success_history = list()
    # Тестирование метода релаксаций
    for pert_eps in (10**(-k) for k in range(10)): # Размер возмущений
        A_eps = s_pert(A, pert_eps)
        b_eps = s_pert(b, pert_eps)
        for j in range(10**3): # варьируем стартовые x
            start_x = get_start_x(A_eps)
            solver = RelaxationSolver()
            try:
                x, i = iter_solve(A_eps, b_eps, start_x, solver, eps)
                print('Удача! Норма разности: ', np.linalg.norm(x-x0, ord=np.inf))
                print('Start_x: ', start_x)
            except:
                pass
