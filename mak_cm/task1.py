'''Задача 1.'''


import numpy as np
from scipy.linalg import hilbert


def u_pert(x, eps):
    '''Делает случайное возмущение для каждого элемента x, используя
    равномерное распределение с параметрами [-eps; +eps]'''
    if eps <= 0:
        raise ValueError('eps должно быть положительным')
    return np.random.uniform(-eps, eps, x.shape) + x


def s_pert(x, eps):
    '''Добавляет случайно к каждому элементу x либо eps, либо -eps'''
    return np.random.choice([-eps, eps], size=x.shape) + x


def cond_s(A):
    '''Спектральное число обусловленности
    Норма: largest singular value'''
    # Норма: largest singular value
    return np.linalg.cond(A, p=2)


def cond_v(A):
    '''Объемное число обусловленности.
    В презентациях есть опечатка?'''
    return np.sqrt(np.product([row @ row for row in A])) / np.linalg.det(A)


def cond_a(A):
    '''Угловое число обусловленности(использует евклидову норму)'''
    C = np.linalg.inv(A)
    return np.max([np.abs(np.linalg.norm(row))*np.abs(np.linalg.norm(column))
                   for row, column in zip(A, C.T)])


def matrices_for_test():
    '''Матрицы для тестов: Гильбертовы и из методички Пакулиной'''
    return [
        hilbert(5),
        hilbert(7),
        np.array([[-403.15, 200.95],
                  [1205.70, -604.10 ]]),
        np.array([[-402.94, 200.02],
                  [1200.12, -600.96]])
    ]


def exact_solutions_for_test(l_A):
    '''Возвращает точное решение для тестов: случайный вектор и единичный
    l_A - список матриц'''
    return [[np.random.randn(A.shape[1]), np.full(A.shape[1], 1)]
            for A in l_A]


def test(A, x):
    '''Тестирование матрицы и точного решения'''
    b = A.dot(x.T)
    print(f'Матрица A:\n{A}')
    print()

    print(f'Точное решение x:\n {x}')
    print()

    print(f'cond_s: {cond_s(A)}')
    print(f'cond_v: {cond_v(A)}')
    print(f'cond_a: {cond_a(A)}')
    print()

    for eps in [1e-2, 1e-5, 1e-8]:
        print(f'eps = {eps}')
        A_hat = s_pert(A, eps)
        b_hat = s_pert(b, eps)
        x_hat = np.linalg.solve(A_hat, b_hat)
        print(f'Приближенное решение:{x_hat}')
        print(f'Модуль невязки:{np.abs(x-x_hat)}')
        print()


if __name__ == '__main__':
    # Краткий вывод матриц
    np.set_printoptions(precision=4)
    # Инициализация генератора случайных чисел
    np.random.seed(123)
    # Матрицы для тестов
    l_A = matrices_for_test()
    # Точные решения
    l_x = exact_solutions_for_test(l_A)
    # Тестируем каждую пару матрица-решение
    for A, vecs_for_A in zip(l_A, l_x):
        for x in vecs_for_A:
            test(A, x)
            print('-------------------------------')
