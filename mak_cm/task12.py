'''Задание 12'''


from functools import partial

import numpy as np
import matplotlib.pyplot as plt


class OneCallException(Exception):
    pass


class Distances:
    '''Виды расстояний'''
    @staticmethod
    def eucl_dist(X1, X2, ord=2):
        '''Евклидово растояние в n-мерном пространстве'''
        return np.linalg.norm(X1 - X2, ord=ord, axis=1)

    @staticmethod
    def max_dist(X1, X2):
        '''max(abs()) distance'''
        return np.linalg.norm(X1 - X2, ord=np.inf, axis=1)

    @staticmethod
    def rbf_dist(X1, X2, gamma):
        '''dist = exp(-gamma*||x - y||**2)'''
        return np.exp(-gamma*np.linalg.norm(X1 - X2, ord=2, axis=1)**2)


class StartGenerators:
    @staticmethod
    def random_init(k, X):
        means = X.mean(axis=0)
        stds = X.std(axis=1)
        return np.random.randn(k, X.shape[1])*stds + means

    @staticmethod
    def random_minmax_init(k, X):
        '''Выбирает случайно либо максимум, либо минимум
         по координате'''
        maxs = X.max(axis=0)
        mins = X.min(axis=0)
        res = np.empty((k, X.shape[1]))
        for i in range(k):
            res[i] = np.where(np.random.randint(0, 2, X.shape[1]),
                              mins, maxs)
        return res
    

class KMeans:
    def __init__(self, k, dist_fun=Distances.eucl_dist):
        self.k = k
        self.dist_fun = dist_fun
        self.is_used = False

    def fit(self, X, start_generator=StartGenerators.random_init):
        if self.is_used:
            raise OneCallException('Создайте новый объект для использования')
        self.centers = start_generator(self.k, X)
        old_labels = self.predict(X)
        is_changed = True
        while is_changed:
            new_labels = self.predict(X)
            is_changed = (old_labels == new_labels).all()
            old_labels = new_labels

    def predict(self, X):
        self.labels = np.empty(X.shape[0], dtype=np.int32)
        self.distances = np.empty((X.shape[0], self.k))
        for i in range(self.k):
            self.distances[:, i] = self.dist_fun(X, self.centers[i])
        self.labels = np.argmax(self.distances, axis=1)
        return self.labels


def inertia(est, X):
    est.predict(X)
    closest_dists = est.distances[np.arange(est.distances.shape[0]),
                                  est.labels]
    return closest_dists.sum()


def plot_kmeans(X,
                l_k,
                dist_fun=Distances.eucl_dist,
                start_generator=StartGenerators.random_init):
    l_inertias = list()
    for k in l_k:
        est = KMeans(k, dist_fun=dist_fun)
        est.fit(X, start_generator=start_generator)
        l_inertias.append(inertia(est, X))
    
    plt.plot(l_k, l_inertias)


if __name__ == '__main__':
    np.random.seed(123)
    X = np.random.randn(10, 4)
    Y = np.random.randn(10, 4)
    