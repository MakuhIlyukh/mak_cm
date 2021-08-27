'''Задание 12'''


from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


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
        stds = X.std(axis=0)
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
    
    @staticmethod
    def first_points_init(k, X):
        if X.shape[0] < k:
            raise ValueError('X.shape[0] < k')
        return X[:k].copy()
    

class KMeans:
    def __init__(self, k, dist_fun=Distances.eucl_dist):
        self.k = k
        self.dist_fun = dist_fun
        self.is_used = False

    def fit(self, X, start_generator=StartGenerators.random_init):
        if self.is_used:
            raise OneCallException('Создайте новый объект для использования')
        self.dim = X.shape[1]
        self.centers = start_generator(self.k, X)
        old_labels = self.predict(X)
        is_changed = True
        while is_changed:
            self.comp_new_centers(X)
            is_changed = (old_labels != self.labels).any()
            old_labels = self.labels

    def predict(self, X):
        self.labels = np.empty(X.shape[0], dtype=np.int32)
        self.distances = np.empty((X.shape[0], self.k))
        for i in range(self.k):
            self.distances[:, i] = self.dist_fun(X, self.centers[i])
        self.labels = np.argmin(self.distances, axis=1)
        return self.labels

    def comp_new_centers(self, X):
        self.predict(X)
        ns = np.zeros(self.k, dtype=np.int32)
        self.centers = np.zeros((self.k, self.dim))
        for i in range(X.shape[0]):
            l = self.labels[i]
            self.centers[l] += X[i]
            ns[l] += 1
        for i in range(self.k):
            if ns[i] > 0:
                self.centers[i] /= ns[i]
            else:
                self.centers[i] = X[np.random.randint(0, X.shape[0])]
        self.predict(X)


def inertia(est, X):
    est.predict(X)
    closest_dists = est.distances[np.arange(est.distances.shape[0]),
                                  est.labels]
    return closest_dists.sum()


def part_a(X,
           l_k,
           dist_fun=Distances.eucl_dist,
           start_generator=StartGenerators.random_init):
    np.random.seed(123)
    l_inertias = list()
    for k in tqdm(l_k):
        est = KMeans(k, dist_fun=dist_fun)
        est.fit(X, start_generator=start_generator)
        l_inertias.append(inertia(est, X))
    
    plt.plot(l_k, l_inertias)
    plt.show()


def load_data():
    df = pd.read_csv('data/mydata.csv')
    X = df.values
    return X


def contour_plot(minX, maxX, minY, maxY, nx, ny, est, ax):
    x = np.linspace(minX, maxX, nx)
    y = np.linspace(minY, maxY, ny)
    XX, YY = np.meshgrid(x, y)
    XY = np.array([XX.ravel(), YY.ravel()]).T
    Z = est.predict(XY)
    ax.contourf(XX, YY, Z.reshape(XX.shape), alpha=0.4)


def part_b():
    np.random.seed(111)
    X = load_data()
    fig, ax = plt.subplots(2, 2)
    
    k = 5
    dists = [Distances.eucl_dist, Distances.max_dist]
    inits = [StartGenerators.random_init, StartGenerators.random_minmax_init]
    for i in range(2):
        for j in range(2):
            est = KMeans(k, dist_fun=dists[i])
            est.fit(X, start_generator=inits[j])
            ax[i, j].scatter(X[:, 0], X[:, 1], c=est.labels)
            contour_plot(0, 100, 0, 100, 100, 100, est, ax[i, j])
    plt.show()


if __name__ == '__main__':
    X = load_data()
    part_a(X, np.arange(1, 10), start_generator=StartGenerators.random_init)
    part_b()
