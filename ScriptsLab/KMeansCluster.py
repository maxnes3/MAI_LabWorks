import numpy as np


class KMeansCluster:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Инициализация центроидов случайными точками из данных
        centroids = X[np.random.choice(len(X), self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Расчет расстояний от каждой точки до центроидов
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)

            # Присваивание кластеров в соответствии с ближайшим центроидом
            labels = np.argmin(distances, axis=1)

            # Обновление центроидов
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Проверка на сходимость
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return labels, centroids