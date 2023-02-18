from itertools import combinations

import numpy as np
import numpy.ma as ma


class KMeans:

    def __init__(self, data):

        self.X = data.reshape(data.shape[0], -1)
        self.n_examples, self.n_dims = self.X.shape
        self.n_clusters = 2
        self.combinations_ = list(
            combinations(range(self.n_examples), self.n_clusters)
            )
        self.n_combinations = len(self.combinations_)
        self.labels = None
        self.cluster_centers = None
        self.radius = None
        self.distances = None

    def fit(self, max_iter=300):

        # broadcast X to number of combinations
        X_ = np.broadcast_to(
            self.X,
            shape=(self.n_combinations, self.n_examples, self.n_dims)
            )

        # broadcast X to the number of clusters and combinations with
        # the order of axes (n_combinations, n_clusters, n_examples, n_dims)
        X__ = np.broadcast_to(
            self.X,
            shape=(self.n_combinations, self.n_clusters, *self.X.shape)
            )

        cluster_centers = self.X[self.combinations_, :]
        center_of_mass = np.zeros_like(cluster_centers)

        labels = np.zeros((self.n_combinations, self.n_examples))
        labels_ = None

        iter_ = 0

        while (iter_ < max_iter):

            iter_ += 1

            distances = np.abs(X__ - cluster_centers[:, :, np.newaxis])
            distances = np.mean(distances, axis=-1)

            labels_ = np.argmin(distances, axis=1)

            if (labels != labels_).any():
                labels = labels_
            else:
                break

            for i in range(self.n_clusters):

                broadcasted_labels = np.broadcast_to(
                    labels,
                    shape=(self.n_dims, *labels.shape)
                    )
                broadcasted_labels = np.transpose(broadcasted_labels, (1, 2, 0))

                mask = (broadcasted_labels != i)
                masked_X = ma.masked_where(mask, X_)

                center_of_mass[:, i, :] = np.mean(masked_X, axis=1)

            cluster_centers = center_of_mass

        distance_var = np.mean(distances ** 2, axis=-1)
        avg_radius = np.mean(distance_var, axis=-1)

        best_centroid_index = np.argmin(avg_radius)

        self.cluster_centers = cluster_centers[best_centroid_index]
        self.labels = labels[best_centroid_index]
        self.distances = distances[best_centroid_index]
        self.radius = avg_radius[best_centroid_index]

        return self
    