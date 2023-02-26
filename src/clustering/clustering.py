gfrom itertools import combinations

import numpy as np
import numpy.ma as ma


class KMeans:
    """K-means clustering algorithm

    :param n_clusters: number of clusters for the K-means algorithm, the default is 2.
    :type n_clusters: = int
    :param max_iter: maximum number of iterations of the k-means algorithm for a single run.
    :type max_iter: int
    :param metric: the metric for the ditance between the objects.
    :type metric: callable or None.
    """

    def __init__(self, n_cluster=2, , max_iter=50, metric=None):

        self.n_clusters = n_cluster
        self.max_iter = max_iter
        self.metric = metric
        self.combinations_ = None
        self.n_combinations = None
        self.metric = None
        self.labels = None
        self.cluster_centers = None
        self.radius = None
        self.distances = None

    def __initial_centroids_labels__(self, n_examples):
        """specify all the possible combinations for the inital centroids
        labels and set the value of the self.combinations_ and
        self.n_combinations

        :param n_examples: number of examples of the data
        """

        self.combinations_ = list(
            combinations(range(n_examples), self.n_clusters)
            )
        self.n_combinations = len(self.combinations_)

    def __similarity__(self, X, Y):
        """calculate the distances between two array X and Y based on a 
        specific similarity measure.

        :param X, Y: nd-array of the objects
        """

        if self.metric is None:
            distances = np.abs(X - Y)
            distances = np.mean(distances, axis=-1)

        else:
            distances = self.metric(X, Y)

        return distances

    def __label__(self, distances):
        """labels the objects to the closest centroids accordng to
        the given distances.

        :param distance: nd-array of the distance between objects
        and the cluster centers
        """

        return np.argmin(distances, axis=1)

    def __center_of_mass__(self, X, labels, n_dims):
        """calculate the center of mass of the clusters

        :param X: the nd-array of the flatened matrices
        :param labels: the nd-array of the labels of the object for
        each combination of clustering
        :param n_dims: the number of the dimensions of the flatened matrices
        """

        # broadcast X to number of combinations
        X_ = np.broadcast_to(
            X,
            shape=(self.n_combinations, *X.shape)
            )

        center_of_mass = np.zeros(
            (self.n_combinations, self.n_clusters, *X[0].shape)
        )

        for i in range(self.n_clusters):

            broadcasted_labels = np.broadcast_to(
                labels,
                shape=(n_dims, *labels.shape)
                )
            broadcasted_labels = np.transpose(broadcasted_labels, (1, 2, 0))

            mask = (broadcasted_labels != i)
            masked_X = ma.masked_where(mask, X_)

            center_of_mass[:, i, :] = np.mean(masked_X, axis=1)

        return center_of_mass

    def __clusters_radius__(self, distances):
        """calculate the average of the final radius for each combination

        :param distances: the nd-array of the distance between the objects and
        the cluster centers
        """

        distance_var = np.mean(distances ** 2, axis=-1)
        avg_radius = np.mean(distance_var, axis=-1)

        return avg_radius

    def __best_index__(self, avg_radius):
        """choose the index of the best clustering in order to minimize the
        average radius of the clusters

        :param avg_radius: the nd-array of the average radiuses of the
        clusters in each combination
        """
        return np.argmin(avg_radius)

    def fit(self, data):
        """perform the K-means algorithm on the given data.
        
        :param data: the training examples to cluster
        :type data: nd-array
        """

        X = data.reshape(data.shape[0], -1)
        n_examples, n_dims = X.shape

        self.__initial_centroids_labels__(n_examples)

        # broadcast X to the number of clusters and combinations with
        # the order of axes (n_combinations, n_clusters, n_examples, n_dims)
        X__ = np.broadcast_to(
            X,
            shape=(self.n_combinations, self.n_clusters, *X.shape)
            )

        cluster_centers = X[self.combinations_, :]

        labels = np.zeros((self.n_combinations, n_examples))
        temp_labels = None

        iter_ = 0

        while (iter_ < max_iter):

            iter_ += 1

            distances = self.__similarity__(
                X__,
                cluster_centers[:, :, np.newaxis]
                )

            temp_labels = self.__label__(distances)

            if (labels != temp_labels).any():
                labels = temp_labels
            else:
                break

            center_of_mass = self.__center_of_mass__(X, labels, n_dims)
            cluster_centers = center_of_mass

        avg_radius = self.__clusters_radius__(distances)

        best_centroid_index = self.__best_index__(avg_radius)

        self.cluster_centers = cluster_centers[best_centroid_index]
        self.labels = labels[best_centroid_index]
        self.distances = distances[best_centroid_index]
        self.radius = avg_radius[best_centroid_index]

        return self
