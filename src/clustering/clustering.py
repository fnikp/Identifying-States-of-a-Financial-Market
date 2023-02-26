from itertools import combinations

import numpy as np
import numpy.ma as ma


class KMeans:
    """K-means clustering algorithm

    :param n_clusters: number of clusters for the K-means algorithm,
    the default is 2.
    :type n_clusters: = int
    :param max_iter: maximum number of iterations of the k-means algorithm for
    a single run.
    :type max_iter: int
    :pram filtered: if True, the filtered set of combinations for the initial
    centroids are considered. the default is False
    :type filtered: bool
    :param metric: the metric for the ditance between the objects.
    :type metric: callable or None.
    """

    def __init__(self, n_cluster=2, max_iter=50, filtered=False, metric=None):

        self.n_clusters = n_cluster
        self.max_iter = max_iter
        self.filtered = filtered
        self.metric = metric
        self.combinations_ = None
        self.n_combinations = None
        self.metric = None
        self.labels = None
        self.cluster_centers = None
        self.radius = -1
        self.distances = None

    def __similarity__(self, X, Y=None):
        """calculate the distances between two array X and Y based on a
        specific similarity measure.

        :param X, Y: nd-array of the objects
        """

        if Y is None:
            Y = np.zeros_like(X)

        if self.metric is None:
            distances = np.abs(X - Y)
            distances = np.mean(distances, axis=-1)

        else:
            distances = self.metric(X, Y)

        return distances

    def __filtered_centroids_labels__(self, coordinates, n_clusters):
        """filters the initial centroids labels and choose only those
        combinations that the distance between the centroids is less than
        the mean of the whole combinataions

        :param coordinates: the coordinates of the training matrices based on
        the similarity measure
        """

        center_of_mass = np.mean(coordinates)

        selected_index = np.where(coordinates <= center_of_mass)[0]
        selected_centroids_labels = list(
            combinations(selected_index, self.n_clusters)
            )
        selected_centroids_labels = np.array(selected_centroids_labels)

        return selected_centroids_labels

    def __centroids_labels__(self, X, n_examples):
        """specify all the possible combinations for the inital centroids 
        labels

        :param X: the nd-array of the flatened training matrices
        :param n_examples: number of examples of the data
        """

        if self.filtered is True:
            coordinates = self.__similarity__(X)
            centroids_labels = self.__filtered_centroids_labels__(
                coordinates,
                self.n_clusters
                )

        else:
            centroids_labels = list(
                combinations(range(n_examples), self.n_clusters)
                )
            centroids_labels = np.array(centroids_labels)

        return centroids_labels

    def __label__(self, distances):
        """labels the objects to the closest centroids accordng to
        the given distances.

        :param distance: nd-array of the distance between objects
        and the cluster centers
        """

        return np.argmin(distances, axis=0)

    def __center_of_mass__(self, X, labels_comb):
        """calculate the center of mass of the clusters

        :param X: the nd-array of the flatened matrices
        :param labels_comb: the labels of the object for a specific
        combination of centroids
        """

        center_of_mass = np.zeros((self.n_clusters, *X[0].shape))

        for cluster in range(self.n_clusters):
            members = X[labels_comb == cluster]
            center_of_mass[cluster] = np.mean(members, axis=0)

        return center_of_mass

    def __clusters_radius__(self, distances):
        """calculate the average of the final radius for a specific combination

        :param distances: the nd-array of the distance between the objects and
        the cluster centers
        """

        distances_var = np.mean(distances ** 2, axis=-1)
        avg_radius = np.mean(distances_var, axis=-1)
        return avg_radius

    def __best_index__(self, clusters_radius):
        """choose the index of the best clustering in order to minimize the
        average radius of the clusters

        :param clusters_radius: the nd-array of the average radiuses of the
        clusters in each combination in a batch
        """
        return np.argmin(clusters_radius)

    def __recast__(self, X, centroids, n_combinations, n_examples):
        """labels the objects based on the given centroids, then recast the
        centroids to the center of mass of each cluster

        :param X: the nd-array of the flatened training matrices
        :param centroids: current cluster centers
        :param n_combinations: the number of combinations considered in a batch
        :param n_examples: the number of training examples
        """

        distances = np.zeros((n_combinations, self.n_clusters, n_examples))
        temp_labels = np.zeros((n_combinations, n_examples))
        center_of_mass = np.zeros_like(centroids)

        for comb in range(n_combinations):

            centroids_comb = centroids[comb]
            broadcasted_X = np.broadcast_to(
                X,
                shape=(self.n_clusters, *X.shape)
                )
            distances_comb = self.__similarity__(
                broadcasted_X,
                centroids_comb[:, np.newaxis]
                )
            distances[comb] = distances_comb

            labels_comb = self.__label__(distances_comb)
            temp_labels[comb] = labels_comb

            center_of_mass[comb] = self.__center_of_mass__(X, labels_comb)

        return distances, temp_labels, center_of_mass

    def __batch_KMeans__(self,
                         X,
                         initial_centroids_labels,
                         n_combinatiosn,
                         n_examples):
        """perfroms the K-means algorithm for a batch of combinations of
        the initial centroids

        :param X: the nd-array of the flatened training matrices
        :param initial_centroid_labels: the array of the indices of the
        initial centroids
        :param n_combinations: number of the combinations in a batch
        :param n_examples: number of the training examples
        """

        n_combinations = len(initial_centroids_labels)

        cluster_centers = X[initial_centroids_labels]

        labels = np.zeros((n_combinations, n_examples))

        iter_ = 0

        while (iter_ <= self.max_iter):
            iter_ += 1

            distances, temp_labels, center_of_mass = self.__recast__(
                X,
                cluster_centers,
                n_combinations,
                n_examples
                )
            if (labels != temp_labels).any():
                labels = temp_labels

            else:
                break

            cluster_centers = center_of_mass

        clusters_radius = self.__clusters_radius__(distances)
        best_index = self.__best_index__(clusters_radius)

        labels = labels[best_index]
        cluster_centers = cluster_centers[best_index]
        distances = distances[best_index]
        radius = clusters_radius[best_index]

        return labels, cluster_centers, distances, radius

    def fit(self, data):
        """perform the K-means algorithm on the given data.

        :param data: the training examples to cluster
        :type data: nd-array
        """

        X = data.reshape(data.shape[0], -1)
        n_examples, n_dims = X.shape

        centroids_labels = self.__centroids_labels__(X, n_examples)
        first_centroid_labels = np.unique(centroids_labels[:, 0])

        for i in first_centroid_labels:

            initial_centroids_labels = centroids_labels[
                centroids_labels[:, 0] == i
                ]
            n_combinations = len(initial_centroids_labels)

            (labels_,
             cluster_centers_,
             distances_,
             radius_) = self.__batch_KMeans__(
                X,
                initial_centroids_labels,
                n_combinations,
                n_examples
                )

            if (radius_ < self.radius) or (self.radius < 0):

                # print('________________________changed!______________________')

                self.radius = radius_
                self.labels = labels_
                self.distances = distances_
                self.cluster_centers = cluster_centers_

        return self
