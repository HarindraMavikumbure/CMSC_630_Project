import math
import os
import random
import numpy as np
from matplotlib import pyplot as plt


class Segmentation:
    """
            This class contains the Segmentation related functions
    """

    def __init__(self, output_path):
        self.path = output_path

    # Function to get binary image using histogram clustering
    def hist_thresholding(self, gray):
        pixel_number = gray.shape[0] * gray.shape[1]
        mean_weight = 1.0 / pixel_number
        his, bins = np.histogram(gray, np.arange(0, 257))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(256)
        for t in bins[1:-1]:
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            Wb = pcb * mean_weight
            Wf = pcf * mean_weight

            mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
            muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
            # print mub, muf
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = gray.copy()
        print(final_thresh)
        final_img[gray > final_thresh] = 255
        final_img[gray < final_thresh] = 0
        return final_img

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Function to perform k means clustering
    def kmeans_clustering(self, image, K, max_iters):
        clusters = [[] for _ in range(K)]
        centroids = []
        y_pred, centroids = self.predict(image, K, max_iters, clusters, centroids)
        centers = np.uint8(centroids)
        y_pred = y_pred.astype(int)
        np.unique(y_pred)
        labels = y_pred.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        return segmented_image

    def predict(self, X, K, max_iters, clusters, centroids):
        X = X
        n_samples, n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(n_samples, K, replace=False)
        centroids = [X[idx] for idx in random_sample_idxs]
        # Optimize clusters
        for _ in range(max_iters):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(centroids, K, X)
            # Calculate new centroids from the clusters
            centroids_old = centroids
            centroids = self._get_centroids(clusters, K, X, n_features)

            # check if clusters have changed
            if self._is_converged(centroids_old, centroids, K):
                break
        # Classify samples as the index of their clusters
        return self._get_cluster_labels(clusters, n_samples), centroids

    def _get_cluster_labels(self, clusters, n_samples):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids, K, X):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(K)]
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [self.euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters, K, X, n_features):
        # assign mean value of clusters to centroids
        centroids = np.zeros((K, n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids, K):
        # distances between each old and new centroids, fol all centroids
        distances = [self.euclidean_distance(centroids_old[i], centroids[i]) for i in range(K)]
        return sum(distances) == 0
