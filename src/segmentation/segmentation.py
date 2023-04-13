import math
import os
import random
import numpy as np


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

    def k_means_segmentation(self, img, k=2, max_iters=10):
        image = np.array(img)
        # Initialize centroids randomly
        rows, cols = image.shape
        centroids = np.random.rand(k, 1) * 255

        # Loop over iterations
        for i in range(max_iters):
            # Assign each pixel to closest centroid
            labels = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    pixel = image[r, c]
                    distances = np.linalg.norm(centroids - pixel, axis=1)
                    labels[r, c] = np.argmin(distances)
                    #print(distances)

            # Update centroids as mean of all pixels assigned to them
            for k in range(k):
                cluster_pixels = image[labels == k]
                if len(cluster_pixels) > 0:
                    centroids[k] = np.mean(cluster_pixels, axis=0)

        # Assign each pixel to final centroid
        segmented_img = np.zeros_like(image)

        for r in range(rows):
            for c in range(cols):
                pixel = image[r, c]
                distances = np.linalg.norm(centroids - pixel, axis=1)
                label = np.argmin(distances)
                segmented_img[r, c] = centroids[label]

        return segmented_img