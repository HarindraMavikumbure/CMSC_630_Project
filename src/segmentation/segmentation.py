import math
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy import floor


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

    def map_pixels(self, image_flat, cluster_centers, pixel_mapping_temp, num_cluster_centers, num_image_rows):
        """
        Called from Segmentation.k_means_segmentation. Maps all image pixels to a cluster by taking a 1x3 pixel and
        calculating the distance between every cluster center. Appends pixels location to the cluster it has minimum
        distance to.
        Parameters:
        -----------
            image_flat (numpy array) : the 3D flattened image to calculate the distances over
            cluster_centers (numpy array) : array of all cluster centroids
            pixel_mapping_temp (numpy array) : the mapping of pixel to centroid. This array is passed and changed by reference.
            num_cluster_centers (int) : the number of clusters
            num_image_rows (int) : the length of image_flat which is equal to the number of pixels in the image
        Returns:
        --------

            None
        """

        # For every pixel in the image
        for i in range(num_image_rows):
            distances = []
            # For every cluster calculate which one the pixel is closest to
            for j in range(num_cluster_centers):
                # calculating Euclidean distance for the pixel against each cluster
                distances.append(np.linalg.norm(image_flat[i][:] - cluster_centers[j]))
            # For every pixel map to which cluster it belongs based on which one it is closest to
            pixel_mapping_temp[i] = distances.index(min(distances))

    def recalculate_cluster_centers(self, image_flat, cluster_centers, pixel_to_cluster_mapping, num_cluster_centers,
                                    num_image_rows):
        """
        Called from Segmentation.k_means_segmentation.Recalculate cluster centers by averaging of all pixel values that
        are assigned to the cluster. This cluster to pixel assignment is contained in the pixel_to_cluster_mapping array.
        Cluster centers are then moved to their new centers of gravity. cluster_centers is passed and changed by reference.
        If a cluster winds up with no pixels assigned to it, then that cluster will be randomly moved to a different location.
        Parameters:
        -----------
            image_flat (numpy array) : the 3D flattened image
            cluster_centers (numpy array) : array of all cluster centroids. Passed and changed by reference.
            pixel_to_cluster_mapping (numpy array) : the mapping of pixel to centroid.
            num_cluster_centers (int) : the number of clusters
            num_image_rows (int) : the length of image_flat which is equal to the number of pixels in the image
        Returns:
        --------

            None
        """

        # Calculate every new cluster center
        for i in range(num_cluster_centers):
            cluster_sum = np.zeros(3)
            cluster_average = np.zeros(3)
            pixel_count = 0

            for j in range(num_image_rows):
                # If a pixel belongs to a cluster then sum its values with other cluster pixels
                if pixel_to_cluster_mapping[j] == i:
                    cluster_sum = cluster_sum + image_flat[j][:]
                    pixel_count = pixel_count + 1

            # Only recalculate cluster center if a pixel is assigned to it
            if pixel_count != 0:
                cluster_average = cluster_sum / pixel_count  # get average of pixel values
                cluster_centers[i][:] = cluster_average  # move cluster center
            # If a cluster center has no pixels assigned to it then reinitialize it randomly
            else:
                cluster_centers[i][0] = random.randint(0, 255)  # red
                cluster_centers[i][1] = random.randint(0, 255)  # green
                cluster_centers[i][2] = random.randint(0, 255)  # blue

    def segment_image(self, image_flat, cluster_centers, pixel_to_cluster_mapping, num_cluster_centers, num_image_rows):

        # Calculate every new cluster center
        for i in range(num_cluster_centers):
            for j in range(num_image_rows):
                # If a pixel belongs to a cluster reassign it's value to that of it's cluster centroid
                if pixel_to_cluster_mapping[j] == i:
                    image_flat[j][:] = cluster_centers[i][:]

    # Function to perform k means clustering
    def k_means_segmentation(self, image, num_cluster_centers, k_means_max_iterations):
        print("K means")
        original_image_height, original_image_width, _ = image.shape
        mid_row = floor(original_image_height / 2)
        mid_column = floor(original_image_width / 2)
        image_copy = np.copy(image)  # make copy of image
        # get coordinates of pixels in (x,y)
        # xy_coords = np.flip(np.column_stack(np.where(image_copy[:,:,0] >= 0)), axis=1)

        # reshape the array into 2D matrix of 3 columns with each column representing a color spectrum
        image_flat = np.reshape(image_copy, (image_copy.shape[0] * image_copy.shape[1], image_copy.shape[2]))
        num_image_rows, num_image_columns = image_flat.shape

        # initialize new array with height equal to number of clusters and width of num_image_columns
        cluster_centers = np.empty([num_cluster_centers, num_image_columns], dtype=np.float64)

        # list[i] of which pixel of image_flat[i][:] belongs to which cluster
        pixel_to_cluster_mapping = np.empty([num_image_rows, 1], dtype=np.float64)
        pixel_mapping_temp = np.copy(pixel_to_cluster_mapping)  # temp mapping array

        # randomly initialize cluster centers after setting first cluster to central pixel's color
        for i in range(num_cluster_centers):
            # ensure first cluster finds color of central pixel in image
            if i == 0:
                cluster_centers[i][0] = image[mid_row][mid_column][0]  # red
                cluster_centers[i][1] = image[mid_row][mid_column][1]  # green
                cluster_centers[i][2] = image[mid_row][mid_column][2]  # blue
            else:
                cluster_centers[i][0] = random.randint(0, 255)  # red
                cluster_centers[i][1] = random.randint(0, 255)  # green
                cluster_centers[i][2] = random.randint(0, 255)  # blue

        pixel_changed_cluster_flag = True
        iteration = 0

        # Perform K-means segmentation. Loop as long as pixels are being reassigned to clusters and max iterations not reached.
        while pixel_changed_cluster_flag and (iteration != k_means_max_iterations):
            # pixel_mapping_temp numpy array passed by reference
            self.map_pixels(image_flat, cluster_centers, pixel_mapping_temp, num_cluster_centers, num_image_rows)

            # Check if any pixels changed clusters
            comparison = pixel_to_cluster_mapping == pixel_mapping_temp
            if comparison.all():
                # no pixels changed clusters but need to check if all clusters have pixels assigned to them
                for i in range(num_cluster_centers):
                    # break if cluster found with no pixels assigned
                    if i not in pixel_mapping_temp:
                        pixel_changed_cluster_flag = True
                        # Randomly reassign cluster which has no pixels
                        cluster_centers[i][0] = random.randint(0, 255)  # red
                        cluster_centers[i][1] = random.randint(0, 255)  # green
                        cluster_centers[i][2] = random.randint(0, 255)  # blue
                        break  # break out of for loop
                    else:
                        pixel_changed_cluster_flag = False
            else:
                pixel_changed_cluster_flag = True  # signal that pixels changed cluster assignments
                pixel_to_cluster_mapping = np.copy(pixel_mapping_temp)  # set mapping to temp mapping

                # calculate new cluster_centers based on average of pixels assigned to it
                self.recalculate_cluster_centers(image_flat, cluster_centers, pixel_to_cluster_mapping,
                                            num_cluster_centers, num_image_rows)
                print(cluster_centers)

            iteration = iteration + 1  # keep track of num iterations

        # Reassign image pixel values to the cluster centroid values
        self.segment_image(image_flat, cluster_centers, pixel_to_cluster_mapping, num_cluster_centers, num_image_rows)
        print("Done")
        return np.reshape(np.rint(image_flat), image.shape)