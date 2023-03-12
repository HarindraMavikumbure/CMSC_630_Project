import numpy as np


class Histogram:
    """
        This class contains the Histogram related functions such as create histogram, histogram equalization, average
        histogram
    """

    def __init__(self):
        # Store class wise histogram values
        self.averaged_histograms = None
        self.averaged_histograms_bins = None
        self.class_wise_histograms = {'cyl': list(),
                                      'inter': list(),
                                      'let': list(),
                                      'mod': list(),
                                      'para': list(),
                                      'super': list(),
                                      'svar': list()
                                      }

        self.class_wise_histograms_bins = {'cyl': list(),
                                           'inter': list(),
                                           'let': list(),
                                           'mod': list(),
                                           'para': list(),
                                           'super': list(),
                                           'svar': list()
                                           }

    def histogram(self, image):
        hist = np.zeros(256)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i][j]] += 1

        return hist

    def create_histogram(self, image, bins=256, span=None):
        """
        Performs histogram calculation (defaults to 255 bins).
        """
        if span is None:
            span = [0, 255]
        bin_values, bins = np.histogram(image, bins, span)
        return bin_values, bins

    def histogram_equalization(self, image):
        """
        Performs histogram equalization on a given image.
        """

        num_pixels = image.size  # get total number of pixels within the image
        bin_values, bins = self.create_histogram(image)  # get histogram
        number_of_bins = len(bins) - 1  # get total number of bins
        equalized_bin_values = []  # bin values after histogram is equalized
        current_sum = 0

        for i in range(number_of_bins):
            current_sum = current_sum + bin_values[i]
            equalized_bin_values.append(round((current_sum * 255) / num_pixels))

        # Create equalized image
        image_list = list(image.astype(int).flatten())
        equalized_image_list = []
        for i in image_list:
            equalized_image_list.append(equalized_bin_values[i])

        equalized_image = np.reshape(np.asarray(equalized_image_list), image.shape)

        return equalized_image

    def average_histogram_per_class(self):
        """
        Averages all histograms in class_wise_histograms dictionary
        """
        averaged_histograms = {}
        averaged_histograms_bins = {}

        for i in self.class_wise_histograms:
            if len(self.class_wise_histograms[i]) != 0:
                averaged_histograms[i] = np.mean(self.class_wise_histograms[i], axis=0)

        for i in self.class_wise_histograms_bins:
            if len(self.class_wise_histograms_bins[i]) != 0:
                averaged_histograms_bins[i] = np.mean(self.class_wise_histograms_bins[i], axis=0)

        # Save averaged histograms into class variable
        self.averaged_histograms = averaged_histograms
        self.averaged_histograms_bins = averaged_histograms_bins
