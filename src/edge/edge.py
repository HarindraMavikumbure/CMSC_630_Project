import math

import numpy as np


class Edge_Detection:
    """
            This class contains the Edge detection related functions
    """

    def __init__(self, output_path):
        self.path = output_path

    # function to apply the edge detector
    def apply_edgeDetector(self, image, edge_filter):
        img_copy = np.zeros(image.shape)
        skip = int((edge_filter.shape[0] - 1) / 2)
        if edge_filter.shape[0] % 2 != 0 and edge_filter.shape[1] % 2 != 0:
            for i in range(0, image.shape[0] - edge_filter.shape[0] + 1):
                for j in range(0, image.shape[1] - edge_filter.shape[1] + 1):
                    img_copy[i + skip, j + skip] = np.mean(
                        image[i:(i + edge_filter.shape[0]), j:(j + edge_filter.shape[1])] * edge_filter)
        return img_copy

    def sobel_detector(self, image):
        img = np.zeros(image.shape)
        # X and Y directional filters
        filt_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        filt_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        im_horizontal = self.apply_edgeDetector(image, filt_x)
        im_vertical = self.apply_edgeDetector(image, filt_y)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                img[i][j] = int(math.sqrt(im_horizontal[i][j] ** 2 + im_vertical[i][j] ** 2))

        return img

    def improved_sobel_detector(self, image):
        img = np.zeros(image.shape)

        # X and Y directional filters
        filt_x = np.array([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ])
        filt_y = np.array([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ])
        im_horizontal = self.apply_edgeDetector(image, filt_x)
        im_vertical = self.apply_edgeDetector(image, filt_y)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                img[i][j] = int(math.sqrt(im_horizontal[i][j] ** 2 + im_vertical[i][j] ** 2))
        return img

    def prewitt_detector(self, image):
        img = np.zeros(image.shape)

        filt_x = np.array([
            [-1, 0, 1],
            [-0, 0, 0],
            [-1, 0, 1]
        ])
        filt_y = np.array([
            [-1, -0, -1],
            [0, 0, 0],
            [1, 0, 1]
        ])
        im_horizontal = self.apply_edgeDetector(image, filt_x)
        im_vertical = self.apply_edgeDetector(image, filt_y)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                img[i][j] = int(math.sqrt(im_horizontal[i][j] ** 2 + im_vertical[i][j] ** 2))
        return img

    def edge_detection(self, image, detection_type):
        if detection_type[0] == 2:
            edges = self.improved_sobel_detector(image)
        elif detection_type[0] == 1:
            edges = self.sobel_detector(image)
        elif detection_type[0] == 3:
            edges = self.prewitt_detector(image)
        else:
            return None
        return edges