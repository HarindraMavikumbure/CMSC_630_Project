import math

import numpy as np

import cv2

class Morphological_Ops:
    """
            This class contains the Morphological functions such as dilation and erosion
    """

    def __init__(self, output_path):
        self.path = output_path

    def apply_dilation(self, bin_img):
        kernel = np.ones((3, 3), np.uint8)

        # Create a blank image to store the dilated image
        dilated_img = np.zeros_like(bin_img)

        # Loop through each pixel in the image
        for i in range(1, bin_img.shape[0] - 1):
            for j in range(1, bin_img.shape[1] - 1):

                # Check if the pixel is an edge (i.e., has a non-zero value)
                if bin_img[i, j] > 0:
                    # Dilate the edge by applying the kernel to the surrounding pixels
                    dilated_pixel = np.max(bin_img[i - 1:i + 2, j - 1:j + 2] * kernel)

                    # Set the dilated pixel value in the output image
                    dilated_img[i, j] = dilated_pixel

        return dilated_img

    def apply_erosion(self, img):
        img = np.array(img)

        # Define the kernel for erosion
        kernel = np.ones((3, 3))

        # Define the eroded image array
        eroded_img = np.ones_like(img) * 255

        # Perform erosion
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if np.min(img[i - 1:i + 2, j - 1:j + 2]) == 0:
                    eroded_img[i, j] = 0

        return eroded_img