import math

import numpy as np


class Features:
    """
            This class contains the feature extraction related functions
    """

    def __init__(self, output_path):
        self.path = output_path

    def get_area(self, img):
        area = np.count_nonzero(img)
        print("area:", area)
        return area

    def get_perimeter(self,seg_img, dil_seg_img):
        int_bound = np.logical_xor(seg_img, dil_seg_img)
        perimeter = np.count_nonzero(int_bound)
        print("perimeter:", perimeter)
        return perimeter

    def center_of_mass(self, binary_image):
        # Get the dimensions of the image
        height, width = binary_image.shape

        # Calculate the total mass of the image
        total_mass = np.sum(binary_image)

        # Calculate the center of mass
        x_cen = 0
        y_cen = 0

        for i in range(height):
            for j in range(width):
                if binary_image[i, j] == 1:
                    x_cen += j
                    y_cen += i

        x_cen /= total_mass
        y_cen /= total_mass
        print("center of mass:" ,x_cen,y_cen)
        return x_cen, y_cen

    def moment_of_inertia(self, image, center_i, center_j):
        inertia_1 = 0
        inertia_2 = 0
        inertia_3 = 0
        print("calc inertia: ", inertia_3)

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                inertia_1 = inertia_1 + (((i - center_i) ** 2) * image[i][j])
                inertia_2 = inertia_2 + (((j - center_j) ** 2) * image[i][j])
                inertia_3 = inertia_3 + ((i - center_i) * (j - center_j) * image[i][j])
        print("inertia", round(inertia_1), round(inertia_2), round(inertia_3))
        return round(inertia_1), round(inertia_2), round(inertia_3)

    def cal_Eccentrisity(self, image, inertia_1, inertia_2, inertia_3, area):
        print("cal eccentrisity", inertia_3,area)
        eccentrisity = (((inertia_1 - inertia_2) ** 2) + 4 * inertia_3) / area
        print("eccentrisity", round(eccentrisity))
        return round(eccentrisity)

    def cal_orientation(self, image, inertia_1, inertia_2, inertia_3):
        orientation = 1 / 2 * math.degrees(math.atan(2 * inertia_3 / (inertia_1 - inertia_2)))
        print("orientation", round(orientation))
        return round(orientation)

    def cal_Compactness(self, area, perimeter):
        compactness = 4 * math.pi * area / perimeter ** 2
        print("compactness", round(compactness))
        return compactness

    def feature_extraction(self, b_image, e_image, dil_image):
        area = self.get_area(b_image)
        perimeter = self.get_perimeter(b_image,dil_image)
        compactness = self.cal_Compactness(area, perimeter)
        center_i, center_j = self.center_of_mass(b_image)
        inertia_1, inertia_2, inertia_3 = self.moment_of_inertia(b_image, center_i, center_j)
        eccentrisity = self.cal_Eccentrisity(b_image, inertia_1, inertia_2, inertia_3, area)
        orientation = self.cal_orientation(b_image, inertia_1, inertia_2, inertia_3)
        return area, perimeter, compactness, center_i, center_j, inertia_1, inertia_2, inertia_3, eccentrisity, orientation
