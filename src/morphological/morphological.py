import math

import numpy as np


class Morphological_Ops:
    """
            This class contains the Morphological functions such as dilation and erosion
    """

    def __init__(self, output_path):
        self.path = output_path

    def binarize(self, image, pivot=127):
        copy_image = image.copy()
        copy_image[copy_image > pivot] = 255
        copy_image[copy_image < pivot] = 0
        return copy_image

    def apply_erosion(self, image_edges, num_layers=1, structuring_element=np.array([[False]])):
        """
        Perform edge erosion of an edge based image using. Any size structuring element of
        boolean values can be defined. The True in the matrix defines the places to apply the element.
        Parameters:
        -----------
            image_edges (numpy_array) : 2D images of edges
            num_layers (int) : the number of times to apply the structuring element to the edges
            structuring_element (boolean numpy_array) : 2D array of boolean values which defines the
                                                        structuring element to perform erosion with
        Returns:
        --------
            image_edges_copy (numpy_array) : the eroded edge image
        """
        structuring_element = np.array([[False, True, False], [True, True, True], [False, True, False]])
        structure_height, structure_width = structuring_element.shape
        floor_structure_height = math.floor(structure_height / 2)
        floor_structure_width = math.floor(structure_width / 2)

        # get size of image
        height, width = image_edges.shape
        # copy image_edges
        image_edges_copy = np.copy(image_edges)
        edge_erosion_temp = np.copy(image_edges)

        # Remove number of specified layers
        for _ in range(num_layers):
            # go through every pixel of image
            for row in range(floor_structure_height, height - floor_structure_height):
                for column in range(floor_structure_width, width - floor_structure_width):
                    # if over an edge pixel apply the structuring element to it (intensity 0 == black)
                    if image_edges_copy[row][column] == 0:
                        # mapping of structuring element with zeros array and image
                        # numpy.where() iterates over the structuring element bool array
                        # and for every True it yields corresponding element from the first list (0 == for a black edge pixel)
                        # and for every False it yields corresponding element from the second list (image_edges_copy pixel)
                        mapping = np.where(structuring_element,
                                           np.zeros((structure_height, structure_width)),
                                           image_edges_copy[
                                           row - floor_structure_height:row + floor_structure_height + 1,
                                           column - floor_structure_width:column + floor_structure_width + 1])

                        # if structuring element present in edge image then retain pixel
                        if np.array_equal(mapping, image_edges_copy[
                                                   row - floor_structure_height:row + floor_structure_height + 1,
                                                   column - floor_structure_width:column + floor_structure_width + 1]):
                            edge_erosion_temp[row][column] = 0
                        # else remove pixel
                        else:
                            edge_erosion_temp[row][column] = 255

            image_edges_copy = np.copy(edge_erosion_temp)  # copy over in preparation for removing another layer

        return image_edges_copy
