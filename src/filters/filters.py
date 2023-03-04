import numpy as np


class Filters:

    def __init__(self, output_path):
        self.path = output_path

    def linear_filter(self, gray_img, filter, size, dtype=np.uint8):
        """
        apply linear filter based on user defined filter size and weights

        Parameters
        ----------
        gray_img
        filter
        scale
        dtype

        Returns
        -------
        image

        """

        filter = np.array(filter)
        f_w, f_h = filter.shape
        i_w, i_h = gray_img.shape
        o_w = i_w - f_w + 1
        o_h = i_h - f_h + 1
        new_img = np.zeros((o_w, o_h))
        for i, j in np.ndindex(new_img.shape):
            result = np.sum(filter * gray_img[i: i + f_w, j: j + f_h])
            scaled_result = result / (size[0] * size[1])
            new_img[i, j] = scaled_result
        new_img = np.rint(new_img)
        new_img = new_img.astype(dtype)
        return new_img

    def median_filter(self, gray_img, weights):
        weights = np.array(weights)
        f_w, f_h = weights.shape
        i_w, i_h = gray_img.shape
        o_w = i_w - f_w + 1
        o_h = i_h - f_h + 1
        new_img = np.zeros((o_w, o_h))
        for i, j in np.ndindex(new_img.shape):
            pixel_list = np.array([])
            for k, l in np.ndindex(f_w, f_h):
                pixel_list = np.append(
                    pixel_list,
                    [gray_img[i: i + f_w, j: j + f_h][k, l]] * weights[k, l],
                )
            result = np.median(pixel_list)
            new_img[i, j] = result
        new_img = np.rint(new_img)
        new_img = new_img.astype(np.uint8)
        return new_img
