import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Utils:

    def __init__(self, output_path, histogram_path, stat_path):
        self.save_images_path = output_path
        self.save_histogram_path = histogram_path
        self.save_stat_path = stat_path

    def get_image_path(self, root_dir):
        return [filename for filename in glob.iglob(root_dir + '**/*.BMP', recursive=True)]

    def get_image(self, image_path):
        """
        Loads an RGB image into a numpy array and returns it
        """
        img = plt.imread(image_path)  # load the image into a numpy array
        return img

    def rgb_to_grayscale(self, image):
        """
        Convert rgb image to grayscale
        """
        return np.asarray(np.rint((0.2989 * image[:, :, 0]) + (0.5870 * image[:, :, 1]) + (0.1140 * image[:, :, 2])),
                          dtype=np.float64)

    def save_stat_to_csv(self, stats):
        path = os.path.join(self.save_stat_path, 'statistics')  # join save path and filename
        data = pd.DataFrame.from_dict(stats, orient='index')
        data.to_csv(path)

    def save_histogram(self, bins, vals, image_pathname):
        """
        save histograms as images
        Returns
        -------
        None
        """
        plt.title("Histogram")
        plt.xlabel("pixel value")
        plt.ylabel("pixel count")
        plt.figure()
        plt.bar(vals[:-1] - 0.5, bins, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        filename = 'histogram_' + os.path.basename(
            image_pathname).split('.')[0] + '.jpg'  # extract filename of image from its pathname and add modified_
        path = os.path.join(self.save_histogram_path, filename)  # join save path and filename
        plt.savefig(path)  # Save histogram to image
        plt.close()
        print("saving histogram...")

    def save_image(self, image, image_pathname):
        """
        Save image as grayscale or rgb based on image dimensions.
        """
        filename = 'result_' + os.path.basename(
            image_pathname)  # extract filename of image from its pathname and add modified_
        path = os.path.join(self.save_images_path, filename)  # join save path and filename

        if len(image.shape) == 2:
            plt.imsave(path, image, cmap='gray', vmin=0, vmax=255)  # Save back grayscale image
        elif len(image.shape) == 3:
            # have to normalize image values between 0 and 1 to save as rgb
            plt.imsave(path, (image - np.min(image)) / (np.max(image) - np.min(image)))  # Save back rgb image
        else:
            print('Failed to save image! Image should be 2D or 3D')
