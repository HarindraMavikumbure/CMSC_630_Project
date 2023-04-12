import multiprocessing
import os
import time
from pathlib import Path
import yaml
from filters.filters import Filters
from histogram.histogram import Histogram
from noise.noise import Noise
from src.edge.edge import Edge_Detection
from src.morphological.morphological import Morphological_Ops
from src.segmentation.segmentation import Segmentation
from util.utils import Utils


class BatchProcessor:
    """
            This class contains the main starting point to initialize all the operations based on user inputs from the
            config.yaml file
    """

    def __init__(self, config_file=None):
        # create or set paths for files
        if config_file is not None:
            with open(config_file, "r") as file:
                self._config = yaml.safe_load(file)

        self.datasets_path = self._config["datasets_path"]

        self.save_path = self._config["image_results_path"]
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(self.save_path):
            os.remove(os.path.join(self.save_path, f))

        self.stats_path = self._config["stat_results_path"]
        Path(self.stats_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(self.stats_path):
            os.remove(os.path.join(self.stats_path, f))

        self.hist_path = self._config["histogram_results_path"]
        Path(self.hist_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(self.hist_path):
            os.remove(os.path.join(self.hist_path, f))

        self.run_parallel = self._config["run_parallel"]
        self.function_list = self._config["function_list"]
        self.histogram_avg_class_list = self._config["histogram_avg_class_list"]
        self.filter_mask_size = self._config["filter_mask_size"]
        self.median_filter_weights = self._config["median_filter_weights"]
        self.linear_filter_weights = self._config["linear_filter_weights"]
        self.edge_detector_type = self._config["edge_detector_type"]
        self.k = self._config["k"]
        self.max_iters = self._config["max_iters"]

        # initialize classes for different operations
        self.filters = Filters(output_path=self.save_path)
        self.utils = Utils(output_path=self.save_path, histogram_path=self.hist_path, stat_path=self.stats_path)
        self.noise_functions = Noise(self._config["salt_pepper_noise_strength"],
                                     self._config["gaussian_noise_mean"], self._config["gaussian_noise_std"])
        self.histogram_functions = Histogram()
        self.edge_detection = Edge_Detection(output_path=self.save_path)
        self.morphological_ops = Morphological_Ops(output_path=self.save_path)
        self.segmentation = Segmentation(output_path=self.save_path)

        self.func_wise_time_stats = {}
        self.time_stats = []

        # functionality dictionary will be used to choose function which is input from the config.yaml
        self.function_dictionary = {'1': self.rgb_to_gray,
                                    '2': self.add_salt_pepper_noise,
                                    '3': self.add_gaussian_noise,
                                    '4': self.create_histogram,
                                    '5': self.histogram_equalization,
                                    '6': self.create_average_histograms,
                                    '7': self.apply_linear_filter,
                                    '8': self.apply_median_filter,
                                    '9': self.edge_detection_func,
                                    '10': self.apply_erosion,
                                    '11': self.apply_dilation,
                                    '12': self.hist_thresholding,
                                    '13': self.k_means_clustering}

        self.class_dictionary = {'1': 'cyl',
                                 '2': 'inter',
                                 '3': 'let',
                                 '4': 'mod',
                                 '5': 'para',
                                 '6': 'super',
                                 '7': 'svar'}

    def save_timing_data(self, result):
        self.time_stats.append(result[0])

    def rgb_to_gray(self, image, cur_image_path):
        return self.utils.rgb_to_grayscale(image)

    def add_salt_pepper_noise(self, image, cur_image_path):
        return self.noise_functions.add_salt_and_pepper_noise(image)

    def add_gaussian_noise(self, image, cur_image_path):
        return self.noise_functions.add_gaussian_noise(image)

    def create_histogram(self, image, cur_image_path):
        bins, vals = self.histogram_functions.create_histogram(image)
        self.utils.save_histogram(bins, vals, cur_image_path)
        return None

    def histogram_equalization(self, image, cur_image_path):
        return self.histogram_functions.histogram_equalization(image)

    def resolve_image_class(self, cur_image_path):
        class_result = None
        for class_id in self.histogram_avg_class_list:
            class_val = self.class_dictionary.get(str(class_id))
            if class_val in cur_image_path:
                class_result = class_val
                return class_result
        return class_result

    def create_average_histograms(self, image, cur_image_path):
        class_val = self.resolve_image_class(cur_image_path)
        bins, vals = self.histogram_functions.create_histogram(image)
        if class_val is not None:
            self.histogram_functions.class_wise_histograms[class_val].append(bins)
            self.histogram_functions.class_wise_histograms_bins[class_val].append(vals)

    def apply_linear_filter(self, image, cur_image_path):
        return self.filters.linear_filter(image, self.linear_filter_weights, self.filter_mask_size)

    def apply_median_filter(self, image, cur_image_path):
        return self.filters.median_filter(image, self.median_filter_weights)

    def edge_detection_func(self, image, cur_image_path):
        return self.edge_detection.edge_detection(image, self.edge_detector_type)

    def apply_erosion(self, image, cur_image_path):
        return self.morphological_ops.apply_erosion(image)

    def apply_dilation(self, image, cur_image_path):
        return self.morphological_ops.apply_dilation(image)

    def hist_thresholding(self, image, cur_image_path):
        return self.segmentation.hist_thresholding(image)

    def k_means_clustering(self, image, cur_image_path):
        return self.segmentation.k_means_segmentation(image, self.k, self.max_iters)

    def process(self, path, function_list):

        time_start = time.time()
        current_process = 1
        if self.run_parallel:
            current_process = multiprocessing.Process().name
            print(f"{current_process} : processing the image {path}")

        # load image
        image = self.utils.get_image(path)

        # apply all requested functions
        for i in function_list:
            image = self.function_dictionary[str(i)](image, path)
            if 'image' in locals() and image is not None:
                self.utils.save_image(image, path)  # save image as grayscale
        print(f"{current_process} : done with image {path}")

        # return the time took to complete the process
        print([time.time() - time_start])
        return [time.time() - time_start]

    def run_parallel_batch_mode(self):
        # Creating the process pool based on the cpu count
        pool = multiprocessing.Pool(os.cpu_count())
        _ = [pool.apply_async(self.process, callback=self.save_timing_data, args=(path, self.function_list)) for path in
             self.utils.get_image_path(self.datasets_path)]

        if 6 in self.function_list:
            start = time.time()
            self.histogram_functions.average_histogram_per_class()
            for key, bins in self.histogram_functions.averaged_histograms.items():
                for key1, vals in self.histogram_functions.averaged_histograms_bins.items():
                    if key1 == key:
                        self.utils.save_histogram(bins, vals, key + '.BMP')
            self.time_stats.append(time.time() - start)

        pool.close()
        pool.join()

    def run_batch_mode(self):
        for path in self.utils.get_image_path(self.datasets_path):
            time_spent = self.process(path, self.function_list)
            self.save_timing_data(time_spent)

        if 6 in self.function_list:
            start = time.time()
            self.histogram_functions.average_histogram_per_class()
            for key, bins in self.histogram_functions.averaged_histograms.items():
                for key1, vals in self.histogram_functions.averaged_histograms_bins.items():
                    if key1 == key:
                        self.utils.save_histogram(bins, vals, key + '.BMP')
            self.time_stats.append(time.time() - start)


if __name__ == "__main__":
    config_file_path = "../config.yaml"
    processor = BatchProcessor(config_file_path)

    start_time = time.time()

    if processor.run_parallel:
        processor.run_parallel_batch_mode()
    else:
        processor.run_batch_mode()

    print("\n--- Batch Processing Time: %s seconds ---" % (time.time() - start_time))

    average_processing_time = sum(processor.time_stats) / len(processor.time_stats)

    print("--- Processing Time Per Image: %s seconds ---\n" % average_processing_time)

    processor.func_wise_time_stats['Batch processing time'] = time.time() - start_time
    processor.func_wise_time_stats['processing time per image'] = average_processing_time
    processor.utils.save_stat_to_csv(processor.func_wise_time_stats)
