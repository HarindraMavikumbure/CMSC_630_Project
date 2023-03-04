import numpy as np
import random
from numpy.random import default_rng


class Noise:

    def __init__(self, noise_strength, mean, std):
        self._rng = default_rng(seed=42)
        self.salt_and_pepper_noise_strength = noise_strength
        self.mean = mean
        self.std = std

    def add_salt_and_pepper_noise(self, image):
        output = np.zeros(image.shape, np.uint8)
        threshold = 1 - self.salt_and_pepper_noise_strength
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < self.salt_and_pepper_noise_strength:
                    output[i][j] = 0
                elif rdn > threshold:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def add_gaussian_noise(self, image):
        noise = self._rng.normal(loc=self.mean, scale=self.std, size=image.shape)
        image = image + noise
        image = np.clip(image, 0, 255)
        image = np.rint(image)
        image = image.astype(np.uint8)
        return image
