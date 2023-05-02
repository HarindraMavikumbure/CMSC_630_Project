import math
import os
import random
import numpy as np

from src.util.utils import Utils
from scipy import stats


class Classification:
    """
            This class contains the KNN based classification related functions
    """

    def __init__(self, output_path):
        self.path = output_path

    def KNN(self, test, train, k):
        new_label = np.zeros(len(test))
        for i in range(0, len(test)):
            features = train[:, 0:-1] - test[:, 0:-1][i]
            nearest_neighbors = np.sqrt(np.sum(features ** 2, axis=1))
            nearest_neighbors = np.append(nearest_neighbors[:, np.newaxis], np.reshape(train[:, -1], (len(train), 1)),
                                          axis=1)
            nearest_neighbors = nearest_neighbors[nearest_neighbors[:, 0].argsort()]
            nearest_neighbors = nearest_neighbors[0:k, :]
            mode_label = stats.mode(nearest_neighbors[:, 1])
            new_label[i] = mode_label[0]
        return new_label

    def cross_validation(self, features, folds, k):
        np.random.shuffle(features)
        test = int(len(features) / folds)
        new_label = np.array([])
        for i in range(1, folds + 1):
            train_features = np.copy(features)
            test_features = features[i * test - test:i * test, :]
            new_test_label = np.zeros(len(test_features))
            train_features = np.delete(train_features, slice(i * test - test, i * test), axis=0)
            new_test_label = self.KNN(test_features, train_features, k)
            new_label = np.hstack((new_label, new_test_label))
        return new_label

    def calculate_metrics(self, actual, predicted):
        """
        Calculates various classification metrics given actual and predicted labels
        """
        tp = 0  # True positives
        fp = 0  # False positives
        tn = 0  # True negatives
        fn = 0  # False negatives

        # Calculate true/false positives/negatives
        for i in range(len(actual)):
            if actual[i] == 1 and predicted[i] == 1:
                tp += 1
            elif actual[i] == 0 and predicted[i] == 1:
                fp += 1
            elif actual[i] == 0 and predicted[i] == 0:
                tn += 1
            elif actual[i] == 1 and predicted[i] == 0:
                fn += 1

        # Calculate precision, recall, and F1-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "true_positives": tp,
            "true_negatives": tn,
            "false_negatives": fn
        }

    def classifcation(self, df_read, fold, k):
        df_normalized = stats.zscore(df_read[:, 0:-1])
        df_normalized = np.append(df_normalized, np.reshape(df_read[:, -1], (len(df_read), 1)), axis=1)
        new_label = self.cross_validation(df_normalized, fold, k)
        metrics = self.calculate_metrics(df_normalized[:, -1], new_label)
        print('Metrics : ', metrics)
        return metrics
