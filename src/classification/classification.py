import copy
import math
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.util.utils import Utils
from scipy.stats import zscore, stats


class Classification:
    """
            This class contains the KNN based classification related functions
    """

    def __init__(self, output_path):
        self.path = output_path

    # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = dataset.values.tolist()
        fold_size = int(len(dataset) / n_folds)

        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = random.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None

            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors):
        neighbors = self.get_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    def sklearn_knn(self, X_train, y_train, X_test, num_neighbors):
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return y_pred

    # kNN Algorithm
    def k_nearest_neighbors(self, train, test, num_neighbors):
        predictions = list()
        for row in test:
            output = self.predict_classification(train, row, num_neighbors)
            predictions.append(output)
        return predictions

    def knn_classifcation(self, df_read, fold, k):
        # df_read = df_read.astype('float64')
        sklearn = False
        if sklearn:
            y = df_read['label']
            X = df_read.drop('label', axis=1)

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Scale the features using StandardScaler
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            pred = self.sklearn_knn(X_train, y_train, X_test, k)
            accuracy = accuracy_score(y_test, pred)
            print('Scores: %s' % accuracy)

            k_values = [i for i in range(2, 7)]
            scores = []

            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                score = cross_val_score(knn, X, y, cv=10)
                scores.append(np.mean(score))
            best_index = np.argmax(scores)
            best_k = k_values[best_index]
            print(best_k)
            knn = KNeighborsClassifier(n_neighbors=best_k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(scores)
            print('Mean Accuracy: %.3f%%' % accuracy)
            return accuracy
        else:
            scores = self.evaluate_algorithm(df_read, self.k_nearest_neighbors, fold, k)
            print('Scores: %s' % scores)
            print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            return scores
