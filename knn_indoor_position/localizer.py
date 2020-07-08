"""This Module contains a KNN regression based indoor positioning algorithm"""

import numpy as np

class KnnLocalizer:

    """ KNN Regression Class for indoor localization using a 3-dimensional label space
        e.g.  Longitude, Latitude, and Floor

    Attributes:
        features - training data features
        labels -  training data labels
        k (int) number of nearest neighbors

    """

    def __init__(self, features, labels, k=3):

        self.features = np.array(features)
        self.labels = np.array(labels)
        self.k = k

    @staticmethod
    def euclidean_distance(array1, array2):

        """ Euclidean distance function without squareroot taken for efficiency

		Args:
			Two arrays of equal length

		Returns:
			Distance squared between array1 and array2
		"""

        distance = np.sum((array1-array2)**2)
        return distance

    @staticmethod
    def timsort(array, element):

        """ Sort method based on Python's sort() function that sorts tuples in descending order

        Args:
            array (array or list of tuples) array of tuples containing distance and label
            element (Int) the index of the element within tuple that is used to sort tuples

        Returns:
            Array of tuples sorted in descending order at index given by element
        """

        array.sort(key=lambda tup: tup[element])
        return array

    @staticmethod
    def coordinate_mean(distance_list):

        """ Method for calculating the mean predictions from the 3D label-space
        logitude,latitude, and floor from array of tuples containing coordinates
        and sorted distances. (floor is rounded to nearest floor.)

        Args:
            Array of tuples containing position coordinates

        Returns:
            Array of predicted coordinates
        """

        count = len(distance_list)
        new_list = []

        #isolate coordinate labels to new array
        for i in distance_list:
            new_list.append(i[0])
        new_array = np.array(new_list)

        # Calculate mean of each coordinate; round floor
        new_array = np.sum(new_array, 0) / count
        new_array[2] = np.around(new_array[2], int)
        return new_array

    def knn_regression(self, instance):

        """ KNN regression function modified to return regression predictions
         of 3-dimensional labelspace: (logitude, latitude, floor).

        Args:
            Test_features (array) test data for which to predict location

        Returns:
            An array containing coordinate prediction
        """
        temp_results = []
        for i in range(len(self.features)):
            distance = self.euclidean_distance(instance, self.features[i])
            temp_results.append((self.labels[i], distance))
        sorted_distances = self.timsort(temp_results, 1)
        results = self.coordinate_mean(sorted_distances[:self.k])
        return np.stack(results, axis=0)

    def fit_predict(self, test_data):

        """ KNN Localizer prediction function

        Args:
        	test_data (array) test feature data

        Returns:
        	List of arrays containing coordinate predictions
        """

        predictions = []
        for sample in test_data:
            predictions.append(self.knn_regression(sample))
        return predictions
