from builtins import object
import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self,X,y):
        """
        Function for initialising training data
        for KKNearestNeighbor

        Inputs:
        X - numpy array of shape (num_train,D) containing training samples each of dimension D
        y - numpy array of shape (N,) containing the class labels for each input where
            y[i] is the label for X[i]

        """
        self.X_train = X
        self.y_train = y

    def predict(self,X,k=1):
        """
        Predicts the labels for test data using classifier

        Inputs:
        X - numpy array of shape (num_test,D) containing the test samples, each of dimension D
        k - number of nearest neighbors that vote for the predicted labels

        Returns:
        y_test - numpy array of shape (num_test,) containing the predicted labels for test data
                 where y_test[i] is label for X_test[i]
        """
        dists = self.compute_distance(X)
        y_test = self.predict_labels(dists,k=k)
        return y_test

    def compute_distance(self,X):
        """
        Function to calculate the Euclidean(L2) distance between two numpy arrays

        Inputs:
        X - numpy array of shape (num_test,D) containing the test samples, each of dimension D

        Returns:
        dists - numpy array of shape (num_test, num_train) where dists[i, j]
                is the Euclidean distance between the ith test point and the jth training
                point.        
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))
        testsq = np.sum(np.square(X),axis=1).reshape(num_test,1)
        trainsq = np.sum(np.square(self.X_train),axis=1).reshape(1,num_train)
        dists = np.sqrt(testsq + trainsq - 2 * (np.dot(X,self.X_train.T)))
        return dists

    def predict_labels(self,dists,k=1):
        """
        Function to find the label for each test point using the distance between
        train and test matrices

        Inputs:
        dists - numpy array of shape (num_test, num_train) where dists[i, j]
                is the Euclidean distance between the ith test point and the jth training
                point.
        k - number of nearest neighbors that vote for the predicted labels 

        Returns:
        y_test - numpy array of shape (num_test,) containing the predicted labels for test data
                 where y_test[i] is label for X_test[i]
        """
        num_test = dists.shape[0]
        y_test = np.zeros((num_test,))
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])][:k]
            y_test[i] = np.argmax(np.bincount(closest_y))
        return y_test
