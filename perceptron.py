# Author: Youssef Aitbouddroub
# Supervised Learning : Perceptron implementation

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    A Perceptron is a simple linear binary classifier that uses a linear decision boundary to categorize new points into one of two classes.

    Attributes:
        learning_rate (float): The rate at which the model adjusts its weights during training. A smaller learning rate requires more training iterations but can lead to more precise convergence.
        n_iterations (int): The number of passes over the training dataset. Determines how many times the model's weights will be updated.
        weights (numpy.ndarray): The weights assigned to the input features after the model has been fit. Initially set to None.
        bias (float): The bias term in the decision function, allowing adjustments independently from the input features. Initially set to None.

    Methods:
        fit(X, y): Fits the Perceptron model to binary classified data.
        predict(X): Predicts the class label of samples based on the fitted model.
        plot_decision_boundary(X, y): Plots the decision boundary of the model along with the classified points for visualization.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the Perceptron with specified learning rate and number of iterations.

        Parameters:
            learning_rate (float): The learning rate for adjusting weights.
            n_iterations (int): The number of iterations to train the model.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the Perceptron model to the training data.

        Parameters:
            X (numpy.ndarray): The training samples of shape (n_samples, n_features).
            y (numpy.ndarray): The target values (class labels) of shape (n_samples,).

        This method updates the model's weights and bias based on the training data to find a decision boundary that best separates the classes.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training algorithm
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                prediction = np.where(linear_output >= 0, 1, -1)
                update = self.learning_rate * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predicts the class label of samples based on the fitted model.

        Parameters:
            X (numpy.ndarray): The samples to predict, of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The predicted class labels for each sample in X.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

    def plot_decision_boundary(self, X, y):
        """
        Plots the decision boundary of the model along with the classified points.

        Parameters:
            X (numpy.ndarray): The samples, of shape (n_samples, n_features).
            y (numpy.ndarray): The true class labels for the samples in X.

        This method visualizes how the model has learned to separate the two classes using a decision boundary.
        """
        x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2 = -(self.bias + self.weights[0] * x1) / self.weights[1]
        plt.scatter(X[:, 0], X[:, 1], c=['blue' if label > 0 else 'red' for label in y])
        plt.plot(x1, x2, 'r')
        plt.show()

