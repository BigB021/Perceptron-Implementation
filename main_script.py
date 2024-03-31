# Author : Youssef Aitbouddroub
# tesing perceptron implementation

import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

def generate_data(nb):
    '''Generate linearly separable dataset'''
    X1_1 = np.random.randn(nb//2, 2) + np.array([2, 2])
    X1_2 = np.random.randn(nb//2, 2) + np.array([-2, -2])
    X = np.vstack((X1_1, X1_2))
    y = np.hstack((np.ones(nb//2), -1*np.ones(nb//2)))
    return X, y

if __name__ == "__main__":
    nb = 500  # Number of samples
    X, y = generate_data(nb)
    
    p = Perceptron(learning_rate=0.01, n_iterations=1000)
    p.fit(X, y)
    p.plot_decision_boundary(X, y)
