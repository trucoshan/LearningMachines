import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Predictor:

    def __init__(self, lr = 0.01, n_iters=1000):

        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None
        

    def model(self, X, y):
        
        n,m = X.shape

        self.W = np.zeros(m)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.W) + self.b

            dj_dw = (1/n) * np.dot(X.T, (y_pred - y))

            dj_db = (1/n) * np.sum(y_pred-y)

            self.W = self.W - self.lr * dj_dw
            self.b = self.b - self.lr * dj_db

    
    def predict(self, X):
        y_pred = np.dot(X, self.W) + self.b
        return y_pred
