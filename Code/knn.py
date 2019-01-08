import numpy as np
from scipy import stats
import utils
from utils import euclidean_dist_squared as e_dist

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y


    def predict(self, Xtest):
        T, D = Xtest.shape
        dist = e_dist(self.X, Xtest)

        y_pred = np.zeros(T)

        # Iterate through each X-test point
        for t in range(T):
            temp = np.argsort(dist[:, t])
            # knn is the indices of the nearest neighbor
            knn_indices = temp[0:self.k]
            knn = self.y[knn_indices]
            yhat = stats.mode(knn)[0][0]
            y_pred[t] = yhat

        return y_pred
