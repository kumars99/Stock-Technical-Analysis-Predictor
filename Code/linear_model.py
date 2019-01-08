import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL2(logReg):
    # Logistic Regression
    def __init__(self, verbose=0, L2_lambda=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = L2_lambda
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy/2 * np.linalg.norm(w) ** 2

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

class logRegL1(logReg):
    # Logistic Regression
    def __init__(self, verbose=0, L1_lambda=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = L1_lambda
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                        self.maxEvals, X, y, verbose=self.verbose)

class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj, np.zeros(len(ind)), self.maxEvals, X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the selected set
                # then compute the loss and update the minLoss/bestFeature
                w = np.zeros(d)
                w[list(selected_new)], _ = minimize(list(selected_new))

                loss = self.funObj(w, X, y)[0] + self.L0_lambda * np.count_nonzero(w)
                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))

class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)


    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class logLinearClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i], _ = findMin.findMin(self.funObj, self.W[i],
                                           self.maxEvals, X, ytmp, verbose=self.verbose)

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class softmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            # solve the normal equations
            self.W[i], _ = findMin.findMin(self.funObj, self.W.flatten(),
                                           self.maxEvals, X, y, verbose=self.verbose)

    def funObj(self, w, X, y):
        n, d = X.shape
        w = np.reshape(w, [self.n_classes, d])

        # Calculate the function value
        f = 0
        for i in range(n):
            ytmp = y[i]     # The class
            Xtmp = X[i]     # Entry of X, 1 x 3
            ftmp = Xtmp.dot(w[ytmp]) + np.log(np.sum(np.exp(Xtmp @ w.T)))
            f = f - ftmp

        # Calculate the gradient value
        g = np.zeros((self.n_classes, d))
        for c in range(self.n_classes):
            classtmp = np.zeros((n, d))

            for i in range(n):
                Xtmp = X[i]

                top = np.exp(Xtmp.dot(w[c])) * Xtmp
                bot = np.sum(np.exp(Xtmp.dot(w.T)))
                classtmp[i] = np.divide(top,bot) - Xtmp * (y[i] == c)

            g[c] = np.sum(classtmp)

        g = np.ndarray.flatten(g)
        return f, g

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)