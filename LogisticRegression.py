import numpy as np

#%% To do list
# Annotate

#%% Model

class LogisticRegression:
    # Initialization by __init__ transformer, with default hyperparameters
    def __init__(self, no_iter = None, learning_rate = 0.01, tol = 1e-1):
        self.no_iter = no_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.weights = None
        self.nll = None
        self.nll_logs = None

    def _logistic_function(self, X, theta):
        return 1 / (1 + np.exp(-(np.dot(X, theta))))

    def _nll(self,y_hat, y):
        return -np.nansum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        nll = self.tol
        nll_logs = []
        theta = np.zeros((X.shape[1],))

        if self.no_iter is not None:
            for i in range(self.no_iter):
                y_hat = self._logistic_function(X, theta)
                nll = self._nll(y_hat,y)
                nll_logs.append(nll)
                G = np.dot(X.T, (y_hat - y))
                theta -= self.learning_rate * G
                self.tol = nll
        else:
            i = 0
            tol = self.tol
            while(tol >= self.tol):
                y_hat = self._logistic_function(X, theta)
                nll_old = nll+0
                nll = self._nll(y_hat,y)
                nll_logs.append(nll)
                G = np.dot(X.T, (y_hat - y))
                theta -= self.learning_rate * G
                tol = abs((nll_old-nll)/nll_old)
                i += 1

        self.weights = theta
        self.nll = nll
        self.nll_logs = nll_logs

    def _predict_probas(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        probabilities = self._logistic_function(X, self.weights)
        return probabilities

    def predict(self, X, cutoff = 0.5):
        probabilities = self._predict_probas(X)
        prediction = (probabilities > cutoff) + 0
        return prediction
