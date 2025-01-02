import numpy as np

#%% To do list
# Annotate

#%% Model

class LinearRegression:
    # constructor
    def __init__(self, tol = 1e-4, no_iter = 10000, learning_rate = 1e-1):
        self.learning_rate = learning_rate
        self.tol = tol
        self.no_iter = no_iter
        self.weights_ = None
        self.tol_final = None
        self.loss = None

    # fitting model
    def fit(self, X, y):
        # Adding an extra column to X
        X = np.column_stack((np.ones((X.shape[0],1)), X))
        w = np.zeros((X.shape[1],1))
        N = X.shape[0]
        old_loss = 0

        # Iterative optimization
        for i in range(self.no_iter):
            yhat = np.dot(X,w)
            loss = (1/(2*N))*np.sum((yhat-y)**2)
            dloss = (1/N)*np.dot(X.T,yhat-y)
            w -= self.learning_rate*dloss
            tol = abs(loss - old_loss)
            if tol < self.tol:
                print(f'Exceeded tolerance after {i} iterations.')
                break

        # Saving optimal weights
        self.weights_ = w
        self.loss = loss
        self.tol_final = tol

    # predicting using the learned model
    def predict(self, X):
        X = np.column_stack((np.ones((X.shape[0], 1)), X))
        yhat = np.dot(X,self.weights_)
        return yhat