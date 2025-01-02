import numpy as np
from scipy.stats import multivariate_normal

#%% To do list:
# Implement multivariate normal function
# Annotate

#%% Model

class QDA:

    def __init__(self):
        self.mu_ = None
        self.covariance_matrix_ = None
        self.pi_ = None
        self.k_ = None
        self.posteriors_ = None

    def fit(self, X, y):
        class_index = np.unique(y)
        k = len(class_index)
        n, p = X.shape

        mu_hat = []
        sigma_hat = []
        pi_k = []

        for i in range(k):
            args = np.where(y == class_index[i])[0]
            mu_hat.append(np.mean(X[args,:], axis = 0))
            sigma_hat.append(np.cov(X[args,:].T))
            pi_k.append(len(args)/n)

        self.mu_ = mu_hat
        self.covariance_matrix_ = sigma_hat
        self.pi_ = pi_k
        self.k_ = k

    def predict(self, X):
        n = X.shape[0]
        posteriors = np.zeros((n,self.k_))

        for i in range(n):
            for j in range(self.k_):
                prior = self.pi_[j]
                likelihood = multivariate_normal.pdf(X[i,:], mean = self.mu_[j], cov = self.covariance_matrix_[j])
                marginal_likelihood = np.sum([self.pi_[l]*multivariate_normal.pdf(X[i,:], mean = self.mu_[l], cov = self.covariance_matrix_[l]) for l in range(self.k_)])
                posteriors[i, j] = likelihood*prior/marginal_likelihood

        self.posteriors_ = posteriors
        y_predict = np.argmax(posteriors, axis = 1)
        return y_predict