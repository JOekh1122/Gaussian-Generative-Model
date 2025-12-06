import numpy as np

class GaussianGenerativeModel:
    def __init__(self, lambda_reg=1e-3):
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes = classes
        K = len(classes)

        #hena ha7sb el prioes kol class(b7wsb how many samples fe kol class)
        self.priors = np.zeros(K)
        for k in classes:
            self.priors[k] = np.sum(y == k) / n_samples

        #hena ha7sb el mean kol class(array of means for each class) 
        self.means = np.zeros((K, n_features))
        for k in classes:
            self.means[k] = X[y == k].mean(axis=0)

        #hena ha7sb el Shared covariance (cov=1/n * sum((x_i - mu_k)(x_i - mu_k)^T) for all samples)
        cov = np.zeros((n_features, n_features))
        for i in range(n_samples):
            k = y[i]
            minus = (X[i] - self.means[k]).reshape(-1, 1)
            cov += minus @ minus.T

        cov /= n_samples

        #hena Regularize the covariance to make it invertible
        self.cov = cov + self.lambda_reg * np.eye(n_features)

        self.inv_cov = np.linalg.inv(self.cov) # inverse covariance matrix
        self.log_det = np.log(np.linalg.det(self.cov)) # log determinant of covariance matrix to aviod underflow or overflow


#score=-0.5*(x-mu_k)^T * inv_cov * (x-mu_k) -0.5*log_det + log(prior_k) 
    def _score(self, x, k):
        diff = x - self.means[k]
        term1 = -0.5 * diff.T @ self.inv_cov @ diff # -0.5*(x-mu_k)^T * inv_cov * (x-mu_k)__>distance from mean
        term2 = -0.5 * self.log_det # -0.5*log_det
        term3 = np.log(self.priors[k]) # log(prior_k)
        return term1 + term2 + term3

    def predict(self, X):
        preds = []
        for x in X:
            scores = [self._score(x, k) for k in self.classes]
            preds.append(np.argmax(scores))
        return np.array(preds)
