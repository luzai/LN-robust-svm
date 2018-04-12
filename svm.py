import numpy as np
import cvxopt.solvers
import logging

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


def calc_distmat2(X, Y=None):
    if Y is None:
        Y = X
    distmat2 = (X ** 2).sum(axis=1).reshape(-1, 1) + \
               (Y ** 2).sum(axis=1).reshape(1, -1) - 2 * np.dot(X, Y.T)
    return distmat2


class SVMTrainer(object):
    def __init__(self, kernel='linear', c=1., sigma=1., ln_robust=False, mu=0.5):
        if kernel not in ['linear', 'rbf']:
            raise NotImplementedError('{} not implemented'.format(kernel))
        self._kernel = kernel
        self._c = c
        self._sigma = sigma
        self.ln_robust = ln_robust
        self.mu = mu

    def train(self, X, y, remove_zero=True):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers, remove_zero=remove_zero)

    def _gram_matrix(self, X):
        if self._kernel == 'linear':
            return np.dot(X, X.T)
        elif self._kernel == 'rbf':
            distmat = calc_distmat2(X, X)
            return np.exp(-  (distmat / (2 * self._sigma ** 2)))

    def _construct_predictor(self, X, y, lagrange_multipliers, remove_zero=True):
        if remove_zero:
            support_vector_indices = \
                lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

            support_multipliers = lagrange_multipliers[support_vector_indices]
            support_vectors = X[support_vector_indices]
            support_vector_labels = y[support_vector_indices]

        else:
            support_multipliers = lagrange_multipliers
            support_vectors = X
            support_vector_labels = y

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        svm_bias_zero = SVMPredictor(
            kernel=self._kernel,
            bias=0.0,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels,
        )
        bias = support_vector_labels - svm_bias_zero.predict(support_vectors)
        bias = bias.mean()
        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels,
            sigma=self._sigma,
        )

    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)

        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = (np.outer(y, y) * K)
        if self.ln_robust:
            S = 4 * self.mu * (1 - self.mu)
            M = np.ones_like(P) * (1 - S)
            M += np.identity(P.shape[0]) * S
            P *= M
            P = cvxopt.matrix(P)
        else:
            P = cvxopt.matrix(P)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(self,
                 weights,
                 support_vectors,
                 support_vector_labels,
                 bias,
                 sigma=1.,
                 kernel='linear',
                 ):
        self._sigma = sigma
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)
        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)

    def _gram_matrix(self, X, Y):

        if self._kernel == 'linear':
            return np.dot(X, Y.T)
        elif self._kernel == 'rbf':
            distmat = calc_distmat2(X, Y)
            return np.exp(- (distmat / (2 * self._sigma ** 2)))

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        score = self.score(x)
        return np.sign(score)

    def score(self, x):
        n_support_vectors, n_features = self._support_vectors.shape
        x = x.reshape(-1, n_features)
        res = (self._gram_matrix(x, self._support_vectors)
               * self._weights.reshape(-1, n_support_vectors)
               * self._support_vector_labels.reshape(-1, n_support_vectors)).sum(axis=1)
        res += self._bias
        return res

    def error(self, x, y):
        pred = self.predict(x)
        s = np.sign(pred)
        y = np.sign(y)
        return np.count_nonzero(s != y) / s.shape[0]
