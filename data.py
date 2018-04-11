from lz import *
np.random.seed(16)
n_samples = 100
X = np.concatenate((
    np.random.multivariate_normal([1, 0], [[.5, 0], [0, .5]], n_samples // 2),
    np.random.multivariate_normal([-1, 0], [[.5, 0], [0, .5]], n_samples // 2)
), axis=0)

y = np.concatenate((
    np.ones(n_samples // 2), np.ones(n_samples // 2) * -1
), axis=0)

shuffle_ind = np.random.permutation(n_samples)

X = X[shuffle_ind]
y = y[shuffle_ind]

def svm_plot(X, y):
    y = np.asarray(y, dtype=float).reshape(-1)
    plt.figure()

    for i in np.unique(y):
        X_a = X[y == i]
        plt.plot(X_a[:, 0], X_a[:, 1], 'o')
    plt.show()


if __name__ == '__main__':
    svm_plot(X, y)

