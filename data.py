from lz import *
import svm

# %load_ext autoreload
# %autoreload 2
import matplotlib

matplotlib.style.use('ggplot')
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

logging.root.setLevel(logging.ERROR)


def get_toy_data(n_samples=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

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

    seed = int(time.time() * 1000) % 100
    logging.error(f'seed is {seed}')
    np.random.seed(seed)
    return X, y


def svm_plot(X, y):
    y = np.asarray(y, dtype=float).reshape(-1)
    for i in np.unique(y):
        X_a = X[y == i]
        plt.plot(X_a[:, 0], X_a[:, 1], 'o')


def boundary_plot(X, predictor, grid_size=999):
    svs = predictor._support_vectors
    alphas = predictor._weights
    support_vector_indices = alphas > 1e-5
    # alphas = alphas[support_vector_indices]
    svs = svs[support_vector_indices]
    plt.scatter(svs[:, 0], svs[:, 1], s=85, facecolors='none', edgecolors='black')

    x_min, y_min = X.min(axis=0) - 0.2
    x_max, y_max = X.max(axis=0) + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    xx = xx.ravel()
    yy = yy.ravel()

    # result = [predictor.predict(xy) for xy in np.stack((xx, yy), axis=0).T]
    result = predictor.predict(np.stack((xx, yy), axis=0).T)

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx.reshape(grid_size, grid_size), yy.reshape(grid_size, grid_size), Z.reshape(grid_size, grid_size),
                 # cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.1)


def calc_error(s, y):
    s = np.sign(s)
    y = np.sign(y)

    return np.count_nonzero(s != y) / s.shape[0]


def normalize(s):
    s = s.copy()
    s -= s.min()
    s /= s.max()
    return s


C = 1.
R = 3
X, y = get_toy_data(n_samples=100, seed=16)
trainer = svm.SVMTrainer('linear', C)
predictor = trainer.train(X, y, remove_zero=False)
alpha, b = predictor._weights, predictor._bias

s = predictor.score(X)

svm_plot(X, y)
boundary_plot(X, predictor)
plt.show()

error_hist = [calc_error(s, y)]
print('training error on untainted data is ', calc_error(s, y))

s = normalize(s)

for i in range(R):
    alpha_rnd = np.random.uniform(0, C, size=alpha.shape)
    b_rnd = np.random.uniform(-C, C)
    # b_rnd = np.random.uniform(-.2, .2)
    # b_rnd = 0

    predictor_rnd = svm.SVMPredictor(
        weights=alpha_rnd,
        support_vectors=predictor._support_vectors,
        support_vector_labels=predictor._support_vector_labels,
        bias=b_rnd,
        sigma=predictor._sigma,
        kernel=predictor._kernel
    )
    weight = (alpha_rnd * predictor_rnd._support_vector_labels).reshape(-1, 1) * predictor._support_vectors
    weight = weight.sum(axis=0)
    print('training error of random svm is ', calc_error(predictor_rnd.predict(X), y))
    # svm_plot(X, y)
    # boundary_plot(X, predictor_rnd)
    # plt.show()

    q = predictor_rnd.score(X)
    q = normalize(q)
    beta1 = beta2 = 0.1
    v = alpha / C - np.abs(beta1 * s - beta2 * q)
    # plt.figure()
    # plt.plot(alpha,'o')
    # plt.plot(s,'x')
    # plt.plot(q,'.')
    # plt.legend()
    # plt.show()
    k = np.argsort(v, axis=0)
    y_p = y.copy()
    L = 10
    y_p[k[1:L]] *= -1

    predictor_new = trainer.train(X, y_p)
    print('training error on tainted data  is ', calc_error(predictor_new.predict(X), y))
    plt.figure()
    svm_plot(X, y_p)
    boundary_plot(X, predictor_new)
    flip_pnts = X[k[1:L]]
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.show()
    error_hist.append(calc_error(predictor_new.predict(X), y))

print(np.max(error_hist), error_hist)
