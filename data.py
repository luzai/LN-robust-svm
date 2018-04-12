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
    logging.info(f'seed is {seed}')
    np.random.seed(seed)
    return X, y


def get_sonar_data():
    y = []
    X = np.zeros((208, 60))
    for ind_i, line in enumerate(open('sonar_scale', 'r')):
        cls = line.split()[0]
        cls = float(cls)
        feas = line.split()[1:]
        for fea in feas:
            ind, num = fea.split(':')
            ind = int(ind) - 1
            num = float(num)
            X[ind_i, ind] = num
        y.append(cls)
    X, y = np.asarray(X), np.asarray(y)

    shuffle_ind = np.random.permutation(208)
    X = X[shuffle_ind]
    y = y[shuffle_ind]

    return X, y


def split_train_test(X, y, ratio=.3):
    n_samples = X.shape[0]
    shuffle_ind = np.random.permutation(n_samples)
    X = X[shuffle_ind]
    y = y[shuffle_ind]
    split = int(n_samples * ratio)
    return X[split:], y[split:], X[:split], y[:split]


def svm_plot(X, y):
    y = np.asarray(y, dtype=float).reshape(-1)
    for i in np.unique(y):
        X_a = X[y == i]
        plt.plot(X_a[:, 0], X_a[:, 1], 'o')


def boundary_plot(X, predictor, grid_size=99):
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

    result = predictor.predict(np.stack((xx, yy), axis=0).T)

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx.reshape(grid_size, grid_size), yy.reshape(grid_size, grid_size), Z.reshape(grid_size, grid_size),
                 # cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.1)
    if predictor._kernel == 'linear':
        weight = (alphas * predictor._support_vector_labels).reshape(-1, 1) * predictor._support_vectors
        weight = weight.sum(axis=0)
        b = predictor._bias
        xx = np.linspace(x_min, x_max, 100)
        w1, w2 = weight
        yy = - w1 / w2 * xx - b / w2
        plt.plot(xx, yy)
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])


def calc_error(s, y):
    s = np.sign(s)
    y = np.sign(y)

    return np.count_nonzero(s != y) / s.shape[0]


def normalize(s):
    s = s.copy()
    # s -= s.min()
    s /= s.max()
    return s


def get_adv_data(n_samples=100, seed=16, C=1., R=25, beta1=0.1, beta2=0.1, L=10, **kwargs):
    X, y = get_toy_data(n_samples=n_samples, seed=seed, )
    trainer = svm.SVMTrainer('linear', C)
    predictor = trainer.train(X, y, remove_zero=False)
    alpha, b = predictor._weights, predictor._bias

    s = predictor.score(X)
    s *= predictor._support_vector_labels
    s = normalize(s)

    # svm_plot(X, y)
    # boundary_plot(X, predictor)
    # plt.show()

    error_hist = []
    yp_hist = []
    flip_pnts_hist = []
    print('training error on untainted data is ', calc_error(s, y))

    for i in range(R):
        alpha_rnd = np.random.uniform(-C, C, size=alpha.shape)
        b_rnd = np.random.uniform(-C, C)

        predictor_rnd = svm.SVMPredictor(
            weights=alpha_rnd,
            support_vectors=predictor._support_vectors,
            support_vector_labels=predictor._support_vector_labels,
            bias=b_rnd,
            sigma=predictor._sigma,
            kernel=predictor._kernel
        )
        # weight = (alpha_rnd * predictor_rnd._support_vector_labels).reshape(-1, 1) * predictor._support_vectors
        # weight = weight.sum(axis=0)
        print('training error of random svm is ', calc_error(predictor_rnd.predict(X), y))
        # svm_plot(X, y)
        # boundary_plot(X, predictor_rnd)
        # plt.show()

        q = predictor_rnd.score(X)
        q *= predictor._support_vector_labels
        q = normalize(q)

        v = alpha / C - beta1 * s - beta2 * q
        # plt.figure()
        # plt.plot(alpha,'o')
        # plt.plot(s,'x')
        # plt.plot(q,'.')
        # plt.legend()
        # plt.show()
        k = np.argsort(v, axis=0)
        y_p = y.copy()
        y_p[k[0:L]] *= -1

        predictor_new = trainer.train(X, y_p)
        print('training error on tainted data  is ', calc_error(predictor_new.predict(X), y))
        # plt.figure()
        # svm_plot(X, y_p)
        # boundary_plot(X, predictor_new)
        flip_pnts = X[k[0:L]]
        # plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
        # plt.show()
        error_hist.append(calc_error(predictor_new.predict(X), y))
        yp_hist.append(y_p)
        flip_pnts_hist.append(flip_pnts)

    print(np.max(error_hist), error_hist)
    y_p = yp_hist[np.argmax(error_hist)]
    flip_pnts = flip_pnts_hist[np.argmax(error_hist)]
    return X, y_p, flip_pnts


def get_rand_data(n_samples=100, seed=16, L=10, **kwargs):
    X, y = get_toy_data(n_samples=n_samples, seed=seed)
    X, y_p, flip_pnts = apply_rand_flip(X, y, L)
    return X, y_p, flip_pnts


def apply_rand_flip(X, y, L=10):
    n_samples = X.shape[0]
    flip_ind = np.random.permutation(n_samples)[:L]
    y_p = y.copy()
    y_p[flip_ind] *= -1
    flip_pnts = X[flip_ind]
    return X, y_p, flip_pnts


def get_2d_intuition_data(n_samples=100, seed=10, L=10, C=1., **kwargs):
    X, y = get_toy_data(n_samples=n_samples, seed=seed)
    trainer = svm.SVMTrainer('linear', C)
    predictor = trainer.train(X, y, remove_zero=False)
    alpha, b = predictor._weights, predictor._bias

    weight = (alpha * predictor._support_vector_labels).reshape(-1, 1) * predictor._support_vectors
    weight = weight.sum(axis=0)
    y_p = y.copy()
    dist = np.abs(np.dot(X, weight.T) + b)
    flip_inds = np.argsort(dist)[::-1][:L]
    flip_pnts = X[flip_inds]
    y_p[flip_inds] *= -1
    # plt.figure()
    # svm_plot(X, y_p)
    # boundary_plot(X, predictor)
    # plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    # plt.show()
    return X, y_p, flip_pnts


if __name__ == '__main__':
    # X, y_p, flip_pnts = get_2d_intuition_data(n_samples=100, seed=16)
    get_sonar_data()
