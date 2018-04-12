# from lz import *
import logging, numpy as np, matplotlib.pyplot as plt
logging.root.setLevel(logging.ERROR)
import svm, data

exp = 'proc'
if exp == 'proc':
    C = 1.
    kernel = 'rbf'
    proc_train = np.loadtxt('proc-train', delimiter=' ')
    X, y = proc_train[:, 1:], proc_train[:, 0]
    X_test = np.loadtxt('proc-test', delimiter=' ')
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y, remove_zero=True)
    y_pred = predictor.predict(X_test)
    np.savetxt('y-pred', y_pred, delimiter=' ')

elif exp == 'sonar':
    n_samples = 208
    n_features = 60
    C = 1.
    L = 208 // 10
    kernel = 'rbf'

    X, y = data.get_sonar_data()
    X, y, X_val, y_val = data.split_train_test(X, y)
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y, remove_zero=True)
    print(predictor.error(X_val, y_val))

    X, y, _ = data.apply_rand_flip(X, y, L)

    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y, remove_zero=True)
    print(predictor.error(X_val, y_val))

    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.1)
    predictor = trainer.train(X, y, remove_zero=True)
    print(predictor.error(X_val, y_val))


elif exp == 'toy':

    # ori
    n_samples = 100
    kernel = 'linear'
    seed = 16
    C = 1.
    R = 30
    L = 100 // 10
    beta1 = beta2 = 0.1
    X, y = data.get_toy_data(n_samples=n_samples, seed=seed)
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y)
    data.boundary_plot(X, predictor)
    plt.savefig('ori.png')
    plt.show()

    # tainted data
    X, y_p, flip_pnts = data.get_adv_data(n_samples=n_samples, seed=seed, C=C, R=R, L=L, beta1=beta1, beta2=beta2)
    trainer = svm.SVMTrainer(kernel, C)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.png')
    plt.show()

    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.1)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.robust.mu.0.1.png')
    plt.show()

    trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=0.5)
    predictor = trainer.train(X, y_p, remove_zero=True)

    plt.figure()
    data.svm_plot(X, y_p)
    data.boundary_plot(X, predictor)
    plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
    plt.savefig('ln.robust.mu.0.5.png')
    plt.show()
