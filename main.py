from lz import *

logging.root.setLevel(logging.ERROR)
import svm, data

# ori
n_samples = 100
seed = 16
C = 1.
X, y = data.get_toy_data(n_samples=n_samples, seed=seed)
trainer = svm.SVMTrainer('linear', C)
predictor = trainer.train(X, y, remove_zero=True)

plt.figure()
data.svm_plot(X, y)
data.boundary_plot(X, predictor)
plt.savefig('ori.png')
plt.show()


# tainted data
X, y_p, flip_pnts = data.get_2d_intuition_data(n_samples=n_samples, seed=seed, C=C)
trainer = svm.SVMTrainer('linear', C)
predictor = trainer.train(X, y_p, remove_zero=True)

plt.figure()
data.svm_plot(X, y_p)
data.boundary_plot(X, predictor)
plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
plt.savefig('ln.png')
plt.show()

trainer = svm.SVMTrainer('linear', C, ln_robust=True, mu=0.1)
predictor = trainer.train(X, y_p, remove_zero=True)

plt.figure()
data.svm_plot(X, y_p)
data.boundary_plot(X, predictor)
plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
plt.savefig('ln.robust.mu.0.1.png')
plt.show()

trainer = svm.SVMTrainer('linear', C, ln_robust=True, mu=0.5)
predictor = trainer.train(X, y_p, remove_zero=True)

plt.figure()
data.svm_plot(X, y_p)
data.boundary_plot(X, predictor)
plt.scatter(flip_pnts[:, 0], flip_pnts[:, 1], s=85 * 2, facecolors='none', edgecolors='green')
plt.savefig('ln.robust.mu.0.5.png')
plt.show()
