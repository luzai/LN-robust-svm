from lz import *

logging.root.setLevel(logging.ERROR)
from data import X, y
import svm

trainer = svm.SVMTrainer('linear', 1.)
predictor = trainer.train(X, y)

svs = predictor._support_vectors

plt.figure()
for i in np.unique(y):
    X_a = X[y == i]
    plt.plot(X_a[:, 0], X_a[:, 1], 'o')

for sv in svs:
    plt.scatter(sv[0], sv[1], s=85, facecolors='none', edgecolors='black')
# plt.show()

x_min, y_min = X.min(axis=0) - 0.2
x_max, y_max = X.max(axis=0) + 0.2
grid_size = 999
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
plt.show()
