from lz import *
from data import X, y
from sklearn.svm import LinearSVC, SVC
from svm import *

clf = SVC(C=1., kernel='linear', verbose=True)
clf.fit(X, y)
svs = clf.support_vectors_
y = np.asarray(y, dtype=float).reshape(-1)

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

result = clf.predict(np.stack((xx, yy), axis=0).T)

# for (i, j) in itertools.product(range(grid_size), range(grid_size)):
#     point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
#     result.append(
#         clf.predict(point)
#     )

Z = np.array(result).reshape(xx.shape)

plt.contourf(xx.reshape(grid_size, grid_size), yy.reshape(grid_size, grid_size), Z.reshape(grid_size, grid_size),
             # cmap=cm.Paired,
             levels=[-0.001, 0.001],
             extend='both',
             alpha=0.1)
plt.show()


