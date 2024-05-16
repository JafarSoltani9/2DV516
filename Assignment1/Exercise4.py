from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt('A1_datasets/microchips.csv', delimiter=',')

unknowChips = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])
g_Size = 100
knn_clf = KNeighborsClassifier()


def generate_grid(x_axis, y_axis):
    xx, yy = np.meshgrid(x_axis, y_axis)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return knn_clf.predict(cells).reshape(g_Size, g_Size)

def trainError(k, X, y):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    errors = np.sum(predictions != y)
    return errors

x, y = data[:, :2], data[:, 2]
min_x, max_x, min_y, max_y = min(x[:, 0]), max(x[:, 0]), min(x[:, 1]), max(x[:, 1])
x_x, y_y = np.linspace(min_x, max_x, g_Size), np.linspace(min_y, max_y, g_Size)

plt.figure(figsize=(8, 8))
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title("Original data")
plt.plot(unknowChips[:, 0], unknowChips[:, 1], 'ro', markersize=10)

cmap_bold = ListedColormap(['red', 'green', 'blue'])

fig = plt.figure(figsize=(8, 8))
k_num = {1, 3, 5, 7}
for i, k in enumerate(k_num):
    knn_clf.set_params(n_neighbors=k)
    knn_clf.fit(x, y)
    predict = knn_clf.predict(unknowChips)
    grid = generate_grid(x_x, y_y)
    counter = trainError(k, x, y)
    fig.tight_layout()

    sp = fig.add_subplot(2, 2, 1 + i)
    sp.pcolormesh(x_x, y_y, grid, cmap=ListedColormap(['#c1edb7', '#fff']))
    sp.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', marker='o')

    sp.imshow(grid, origin='lower', extent=(min_x, max_x, min_y, max_y))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], marker='.', cmap=cmap_bold)
    sp.set_title(f"k = {k}, training errors = {counter}")
    print('\n' + 'k = ' + str(k))
    for m in range(len(predict)):
        message = ''
        if (predict[m] == 1):
            message = 'OK'
        else:
            message = 'Fail'
        print('chip ' + str(unknowChips[m]) + ' : ' + str(predict[m]) + ' ==> ' + message)

plt.show()
