import numpy as np
import operator
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# plot the ok and fail data
def plot_data(data, chips):
    ok_arr = data[data[:, 2] == 1]
    fail_arr = data[data[:, 2] == 0]
    plt.figure(1, figsize=(7, 5))
    plt.title("Original figure")
    plt.plot()
    plt.plot(chips[:, 0], chips[:, 1], 'ro', markersize=10)
    plt.plot(ok_arr[:, 0], ok_arr[:, 1], 'go', markersize=5) # green is ok data in the plot
    plt.plot(fail_arr[:, 0], fail_arr[:, 1], 'bo', markersize=5) # fail is fail data in the plot


def knn(trainset, test, k):
    dis = {}
    for x in range(len(trainset)):
        distance = (pow((test[0] - trainset[x][0]), 2)) + pow((test[1] - trainset[x][1]), 2) # Euclidean distance
        dis[x] = np.sqrt(distance)

    distances = sorted(dis.items(), key=operator.itemgetter(1))

    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])

    ok = 0
    fail = 0

    for x in range (len(neighbors)):
        response = trainset[neighbors[x]][2]
        if(response == 1 ):
            ok = ok + 1
        else:
            fail = fail + 1

    if(ok > fail):
        return 1
    else:
        return 0

def predict_chips(data, chips):
    for k in [1,3,5,7]:
        print('k = %i' % + k)
        message = ''
        for x in range(len(chips)):
            result = knn(data, chips[x], k)
            if (result == 1):
                message = 'OK'
            else:
                message = 'Fail'
            print(' chip ' + str((x + 1)) + ': ' + str(chips[x]) + ' ==> ' + message)


def plot_errors(data, k_values):
    x_minimum, x_maximum = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_minimum, y_maximum = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    x_x, y_y = np.meshgrid(np.arange(x_minimum, x_maximum, 0.01),
                           np.arange(y_minimum, y_maximum, 0.01))
    x_y_mesh = np.c_[x_x.ravel(), y_y.ravel()]

    # Plot colors
    cmap_light = ListedColormap(['lightgray', 'lightgreen', 'cyan'])
    cmap_bold = ListedColormap(['red', 'green', 'blue'])

    plt.figure(figsize=(9, 7))
    figure_number = 0

    for k in k_values:
        figure_number += 1
        plt.subplot(2, 2, figure_number)
        counter = trainError(k, data, data[:, 2])
        plt.title(f"k={k}, training errors = {counter}")
        classes = [knn(data, x, k) for x in x_y_mesh]

        clMesh = np.array(classes).reshape(x_x.shape)
        plt.pcolormesh(x_x, y_y, clMesh, cmap=cmap_light)
        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], marker='.', cmap=cmap_bold)
    

def trainError(k, data, y):
    counter = 0
    trainSet = data
    y_array = np.asarray(y)

    index = 0
    for x in data:
        distances = {}
        for y in range(len(trainSet)):
            # Calculate distance for every x
            dist =(pow((x[0] - trainSet[y][0]), 2)) + pow((x[1] - trainSet[y][1]),2)# Euclidean distance
            distances[y] = np.sqrt(dist)

        # Sort distances
        asc_distances = sorted(distances.items(), key=operator.itemgetter(1))

        # Fetch  KNN
        neighbors = []
        for x in range(k):
            neighbors.append(asc_distances[x][0])

        ok = 0
        fail = 0
        for x in range(len(neighbors)):
            response = trainSet[neighbors[x]][2]
            if (response == 1):
                ok = ok + 1
            else:
                fail = fail + 1

        
        if (ok > fail):
            class_result = 1
        else:
            class_result = 0

        # Increase error
        if (y_array[index] != class_result):
            counter = counter + 1

        index = index + 1

    return counter


def main():
    # Load data
    data = np.genfromtxt('A1_datasets/microchips.csv', delimiter=',')
    
    chips = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])

    predict_chips(data, chips)
    plot_data(data, chips)
    plot_errors(data, [1, 3, 5, 7])
    plt.show()
main()

