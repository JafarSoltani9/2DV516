import numpy as np
import matplotlib.pyplot as plt
import operator


def plot_training_test(trainingData, testData):
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title("Training set")
    plt.plot(trainingData[:, 0], trainingData[:, 1], 'bo', markersize=3)
    plt.subplot(1, 2, 2)
    plt.title("Test set")
    plt.plot(testData[:, 0], testData[:, 1], 'ro', markersize=3)


# Knn function
def knn(data_set, x_val, k):
    distance = {}
    for x in range(len(data_set)):
        dis = (pow((x_val - data_set[x][0]), 2))  # Euclidean distance
        distance[x] = np.sqrt(dis)

    distance = sorted(distance.items(), key=operator.itemgetter(1))

    # Fetch knn
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    sum_values = sum(data_set[neighbors[x]][1] for x in range(len(neighbors)))
    return sum_values / k


def compute_test_mse(training_data, test_data):
    mse_results = {}
    for k in [1, 3, 5, 7, 9, 11]:
        predict_test = np.array([knn(training_data, x[0], k) for x in test_data])
        mse_test = np.mean(np.square(predict_test - test_data[:, 1]))
        mse_results[k] = mse_test
    return mse_results


def knn_regression_plot(training_data, test_data):
    plt.figure(figsize=(13, 7))
    figure_number = 0
    for k in [1, 3, 5, 7, 9, 11]:
        figure_number += 1
        xy = []
        for x in np.arange(0, 25, 0.25):
            y = knn(training_data, x, k)
            xy.append([x, y])
        regression = np.asarray(xy)

        predict_one = []
        for x in training_data:
            result = knn(training_data, x[0], k)
            predict_one.append([x[0], result])
        training_predicted = np.asarray(predict_one)

        predict_two = []
        for x in test_data:
            result = knn(test_data, x[0], k)
            predict_two.append([x[0], result])
        test_predicted = np.asarray(predict_two)

        mse_training = np.mean((training_predicted[:, 1]-training_data[:, 1])**2)
        mse_test = np.mean((test_predicted[:, 1] - test_data[:, 1])**2)
        mse_results = compute_test_mse(training_data, test_data)


        plt.subplot(2, 3, figure_number)
        plt.title(f"k = {k}, MSE = {round(mse_test, 2)}")
        plt.plot(training_data[:, 0], training_data[:, 1], "bo", markersize=2)
        plt.plot(regression[:, 0], regression[:, 1], "r-", linewidth=2)


        # Print MSE Error for each k
    for k, mse in mse_results.items():
        print(f"k = {k}: MSE = {mse:.2f}")
        print("-" * 30)



def best_k():
    print('\n\nTask 5\n')
    print('The lowest MSE is achieved when k =9, with an MSE of 27.70. Therefore, based on this criterion,\n' +
            'the best k is 9 as it yields the model that best generalizes to unseen data, according to the provided test MSE values.')

def main():
    # Load data
    data = np.genfromtxt('A1_datasets/polynomial200.csv', delimiter=',')
    # Divide data  training- and test data
    training_data = np.array(data[:100])
    test_data = np.array(data[100:])

    # Plotting the training- and test data
    plot_training_test(training_data, test_data)
    knn_regression_plot(training_data, test_data)
    best_k()
    plt.show()



main()
