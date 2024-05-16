import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load and shuffle the dataset
data = np.loadtxt("artificial.csv", delimiter=",")
np.random.shuffle(data)
X = data[:, :-1]
y = data[:, -1]

# Initialize random number generator
rng = np.random.default_rng()

# Define dynamic sizes for training and testing sets based on the dataset's size
total_samples = len(X)
train_size = min(9000, int(0.9 * total_samples))
test_size = min(1000, total_samples - train_size)

# Split the data into training and testing sets
X_train, X_test = X[:train_size], X[train_size:train_size + test_size]
y_train, y_test = y[:train_size], y[train_size:train_size + test_size]

# Calculate min and max values with margin for the first two features
margin = 0.3
grid_size = 300

min_vals = np.min(X[:, :2], axis=0) - margin
max_vals = np.max(X[:, :2], axis=0) + margin

x_min, y_min = min_vals
x_max, y_max = max_vals

# Generate grid for decision boundary visualization
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                     np.linspace(y_min, y_max, grid_size))
grid = np.c_[xx.ravel(), yy.ravel()]

predictedS = np.zeros_like(y_test, dtype=float)
boundaryS = np.zeros(grid_size ** 2, dtype=float)
errors = []

# Train and visualize 100 decision trees
plt.figure("The decision boundaries of all the models", figsize=(12, 7))
for i in range(100):
    clf = DecisionTreeClassifier()
    r = rng.choice(train_size, size=5000, replace=True)  # Sample 5000 training samples with replacement
    XX, yy_bootstrap = X_train[r], y_train[r]
    clf.fit(XX, yy_bootstrap)
    
    y_predict = clf.predict(X_test)
    predictedS += y_predict
    error = np.mean(y_test != y_predict)
    errors.append(error)
    
    grid_pred = clf.predict(grid)
    boundaryS += grid_pred
    
    if i < 99:  # Plot the first 99 decision trees
        plt.subplot(10, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.contour(xx, yy, grid_pred.reshape(xx.shape), colors="black")

# Plot the ensemble model's decision boundary in the 100th subplot position
plt.subplot(10, 10, 100)
plt.xticks([])
plt.yticks([])
plt.contour(xx, yy, (boundaryS > 50).reshape(xx.shape), colors="red")

# Calculate and print generalization errors
ensemble_predictions = predictedS > 50
general_err = round((1.0 - np.mean(y_test == ensemble_predictions)) * 100, 2)
general_err_ind = round(np.mean(errors) * 100, 2)

print(f"\n\na) The estimate of the generalization error using the test set of the ensemble of 100 decision trees = {general_err} %\n")
print(f"b) The average estimated generalization error of the individual decision trees = {general_err_ind} %\n")
print(f"d) Using an ensemble of trees can reduce generalization errors on the test set because it combines multiple different models, leading to better predictions. However, this approach typically requires more time to execute.\n")

# Show plots
plt.show()
